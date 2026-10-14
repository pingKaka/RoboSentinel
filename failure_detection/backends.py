"""Local Transformers inference with independent SAM video sessions per camera."""
from functools import lru_cache
import json
from threading import RLock

import numpy as np
from PIL import Image

from .pipeline.contracts import PerceptionUpdate, TrackState

_LOCK = RLock()


def rgb(image):
    # Transport uses OpenCV BGR; all model processors expect RGB.
    return Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("VLM output must be a JSON object")
    return value


class QwenBackend:
    def __init__(self, path, quantize, max_new_tokens=768, image_max_side=768):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("The deployment backend requires a CUDA GPU")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.max_new_tokens, self.image_max_side = int(max_new_tokens), int(image_max_side)
        if min(self.max_new_tokens, self.image_max_side) < 1:
            raise ValueError("VLM token and image limits must be positive")
        kwargs = dict(local_files_only=True, device_map="cuda:0", torch_dtype=torch.bfloat16)
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, **kwargs).eval()

    def infer(self, frames, instruction, evidence):
        import torch
        content = []
        for camera, frame in frames.items():
            image = rgb(frame)
            image.thumbnail((self.image_max_side, self.image_max_side))
            content.extend([{"type": "text", "text": camera}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": instruction + "\nEvidence: " + json.dumps(evidence)})
        with _LOCK, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse_json(text)

    def plan_tracking_targets(self, payload):
        result = self.infer(payload["frames"],
            'Return only JSON: {"targets": [{"target_id": "unique_id", "name": "English visual noun phrase", '
            '"role": "manipulated_object|destination|gripper"}]}. Select at most 4 task-relevant visible objects. '
            'Use specific visual attributes to distinguish objects. Do not invent invisible objects.',
            {"task": payload["task"]})
        targets = result.get("targets")
        if not isinstance(targets, list) or not 1 <= len(targets) <= 4:
            raise ValueError("VLM must produce 1 to 4 tracking targets")
        ids = set()
        for target in targets:
            if not isinstance(target, dict) or any(not isinstance(target.get(k), str) or not target[k].strip()
                                                  for k in ("target_id", "name", "role")):
                raise ValueError("Invalid tracking target")
            if target["target_id"] in ids:
                raise ValueError("Duplicate tracking target ID")
            if target["role"] not in {"manipulated_object", "destination", "gripper"}:
                raise ValueError("Unknown tracking target role")
            ids.add(target["target_id"])
        return result

    def diagnose_failure(self, payload):
        evidence = {k: v for k, v in payload.items() if k != "current_frames"}
        result = self.infer(payload["current_frames"],
            'Return only JSON with failure_analysis (Chinese string) and recovery_suggestions (list of Chinese strings). '
            'Explain evidence and uncertainty. Tracking loss does not prove a dropped object. '
            'Suggestions are for human review, not robot commands.', evidence)
        if (not isinstance(result.get("failure_analysis"), str)
                or not isinstance(result.get("recovery_suggestions"), list)
                or any(not isinstance(item, str) for item in result["recovery_suggestions"])):
            raise ValueError("Invalid VLM diagnosis")
        return result


@lru_cache(maxsize=1)
def _qwen(path, quantize, max_new_tokens=768, image_max_side=768):
    return QwenBackend(path, quantize, max_new_tokens, image_max_side)


def load_qwen_once(path, quantize=True, max_new_tokens=768, image_max_side=768):
    with _LOCK:
        return _qwen(path, quantize, max_new_tokens, image_max_side)


@lru_cache(maxsize=1)
def _tracking_models(dino_path, sam_path):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2VideoModel, Sam2VideoProcessor
    dino_processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to("cuda").eval()
    sam_processor = Sam2VideoProcessor.from_pretrained(sam_path, local_files_only=True)
    sam = Sam2VideoModel.from_pretrained(sam_path, local_files_only=True).to("cuda").eval()
    return dino_processor, dino, sam_processor, sam


def create_online_tracker(dino_path, sam_path, options=None):
    with _LOCK:
        return OnlineTracker(_tracking_models(dino_path, sam_path), options or {})


class OnlineTracker:
    def __init__(self, models, options):
        self.dp, self.dino, self.sp, self.sam = models
        self.threshold = float(options.get("box_threshold", 0.3))
        self.text_threshold = float(options.get("text_threshold", 0.25))
        self.min_mask_pixels = int(options.get("min_mask_pixels", 16))
        self.reset_frames = int(options.get("reseed_frames", 60))
        self.retry_frames = int(options.get("redetect_frames", 10))
        if not 0 <= self.threshold <= 1 or not 0 <= self.text_threshold <= 1:
            raise ValueError("Detection thresholds must be in [0, 1]")
        if min(self.min_mask_pixels, self.reset_frames, self.retry_frames) < 1:
            raise ValueError("Mask size and tracking intervals must be positive")
        self.cameras = {}

    def initialize(self, plan, scene):
        self.plan = plan
        self.cameras = {}
        self.initial_perception = self.update(scene)

    def seed(self, image):
        import torch
        boxes, targets = [], []
        for target in self.plan.targets:
            inputs = self.dp(images=image, text=target.name.lower().rstrip(".") + ".", return_tensors="pt").to("cuda")
            output = self.dino(**inputs)
            result = self.dp.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=self.threshold, text_threshold=self.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            if len(result["scores"]):
                index = result["scores"].argmax().item()
                boxes.append(result["boxes"][index].tolist())
                targets.append(target.target_id)
        session = self.sp.init_video_session(inference_device="cuda", dtype=torch.float32,
                                            inference_state_device="cpu", video_storage_device="cpu")
        if boxes:
            self.sp.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=list(range(len(targets))), input_boxes=[boxes],
                original_size=(image.height, image.width),
            )
        return {"session": session, "targets": targets, "index": 0, "lost": False, "size": image.size}

    def update(self, observation):
        import torch
        tracks = []
        with _LOCK, torch.inference_mode():
            for camera, frame in observation.frames.items():
                image = rgb(frame.image)
                state = self.cameras.get(camera)
                if (state is None or state["size"] != image.size or state["index"] >= self.reset_frames
                        or (state["lost"] and state["index"] % self.retry_frames == 0)):
                    state = self.cameras[camera] = self.seed(image)
                visible = {}
                if state["targets"]:
                    inputs = self.sp(images=image, return_tensors="pt").to("cuda")
                    output = self.sam(inference_session=state["session"], frame_idx=state["index"], frame=inputs.pixel_values[0])
                    masks = self.sp.post_process_masks([output.pred_masks], original_sizes=inputs.original_sizes)[0]
                    for target, mask in zip(state["targets"], masks):
                        ys, xs = np.where(mask[0].detach().cpu().numpy() > 0)
                        if len(xs) >= self.min_mask_pixels:
                            visible[target] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                for target in self.plan.targets:
                    box = visible.get(target.target_id)
                    tracks.append(TrackState(target.target_id, camera, box is not None, box))
                state["lost"] = len(visible) < len(self.plan.targets)
                state["index"] += 1
        # Keep dense masks inside SAM; history and network responses contain boxes only.
        return PerceptionUpdate(observation.timestamp_sec, "complete", tuple(tracks), "grounding_dino_sam2")
