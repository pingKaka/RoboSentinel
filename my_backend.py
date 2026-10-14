"""Deployment factory; model weights are shared, task state is not."""
from pathlib import Path

from failure_detection.backends import load_qwen_once, create_online_tracker
from failure_detection.rules import create_rules
from failure_detection.pipeline import StreamingRoboSentinel
from failure_detection.pipeline.rule_engine import StreamingRuleEngine
from failure_detection.stages import VLMScenePlanner, OnlineGroundingSAMTracker, EventTriggeredVLMDiagnoser


def model_settings(config):
    root = Path(config.get("_config_root", Path(__file__).parent))
    paths = {key: str((root / value).resolve()) for key, value in config["models"].items()}
    options = config.get("backend", {})
    return paths, options


def configured_qwen(paths, options):
    return load_qwen_once(paths["vlm"], options.get("quantize_vlm", True),
                          options.get("vlm_max_new_tokens", 768), options.get("vlm_image_max_side", 768))


def preload_models(config):
    """Keep the three cached model objects on CUDA for the server lifetime."""
    import torch
    paths, options = model_settings(config)
    qwen = configured_qwen(paths, options)
    tracker = create_online_tracker(paths["grounding_dino"], paths["sam"], options)
    models = {"vlm": qwen.model, "grounding_dino": tracker.dino, "sam": tracker.sam}
    for name, model in models.items():
        if any(parameter.device.type != "cuda" for parameter in model.parameters()):
            raise RuntimeError(f"{name} has parameters outside CUDA")
    return {"resident_models": list(models), "device": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2)}


def create_pipeline(config):
    paths, options = model_settings(config)
    qwen = configured_qwen(paths, options)
    tracker = create_online_tracker(paths["grounding_dino"], paths["sam"], options)
    return StreamingRoboSentinel(
        VLMScenePlanner(qwen.plan_tracking_targets), OnlineGroundingSAMTracker(tracker),
        StreamingRuleEngine(create_rules(config["stages"]["rules"])),
        EventTriggeredVLMDiagnoser(qwen.diagnose_failure),
        buffer_seconds=config["runtime"]["buffer_seconds"],
    )
