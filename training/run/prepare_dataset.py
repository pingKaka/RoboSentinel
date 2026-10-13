#!/usr/bin/env python3
"""Build synchronized dual-camera frame sequences and Qwen3-VL SFT JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml
from PIL import Image


ALLOWED_SPLITS = {"train", "val", "test"}
FAILURE_TYPES = {
    "none",
    "grasp_failure",
    "wrong_object",
    "task_stalled",
    "target_offset",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    project_root = resolve_path(base, str(config.get("project_root", "../..")))
    config["project_root"] = project_root
    for key in ("annotations", "processed_dir", "sft_dir", "prompt_file"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        config[key] = resolve_path(project_root, str(config[key]))
    return config


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {path}\n"
            "Copy episodes.example.jsonl to episodes.jsonl and edit it first."
        )
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        validate_annotation(record, path, line_number)
        records.append(record)
    if not records:
        raise ValueError(f"No annotations found in {path}")
    return records


def validate_annotation(record: dict[str, Any], path: Path, line_number: int) -> None:
    required = {
        "episode_id",
        "task_id",
        "session_id",
        "split",
        "task_description",
        "external_video",
        "wrist_video",
        "reference_external_video",
        "reference_wrist_video",
        "outcome",
        "failure_type",
        "reason",
        "review_status",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Missing fields at {path}:{line_number}: {missing}")
    if record["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid split at {path}:{line_number}: {record['split']}")
    if record["failure_type"] not in FAILURE_TYPES:
        raise ValueError(
            f"Invalid failure_type at {path}:{line_number}: {record['failure_type']}"
        )
    expected = {
        "success": {"none"},
        "failure": FAILURE_TYPES - {"none", "unknown"},
        "unknown": {"unknown"},
    }
    if record["outcome"] not in expected:
        raise ValueError(f"Invalid outcome at {path}:{line_number}: {record['outcome']}")
    if record["failure_type"] not in expected[record["outcome"]]:
        raise ValueError(
            f"Inconsistent outcome/failure_type at {path}:{line_number}: "
            f"{record['outcome']}/{record['failure_type']}"
        )
    if record["review_status"] != "approved":
        raise ValueError(
            f"Episode {record['episode_id']} is not approved: {record['review_status']}"
        )


def video_info(path: Path) -> tuple[int, float, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if frame_count < 1 or fps <= 0:
        raise ValueError(f"Invalid frame count or FPS in {path}")
    return frame_count, fps, (frame_count - 1) / fps


def sample_video(
    path: Path,
    timestamps: list[float],
    image_size: int,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    frame_count, fps, _ = video_info(path)
    indices = [min(frame_count - 1, round(timestamp * fps)) for timestamp in timestamps]
    capture = cv2.VideoCapture(str(path))
    frames = []
    metadata = []
    for source_index, timestamp in zip(indices, timestamps):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Failed to decode frame {source_index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        frames.append(image)
        metadata.append(
            {"source_frame_index": source_index, "timestamp_sec": timestamp}
        )
    capture.release()
    return frames, metadata


def fit_tile(image: Image.Image, width: int, height: int) -> Image.Image:
    resized = image.copy().convert("RGB")
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return tile


def build_sequence(
    external_path: Path,
    wrist_path: Path,
    output_dir: Path,
    prefix: str,
    config: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    _, _, external_duration = video_info(external_path)
    _, _, wrist_duration = video_info(wrist_path)
    max_delta = float(config.get("max_camera_duration_delta_sec", 0.25))
    if abs(external_duration - wrist_duration) > max_delta:
        raise ValueError(
            f"Camera duration mismatch for {prefix}: {external_duration:.3f}s vs "
            f"{wrist_duration:.3f}s (limit {max_delta:.3f}s)"
        )
    interval = float(config["frame_sample_interval_seconds"])
    if interval <= 0:
        raise ValueError("frame_sample_interval_seconds must be > 0")
    common_end = min(external_duration, wrist_duration)
    sample_count = int(common_end / interval + 1e-9) + 1
    timestamps = [index * interval for index in range(sample_count)]
    external, external_meta = sample_video(
        external_path, timestamps, int(config["source_image_size"])
    )
    wrist, wrist_meta = sample_video(
        wrist_path, timestamps, int(config["source_image_size"])
    )
    width = int(config["tile_width"])
    height = int(config["tile_height"])
    quality = int(config.get("jpeg_quality", 95))
    paths = []
    metadata = []
    for index, (external_image, wrist_image) in enumerate(zip(external, wrist)):
        paired = Image.new("RGB", (width, height * 2), "white")
        paired.paste(fit_tile(external_image, width, height), (0, 0))
        paired.paste(fit_tile(wrist_image, width, height), (0, height))
        image_path = output_dir / f"{prefix}_{index:02d}.jpg"
        paired.save(image_path, quality=quality)
        paths.append(image_path)
        metadata.append(
            {
                "frame_id": f"{prefix.upper()}_{index + 1:02d}",
                "timestamp_sec": timestamps[index],
                "image": str(image_path),
                "external": external_meta[index],
                "wrist": wrist_meta[index],
            }
        )
    return paths, metadata


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Generated media must be inside project root: {path}") from exc


def image_sequence_text(name: str, metadata: list[dict[str, Any]]) -> str:
    lines = []
    count = len(metadata)
    for index, frame in enumerate(metadata):
        lines.append(
            f"{name} | FRAME {index + 1:02d}/{count:02d} | "
            f"TIME {frame['timestamp_sec']:.2f}s | "
            "EXTERNAL camera: top; WRIST camera: bottom."
        )
        lines.append("<image>")
    return "\n".join(lines)


def assistant_answer(record: dict[str, Any]) -> str:
    result = {
        "outcome": record["outcome"],
        "failure_type": record["failure_type"],
        "evidence_frames": record.get("evidence_frames", []),
        "reason": record["reason"],
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    project_root: Path = config["project_root"]
    records = load_annotations(config["annotations"])
    prompt_template = config["prompt_file"].read_text(encoding="utf-8").strip()
    processed_dir: Path = config["processed_dir"]
    sft_dir: Path = config["sft_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in ALLOWED_SPLITS}
    manifest = []

    episode_ids = [str(record["episode_id"]) for record in records]
    duplicates = [key for key, value in Counter(episode_ids).items() if value > 1]
    if duplicates:
        raise ValueError(f"Duplicate episode_id values: {duplicates}")

    for record in records:
        episode_id = str(record["episode_id"])
        episode_dir = processed_dir / episode_id
        overwrite = bool(config.get("overwrite", False) or args.overwrite)
        if episode_dir.exists() and overwrite:
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            key: resolve_path(project_root, str(record[key]))
            for key in (
                "external_video",
                "wrist_video",
                "reference_external_video",
                "reference_wrist_video",
            )
        }
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{episode_id}: missing {key}: {path}")

        reference_paths, reference_meta = build_sequence(
            paths["reference_external_video"],
            paths["reference_wrist_video"],
            episode_dir,
            "reference",
            config,
        )
        target_paths, target_meta = build_sequence(
            paths["external_video"],
            paths["wrist_video"],
            episode_dir,
            "target",
            config,
        )
        media_paths = reference_paths + target_paths
        user_content = "\n".join(
            (
                image_sequence_text("正确操作 / REFERENCE SUCCESS", reference_meta),
                image_sequence_text("待检测操作 / TARGET TO JUDGE", target_meta),
                prompt_template.replace("{task_description}", record["task_description"]),
            )
        )
        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_answer(record)},
            ],
            "images": [portable_path(path, project_root) for path in media_paths],
        }
        outputs[record["split"]].append(sample)
        episode_metadata = {
            "episode_id": episode_id,
            "task_id": record["task_id"],
            "session_id": record["session_id"],
            "split": record["split"],
            "outcome": record["outcome"],
            "failure_type": record["failure_type"],
            "reference_frames": reference_meta,
            "target_frames": target_meta,
        }
        (episode_dir / "metadata.json").write_text(
            json.dumps(episode_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(episode_metadata)
        print(f"Prepared {episode_id} -> {record['split']}")

    for split, samples in outputs.items():
        output_path = sft_dir / f"{split}.jsonl"
        output_path.write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
            encoding="utf-8",
        )
        print(f"Wrote {len(samples)} samples to {output_path}")
    (sft_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
