#!/usr/bin/env python3
"""Validate generated Qwen3-VL SFT data and check group leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


EXPECTED_TYPES = {
    "success": {"none"},
    "failure": {"grasp_failure", "wrong_object", "task_stalled", "target_offset"},
    "unknown": {"unknown"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_config(path: Path) -> tuple[Path, Path]:
    config_path = path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root_value = Path(str(config.get("project_root", "../.."))).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (config_path.parent / root_value).resolve()
    sft_value = Path(str(config["sft_dir"])).expanduser()
    sft_dir = sft_value.resolve() if sft_value.is_absolute() else (root / sft_value).resolve()
    return root, sft_dir


def main() -> int:
    args = parse_args()
    project_root, sft_dir = resolve_config(args.config)
    manifest = load_lines(sft_dir / "manifest.jsonl")
    groups: dict[str, set[str]] = defaultdict(set)
    episode_splits = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        groups[str(row["session_id"])].add(str(row["split"]))
        episode_splits[(str(row["episode_id"]), str(row["split"]))] += 1
        label_counts[str(row["split"])][str(row["failure_type"])] += 1
    leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"session_id leakage across splits: {leaked}")
    duplicate_episodes = [key for key, count in episode_splits.items() if count > 1]
    if duplicate_episodes:
        raise ValueError(f"Duplicate episodes in manifest: {duplicate_episodes}")

    total = 0
    for split in ("train", "val", "test"):
        rows = load_lines(sft_dir / f"{split}.jsonl")
        for index, row in enumerate(rows, 1):
            messages = row.get("messages", [])
            images = row.get("images", [])
            if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError(f"{split}:{index}: expected one user and one assistant message")
            tag_count = str(messages[0].get("content", "")).count("<image>")
            if tag_count != len(images):
                raise ValueError(
                    f"{split}:{index}: {tag_count} <image> tags but {len(images)} paths"
                )
            for image_value in images:
                image_path = Path(str(image_value))
                image_path = image_path if image_path.is_absolute() else project_root / image_path
                if not image_path.is_file():
                    raise FileNotFoundError(f"{split}:{index}: {image_path}")
                with Image.open(image_path) as image:
                    image.verify()
            answer = json.loads(str(messages[1].get("content", "")))
            required = {"outcome", "failure_type", "evidence_frames", "reason"}
            if set(answer) != required:
                raise ValueError(f"{split}:{index}: answer keys must be {sorted(required)}")
            outcome = answer["outcome"]
            failure_type = answer["failure_type"]
            if outcome not in EXPECTED_TYPES or failure_type not in EXPECTED_TYPES[outcome]:
                raise ValueError(
                    f"{split}:{index}: inconsistent outcome/failure_type: "
                    f"{outcome}/{failure_type}"
                )
            if not isinstance(answer["evidence_frames"], list):
                raise ValueError(f"{split}:{index}: evidence_frames must be a list")
            if not isinstance(answer["reason"], str) or not answer["reason"].strip():
                raise ValueError(f"{split}:{index}: reason must be a non-empty string")
        total += len(rows)
        print(f"{split}: {len(rows)} samples; labels={dict(label_counts[split])}")
    print(f"Dataset validation passed: {total} samples, no session leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
