#!/usr/bin/env python3
"""Translate train.yaml into a reproducible ms-swift Qwen3-VL SFT command."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flag_name(key: str) -> str:
    return "--" + key


def append_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        command.extend((flag_name(key), str(value).lower()))
    else:
        command.extend((flag_name(key), str(value)))


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"model", "train_dataset", "val_dataset", "output_dir"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing training config keys: {missing}")

    command = ["swift", "sft"]
    mapping = {
        "model": "model",
        "train_dataset": "dataset",
        "val_dataset": "val_dataset",
        "output_dir": "output_dir",
        "tuner_type": "tuner_type",
        "torch_dtype": "torch_dtype",
        "num_train_epochs": "num_train_epochs",
        "per_device_train_batch_size": "per_device_train_batch_size",
        "per_device_eval_batch_size": "per_device_eval_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate",
        "warmup_ratio": "warmup_ratio",
        "weight_decay": "weight_decay",
        "max_length": "max_length",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "target_modules": "target_modules",
        "freeze_vit": "freeze_vit",
        "freeze_aligner": "freeze_aligner",
        "gradient_checkpointing": "gradient_checkpointing",
        "attn_impl": "attn_impl",
        "packing": "packing",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
        "logging_steps": "logging_steps",
        "save_total_limit": "save_total_limit",
        "dataset_num_proc": "dataset_num_proc",
        "dataloader_num_workers": "dataloader_num_workers",
        "seed": "seed",
    }
    for config_key, cli_key in mapping.items():
        append_arg(command, cli_key, config.get(config_key))
    command.extend(str(value) for value in config.get("extra_args", []))

    environment = os.environ.copy()
    environment["IMAGE_MAX_TOKEN_NUM"] = str(config.get("image_max_token_num", 256))
    environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
        config.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
    )
    print(f"IMAGE_MAX_TOKEN_NUM={environment['IMAGE_MAX_TOKEN_NUM']}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={environment['PYTORCH_CUDA_ALLOC_CONF']}")
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
