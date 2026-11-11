#!/usr/bin/env python3
"""Validate the three-stage RoboSentinel configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_config import load_project_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "cfg.yaml",
    )
    args = parser.parse_args()
    config, root = load_project_config(args.config)
    models = {
        name: {
            "path": str(resolve_path(root, value)),
            "available": resolve_path(root, value).is_dir(),
        }
        for name, value in config["models"].items()
    }
    summary = {
        "schema_version": 1,
        "flow": ["vlm_initial_scene", "online_grounding_dino_sam", "standard_features", "streaming_rules"],
        "feature_schema_version": config["stages"]["features"]["schema_version"],
        "models": models,
        "rule_mode": config["stages"]["rules"]["mode"],
        "rule_schema_version": config["stages"]["rules"]["schema_version"],
        "robot_state_enabled": config["streams"]["robot_state"]["enabled"],
        "status": "configured" if all(item["available"] for item in models.values()) else "missing_models",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
