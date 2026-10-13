#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failure_detection.transport.server import create_app
from project_config import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "cfg.yaml")
    args = parser.parse_args()
    config, _ = load_project_config(args.config)
    deployment = config.get("deployment", {})
    uvicorn.run(
        create_app(config),
        host=str(deployment.get("host", "0.0.0.0")),
        port=int(deployment.get("port", 8000)),
    )


if __name__ == "__main__":
    main()
