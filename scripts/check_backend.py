#!/usr/bin/env python3
"""Run real GPU inference on an image, without cameras or a running server."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from project_config import load_project_config, resolve_path
from failure_detection.backends import create_online_tracker
from failure_detection.pipeline.contracts import Frame, SceneSnapshot, StreamObservation, TrackingPlan, TrackingTarget, Task
from my_backend import create_pipeline, configured_qwen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "cfg.yaml")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mode", choices=["qwen", "tracker", "full"])
    parser.add_argument("--target", help="English target phrase for tracker-only check")
    parser.add_argument("--description")
    args = parser.parse_args()
    config, root = load_project_config(args.config)
    check = config.get("check", {})
    if args.image is None and check.get("image"):
        args.image = resolve_path(root, check["image"])
    if args.image is None:
        parser.error("Set check.image in cfg.yaml or provide --image")
    args.mode = args.mode or check.get("mode", "full")
    args.target = args.target or check.get("target", "bottle")
    args.description = args.description or config["task"]["description"]
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Cannot read --image")
    paths = {key: str(resolve_path(root, value)) for key, value in config["models"].items()}
    frames = {name: Frame(name, 0, 0, image) for name in ("external", "wrist")}
    scene = SceneSnapshot(0, frames)
    if args.mode == "qwen":
        qwen = configured_qwen(paths, config.get("backend", {}))
        print(qwen.plan_tracking_targets({"task": args.description, "frames": {k: image for k in frames}}))
        print(qwen.diagnose_failure({"task": args.description, "current_frames": {k: image for k in frames},
                                    "rule_report": {"failure_type": "test_event", "note": "Diagnostic smoke test; no actual failure established"},
                                    "current_tracks": {}, "robot_state": None}))
        return
    if args.mode == "tracker":
        pipeline = create_online_tracker(paths["grounding_dino"], paths["sam"], config.get("backend", {}))
        pipeline.initialize(TrackingPlan("complete", (TrackingTarget("object", args.target, "manipulated_object"),), "smoke"), scene)
    else:
        pipeline = create_pipeline(config)
        pipeline.initialize(Task("smoke", args.description), scene)
    for index in range(1, int(check.get("frames", 3)) + 1):
        update = pipeline.update(StreamObservation(index * float(check.get("interval_sec", 0.1)), frames))
        perception = update if args.mode == "tracker" else update.perception
        print(perception)
        if not any(track.visible for track in perception.tracks):
            raise RuntimeError("No visible targets: use an image containing the requested object")
    print("GPU inference check passed")


if __name__ == "__main__":
    main()
