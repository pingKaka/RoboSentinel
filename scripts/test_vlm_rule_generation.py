#!/usr/bin/env python3
"""Generate and schema-check monitoring rules from paired video initial frames."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from failure_detection.prompts import load_prompt
from failure_detection.rule_dsl import compile_authoring_plan
from my_backend import configured_qwen
from project_config import load_project_config, resolve_path


def read_first(path: Path):
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read first frame: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cfg.yaml"))
    parser.add_argument("--video-dir", type=Path, default=Path("video"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("result/vlm_cot_schema_retest_2026-11-11"))
    args = parser.parse_args()

    config, root = load_project_config(args.config)
    planner = config["stages"]["planner"]
    instruction = load_prompt(root, planner["prompt"])
    correction = load_prompt(root, planner["correction_prompt"])
    envelope = load_prompt(root, planner["request_envelope"])
    paths = {key: str(resolve_path(root, value)) for key, value in config["models"].items()}
    qwen = configured_qwen(paths, config.get("backend", {}))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for external in sorted(args.video_dir.glob("exterior_*.mp4")):
        label = external.stem.removeprefix("exterior_")
        wrist = args.video_dir / f"wrist_{label}.mp4"
        payload = {
            "task": config["task"]["description"],
            "frames": {"external": read_first(external), "wrist": read_first(wrist)},
        }
        attempts = []
        final_plan = None
        for attempt in range(2):
            raw = qwen.plan_monitoring(payload, instruction, correction, envelope)
            record = {"attempt": attempt + 1, "raw_output": raw}
            try:
                final_plan = compile_authoring_plan(raw, "vlm_rule_test", 6)
                record["valid"] = True
                attempts.append(record)
                break
            except (KeyError, TypeError, ValueError) as exc:
                record.update({"valid": False, "validation_error": str(exc)})
                attempts.append(record)
                payload = {**payload, "validation_error": str(exc),
                           "previous_output": raw, "attempt": attempt + 2}
        result = {
            "task": config["task"]["description"],
            "inputs": {"external": str(external), "wrist": str(wrist), "frame": 0},
            "attempts": attempts,
            "valid": final_plan is not None,
            "parsed_plan": asdict(final_plan) if final_plan is not None else None,
        }
        (args.output_dir / f"{label}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"case": label, "first_attempt": attempts[0]["valid"],
                     "final_valid": result["valid"], "attempts": len(attempts)})
        print(rows[-1], flush=True)

    summary = {"cases": rows, "first_pass": sum(row["first_attempt"] for row in rows),
               "final_pass": sum(row["final_valid"] for row in rows), "total": len(rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
