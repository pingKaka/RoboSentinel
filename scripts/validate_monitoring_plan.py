#!/usr/bin/env python3
"""Validate a human-authored target and rule plan without loading GPU models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failure_detection.rule_dsl import load_monitoring_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="JSON file containing targets and rule_package")
    parser.add_argument("--max-rules", type=int, default=6)
    args = parser.parse_args()
    plan = load_monitoring_plan(Path.cwd(), args.plan, args.max_rules)
    summary = {
        "status": "valid",
        "planner": plan.planner,
        "targets": [asdict(item) for item in plan.targets],
        "task_type": plan.rule_package.task_type,
        "rule_ids": [item.rule_id for item in plan.rule_package.rules],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
