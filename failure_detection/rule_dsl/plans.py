"""Load and combine validated monitoring plans from human or VLM sources."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from ..pipeline.contracts import MonitoringPlan, TrackingPlan, TrackingTarget
from .contracts import RULE_SCHEMA_VERSION, RulePackage
from .schema import parse_rule_package


ALLOWED_TARGET_ROLES = {"manipulated_object", "destination", "gripper"}
TARGET_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def parse_targets(values: Any) -> tuple[TrackingTarget, ...]:
    """Validate tracking targets shared by GroundingDINO/SAM and rules."""
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ValueError("targets must contain 1 to 4 entries")
    targets = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("tracking target must be an object")
        target_id, name, role = (item.get(key) for key in ("target_id", "name", "role"))
        attributes = item.get("attributes", {})
        if (not isinstance(target_id, str) or not TARGET_ID.fullmatch(target_id)
                or not isinstance(name, str) or not name.strip()
                or role not in ALLOWED_TARGET_ROLES or not isinstance(attributes, dict)):
            raise ValueError("invalid tracking target")
        targets.append(TrackingTarget(target_id, name.strip(), role, dict(attributes)))
    result = tuple(targets)
    if len({item.target_id for item in result}) != len(result):
        raise ValueError("duplicate tracking target ID")
    return result


def parse_monitoring_plan(value: Any, planner: str, max_rules: int = 6) -> MonitoringPlan:
    """Validate a complete target-and-rule object from any source."""
    if not isinstance(value, dict):
        raise ValueError("monitoring plan must be a JSON object")
    targets = parse_targets(value.get("targets"))
    raw_package = value.get("rule_package")
    if raw_package is None and "rules" in value:
        raw_package = {key: value.get(key) for key in ("schema_version", "task_type", "rules")}
    package = parse_rule_package(raw_package, {item.target_id for item in targets}, max_rules)
    return MonitoringPlan(TrackingPlan("complete", targets, planner), package)


def load_monitoring_plan(config_root: str | Path, value: str | Path, max_rules: int = 6) -> MonitoringPlan:
    """Load a manually authored plan from JSON and run full semantic validation."""
    root = Path(config_root).expanduser().resolve()
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    return parse_monitoring_plan(json.loads(path.read_text(encoding="utf-8")), f"manual:{path}", max_rules)


def merge_rule_packages(generated: RulePackage, manual: RulePackage, max_rules: int) -> RulePackage:
    """Append human rules to generated rules without silently replacing either."""
    combined = (*generated.rules, *manual.rules)
    ids = [item.rule_id for item in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("generated and manual rules contain duplicate rule_id values")
    if len(combined) > max_rules:
        raise ValueError(f"merged rule count exceeds max_rules={max_rules}")
    task_type = manual.task_type if manual.task_type != "other" else generated.task_type
    return RulePackage(RULE_SCHEMA_VERSION, task_type, tuple(combined))
