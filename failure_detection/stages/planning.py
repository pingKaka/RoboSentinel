from __future__ import annotations

from collections.abc import Callable
import re

from ..pipeline.contracts import MonitoringPlan, SceneSnapshot, Task, TrackingPlan, TrackingTarget
from ..rule_dsl import RULE_SCHEMA_VERSION, RulePackage, parse_rule_package


class VLMScenePlanner:
    name = "vlm_scene_planner"

    def __init__(self, infer: Callable[[dict], dict], *, require_rules: bool = False,
                 max_attempts: int = 2, max_rules: int = 6) -> None:
        self.infer = infer
        self.require_rules = require_rules
        self.max_attempts = max(1, int(max_attempts))
        self.max_rules = max(1, int(max_rules))

    def plan(self, task: Task, scene: SceneSnapshot) -> MonitoringPlan:
        payload = {
            "task": task.description,
            "frames": {name: frame.image for name, frame in scene.frames.items()},
        }
        for attempt in range(self.max_attempts):
            try:
                result = self.infer(payload)
                targets = self._targets(result.get("targets"))
                raw_package = result.get("rule_package")
                if raw_package is None and "rules" in result:
                    raw_package = {key: result.get(key) for key in ("schema_version", "task_type", "rules")}
                if raw_package is None and not self.require_rules:
                    package = RulePackage(RULE_SCHEMA_VERSION, "other", ())
                else:
                    package = parse_rule_package(raw_package, {item.target_id for item in targets}, self.max_rules)
                return MonitoringPlan(TrackingPlan("complete", targets, self.name), package)
            except (KeyError, TypeError, ValueError) as exc:
                payload = {**payload, "validation_error": str(exc), "attempt": attempt + 2}
        return MonitoringPlan(
            TrackingPlan("invalid", (), self.name),
            RulePackage(RULE_SCHEMA_VERSION, "other", ()),
        )

    @staticmethod
    def _targets(values) -> tuple[TrackingTarget, ...]:
        if not isinstance(values, list) or not 1 <= len(values) <= 4:
            raise ValueError("targets must contain 1 to 4 entries")
        targets = []
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("tracking target must be an object")
            target_id, name, role = (item.get(key) for key in ("target_id", "name", "role"))
            attributes = item.get("attributes", {})
            if (not isinstance(target_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", target_id)
                    or not isinstance(name, str) or not name.strip()
                    or role not in {"manipulated_object", "destination", "gripper"}
                    or not isinstance(attributes, dict)):
                raise ValueError("invalid tracking target")
            targets.append(TrackingTarget(target_id, name.strip(), role, dict(attributes)))
        targets = tuple(targets)
        if len({item.target_id for item in targets}) != len(targets):
            raise ValueError("duplicate tracking target ID")
        return targets
