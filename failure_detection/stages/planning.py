from __future__ import annotations

from collections.abc import Callable

from ..pipeline.contracts import MonitoringPlan, SceneSnapshot, Task, TrackingPlan
from ..rule_dsl import (
    RULE_SCHEMA_VERSION, RulePackage, compile_authoring_plan, merge_rule_packages,
    parse_monitoring_plan,
)


class VLMScenePlanner:
    name = "vlm_scene_planner"

    def __init__(self, infer: Callable[[dict], dict], *, require_rules: bool = False,
                 max_attempts: int = 2, max_rules: int = 6,
                 manual_plan: MonitoringPlan | None = None, mode: str = "generated") -> None:
        self.infer = infer
        self.require_rules = require_rules
        self.max_attempts = max(1, int(max_attempts))
        self.max_rules = max(1, int(max_rules))
        self.manual_plan = manual_plan
        self.mode = mode

    def plan(self, task: Task, scene: SceneSnapshot) -> MonitoringPlan:
        if self.mode == "manual":
            if self.manual_plan is None:
                raise RuntimeError("manual rule mode requires a manual monitoring plan")
            return self.manual_plan
        payload = {
            "task": task.description,
            "frames": {name: frame.image for name, frame in scene.frames.items()},
        }
        if self.manual_plan is not None:
            payload["fixed_targets"] = [
                {"target_id": item.target_id, "name": item.name, "role": item.role}
                for item in self.manual_plan.targets
            ]
        for attempt in range(self.max_attempts):
            result = None
            try:
                result = self.infer(payload)
                if result.get("rule_package") is None and not self.require_rules:
                    from ..rule_dsl import parse_targets
                    targets = parse_targets(result.get("targets"))
                    generated = MonitoringPlan(
                        TrackingPlan("complete", targets, self.name),
                        RulePackage(RULE_SCHEMA_VERSION, "other", ()),
                    )
                else:
                    generated = (compile_authoring_plan(result, self.name, self.max_rules)
                                 if "rule_package" not in result and "rules" in result
                                 else parse_monitoring_plan(result, self.name, self.max_rules))
                if self.mode == "merged":
                    if self.manual_plan is None:
                        raise RuntimeError("merged rule mode requires a manual monitoring plan")
                    generated_keys = [(item.target_id, item.name, item.role) for item in generated.targets]
                    manual_keys = [(item.target_id, item.name, item.role) for item in self.manual_plan.targets]
                    if generated_keys != manual_keys:
                        raise ValueError("VLM did not preserve fixed manual tracking targets")
                    package = merge_rule_packages(generated.rule_package, self.manual_plan.rule_package, self.max_rules)
                    return MonitoringPlan(generated.tracking, package)
                return generated
            except (KeyError, TypeError, ValueError) as exc:
                payload = {**payload, "validation_error": str(exc), "attempt": attempt + 2}
                if result is not None:
                    payload["previous_output"] = result
        return MonitoringPlan(
            TrackingPlan("invalid", (), self.name),
            RulePackage(RULE_SCHEMA_VERSION, "other", ()),
        )
