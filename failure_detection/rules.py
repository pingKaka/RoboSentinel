"""Conservative monitoring rules; no task success is inferred from boxes alone."""
import importlib
import math

from .pipeline.contracts import RuleFinding


class TargetLostRule:
    rule_id = "target_lost"

    def __init__(self, seconds=2.0, max_gap_seconds=1.0):
        self.seconds = float(seconds)
        self.max_gap = float(max_gap_seconds)
        if self.seconds <= 0 or self.max_gap <= 0:
            raise ValueError("Rule timeouts must be positive")
        self.since = {}
        self.previous = None

    def update(self, context):
        now = context.observation.timestamp_sec
        if not math.isfinite(now):
            return RuleFinding(self.rule_id, "unknown")
        if self.previous is not None and (now <= self.previous or now - self.previous > self.max_gap):
            self.since.clear()
        self.previous = now
        lost, unknown = [], not any(t.role == "manipulated_object" for t in context.plan.targets)
        for target in context.plan.targets:
            if target.role != "manipulated_object":
                continue
            tracks = [t for t in context.perception.tracks if t.target_id == target.target_id]
            if any(t.visible for t in tracks):
                self.since.pop(target.target_id, None)
            elif context.perception.status != "complete" or not {"external", "wrist"} <= {t.camera for t in tracks}:
                self.since.pop(target.target_id, None)
                unknown = True
            else:
                start = self.since.setdefault(target.target_id, now)
                if now - start >= self.seconds:
                    lost.append(target.target_id)
                else:
                    unknown = True
        if lost:
            return RuleFinding(self.rule_id, "confirmed", "target_unobservable", evidence={"targets": lost, "duration_sec": self.seconds})
        return RuleFinding(self.rule_id, "unknown" if unknown else "clear")


class ControllerFaultRule:
    rule_id = "controller_fault"

    def update(self, context):
        state = (context.observation.robot_state or {}).get("controller_state")
        if state is None:
            return RuleFinding(self.rule_id, "unknown")
        fault = str(state).lower() in {"fault", "error", "protective_stop", "emergency_stop"}
        if not fault and str(state).lower() not in {"running", "idle", "stopped", "normal", "completed"}:
            return RuleFinding(self.rule_id, "unknown", evidence={"controller_state": state})
        return RuleFinding(self.rule_id, "confirmed" if fault else "clear",
                           "controller_fault" if fault else None, evidence={"controller_state": state})


def create_rules(config):
    builtins = {"target_lost": TargetLostRule, "controller_fault": ControllerFaultRule}
    rules = []
    for entry in config.get("plugins", []):
        name = entry if isinstance(entry, str) else entry["name"]
        options = {} if isinstance(entry, str) else entry.get("options", {})
        cls = builtins.get(name)
        if cls is None:
            module, attr = name.split(":", 1)
            cls = getattr(importlib.import_module(module), attr)
        rule = cls(**options)
        if not getattr(rule, "rule_id", None) or not callable(getattr(rule, "update", None)):
            raise ValueError("Rules require a nonempty rule_id and update(context)")
        if any(existing.rule_id == rule.rule_id for existing in rules):
            raise ValueError(f"Duplicate rule_id: {rule.rule_id}")
        rules.append(rule)
    return rules
