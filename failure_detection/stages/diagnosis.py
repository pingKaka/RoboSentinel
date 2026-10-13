from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from ..pipeline.contracts import Diagnosis, RuleContext, RuleReport


class EventTriggeredVLMDiagnoser:
    name = "vlm_event_diagnoser"

    def __init__(self, infer: Callable[[dict], dict]) -> None:
        self.infer = infer

    def diagnose(self, context: RuleContext, report: RuleReport) -> Diagnosis:
        result = self.infer({
            "task": context.task.description,
            "current_frames": {
                name: frame.image for name, frame in context.observation.frames.items()
            },
            "current_tracks": asdict(context.perception),
            "rule_report": asdict(report),
            "robot_state": context.observation.robot_state,
        })
        suggestions = result.get("recovery_suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        return Diagnosis(
            context.observation.timestamp_sec,
            "complete",
            result.get("failure_analysis"),
            tuple(str(item) for item in suggestions),
            self.name,
        )
