from __future__ import annotations

from typing import Protocol

from .contracts import (
    Diagnosis, PerceptionUpdate, RuleContext, RuleFinding, RuleReport,
    SceneSnapshot, StreamObservation, Task, TrackingPlan,
)


class ScenePlanner(Protocol):
    name: str
    def plan(self, task: Task, scene: SceneSnapshot) -> TrackingPlan: ...


class OnlineObjectTracker(Protocol):
    name: str
    def initialize(self, plan: TrackingPlan, scene: SceneSnapshot) -> None: ...
    def update(self, observation: StreamObservation) -> PerceptionUpdate: ...


class StreamingFailureRule(Protocol):
    rule_id: str
    def update(self, context: RuleContext) -> RuleFinding: ...


class RuleEngine(Protocol):
    def update(self, context: RuleContext) -> RuleReport: ...


class FailureDiagnoser(Protocol):
    name: str
    def diagnose(self, context: RuleContext, report: RuleReport) -> Diagnosis: ...
