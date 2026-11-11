from __future__ import annotations

from typing import Protocol

from ..features.contracts import FeatureContext
from .contracts import (
    PerceptionUpdate, RuleFinding, RuleReport,
    MonitoringPlan, SceneSnapshot, StreamObservation, Task, TrackingPlan,
)


class ScenePlanner(Protocol):
    name: str
    def plan(self, task: Task, scene: SceneSnapshot) -> MonitoringPlan: ...


class OnlineObjectTracker(Protocol):
    name: str
    def initialize(self, plan: TrackingPlan, scene: SceneSnapshot) -> None: ...
    def update(self, observation: StreamObservation) -> PerceptionUpdate: ...


class StreamingFailureRule(Protocol):
    rule_id: str
    def update(self, context: FeatureContext) -> RuleFinding: ...


class RuleEngine(Protocol):
    def initialize(self, task: Task, plan: TrackingPlan, rule_package) -> None: ...
    def update(self, context: FeatureContext) -> RuleReport: ...
