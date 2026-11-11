from __future__ import annotations

from ..features import FeatureContext, StandardFeatureExtractor
from ..runtime.buffer import TimeSeriesBuffer
from .contracts import (
    MonitoringPlan, PipelineUpdate, RuleFinding, RuleReport, SceneSnapshot,
    StreamObservation, Task, TrackingPlan,
)
from .interfaces import OnlineObjectTracker, RuleEngine, ScenePlanner


class StreamingRoboSentinel:
    """Process each synchronized image/state observation as it arrives."""

    def __init__(
        self,
        planner: ScenePlanner,
        tracker: OnlineObjectTracker,
        rules: RuleEngine,
        feature_extractor: StandardFeatureExtractor | None = None,
        buffer_seconds: float = 15.0,
    ) -> None:
        self.planner = planner
        self.tracker = tracker
        self.rules = rules
        self.feature_extractor = feature_extractor or StandardFeatureExtractor()
        self.history = TimeSeriesBuffer(buffer_seconds, lambda item: item.timestamp_sec)
        self.task = None
        self.plan = None

    def initialize(self, task: Task, scene: SceneSnapshot) -> None:
        result = self.planner.plan(task, scene)
        if isinstance(result, MonitoringPlan):
            plan, rule_package = result.tracking, result.rule_package
        elif isinstance(result, TrackingPlan):
            plan, rule_package = result, None
        else:
            raise TypeError("Planner must return MonitoringPlan")
        if plan.status != "complete":
            raise RuntimeError("VLM did not produce a valid monitoring plan")
        self.tracker.initialize(plan, scene)
        initialize_rules = getattr(self.rules, "initialize", None)
        if callable(initialize_rules):
            initialize_rules(task, plan, rule_package)
        self.task, self.plan = task, plan
        self.rule_package = rule_package
        self.history.clear()
        self.feature_extractor.reset()

    def update(self, observation: StreamObservation) -> PipelineUpdate:
        if self.task is None or self.plan is None:
            raise RuntimeError("Pipeline must be initialized before streaming")
        perception = self.tracker.update(observation)
        features = self.feature_extractor.extract(self.plan, observation, perception)
        self.history.append(features)
        context = FeatureContext(features, self.history.items())
        if observation.manual_failure:
            report = RuleReport(observation.timestamp_sec, "complete", (
                RuleFinding("manual_failure", "confirmed", "operator_reported_failure", evidence={
                    "source": "operator", "timestamp_sec": observation.timestamp_sec,
                    "note": "Failure reported directly by the operator.",
                }),), True)
            return PipelineUpdate(observation.timestamp_sec, perception, features, report)
        report = self.rules.update(context)
        return PipelineUpdate(observation.timestamp_sec, perception, features, report)
