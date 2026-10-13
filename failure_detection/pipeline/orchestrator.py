from __future__ import annotations

from ..runtime.buffer import TimeSeriesBuffer
from .contracts import PipelineUpdate, RuleContext, SceneSnapshot, StreamObservation, Task
from .interfaces import FailureDiagnoser, OnlineObjectTracker, RuleEngine, ScenePlanner


class StreamingRoboSentinel:
    """Process each synchronized image/state observation as it arrives."""

    def __init__(
        self,
        planner: ScenePlanner,
        tracker: OnlineObjectTracker,
        rules: RuleEngine,
        diagnoser: FailureDiagnoser,
        buffer_seconds: float = 15.0,
    ) -> None:
        self.planner = planner
        self.tracker = tracker
        self.rules = rules
        self.diagnoser = diagnoser
        self.history = TimeSeriesBuffer(buffer_seconds, lambda item: item.timestamp_sec)
        self.task = None
        self.plan = None

    def initialize(self, task: Task, scene: SceneSnapshot) -> None:
        plan = self.planner.plan(task, scene)
        if plan.status != "complete":
            raise RuntimeError("VLM did not produce a valid tracking plan")
        self.tracker.initialize(plan, scene)
        self.task, self.plan = task, plan

    def update(self, observation: StreamObservation) -> PipelineUpdate:
        if self.task is None or self.plan is None:
            raise RuntimeError("Pipeline must be initialized before streaming")
        perception = self.tracker.update(observation)
        self.history.append(perception)
        context = RuleContext(
            self.task, self.plan, observation, perception, self.history.items()
        )
        report = self.rules.update(context)
        diagnosis = self.diagnoser.diagnose(context, report) if report.failure_detected else None
        return PipelineUpdate(observation.timestamp_sec, perception, report, diagnosis)
