from __future__ import annotations

from ..runtime.buffer import TimeSeriesBuffer
from .contracts import PipelineUpdate, RuleContext, RuleFinding, RuleReport, SceneSnapshot, StreamObservation, Task
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
        self._diagnosed_events = set()

    def initialize(self, task: Task, scene: SceneSnapshot) -> None:
        plan = self.planner.plan(task, scene)
        if plan.status != "complete":
            raise RuntimeError("VLM did not produce a valid tracking plan")
        self.tracker.initialize(plan, scene)
        self.task, self.plan = task, plan
        self._diagnosed_events.clear()

    def update(self, observation: StreamObservation) -> PipelineUpdate:
        if self.task is None or self.plan is None:
            raise RuntimeError("Pipeline must be initialized before streaming")
        perception = self.tracker.update(observation)
        self.history.append(perception)
        context = RuleContext(
            self.task, self.plan, observation, perception, self.history.items()
        )
        if observation.reset_diagnosis:
            self._diagnosed_events.clear()
            return PipelineUpdate(observation.timestamp_sec, perception,
                                  RuleReport(observation.timestamp_sec, "reset", (), None), None)
        if observation.manual_failure:
            report = RuleReport(observation.timestamp_sec, "complete", (
                RuleFinding("manual_failure", "confirmed", "operator_reported_failure", evidence={
                    "source": "operator", "timestamp_sec": observation.timestamp_sec,
                    "note": "Human requested diagnosis; failure was not verified by automatic rules.",
                }),), True)
            return PipelineUpdate(observation.timestamp_sec, perception, report,
                                  self.diagnoser.diagnose(context, report))
        report = self.rules.update(context)
        confirmed = {item.rule_id for item in report.findings if item.status == "confirmed"}
        cleared = {item.rule_id for item in report.findings if item.status == "clear"}
        self._diagnosed_events.difference_update(cleared)
        diagnosis = None
        if report.failure_detected and confirmed - self._diagnosed_events:
            diagnosis = self.diagnoser.diagnose(context, report)
            self._diagnosed_events.update(confirmed)
        return PipelineUpdate(observation.timestamp_sec, perception, report, diagnosis)
