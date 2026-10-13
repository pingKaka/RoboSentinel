from __future__ import annotations

import unittest

from failure_detection.pipeline import (
    Frame, PerceptionUpdate, RuleFinding, SceneSnapshot, StreamObservation,
    StreamingRoboSentinel, Task, TrackState,
)
from failure_detection.pipeline.rule_engine import StreamingRuleEngine
from failure_detection.stages import (
    EventTriggeredVLMDiagnoser, OnlineGroundingSAMTracker, VLMScenePlanner,
)
from project_config import load_project_config


class TrackerBackend:
    def initialize(self, plan, scene):
        self.plan = plan

    def update(self, observation):
        return PerceptionUpdate(
            observation.timestamp_sec, "complete",
            (TrackState("object", "external", True, quality=0.9),), "test",
        )


class EventRule:
    rule_id = "test_event"

    def update(self, context):
        status = "confirmed" if context.observation.timestamp_sec >= 2.0 else "clear"
        return RuleFinding(self.rule_id, status, "test_failure" if status == "confirmed" else None)


class StreamingPipelineTest(unittest.TestCase):
    def test_configuration_uses_streams(self):
        config, _ = load_project_config(__import__("pathlib").Path("cfg.yaml"))
        self.assertIn("streams", config)
        self.assertEqual(config["stages"]["tracker"]["backend"], "grounding_dino_sam_online")

    def test_vlm_diagnosis_is_only_called_after_confirmed_event(self):
        calls = []
        planner = VLMScenePlanner(lambda _: {"targets": [
            {"target_id": "object", "name": "bottle", "role": "manipulated_object"}
        ]})
        tracker = OnlineGroundingSAMTracker(TrackerBackend())
        diagnoser = EventTriggeredVLMDiagnoser(lambda payload: calls.append(payload) or {
            "failure_analysis": "test", "recovery_suggestions": ["retry"]
        })
        pipeline = StreamingRoboSentinel(
            planner, tracker, StreamingRuleEngine([EventRule()]), diagnoser
        )
        frame = Frame("external", 0.0, 0, object())
        pipeline.initialize(Task("task", "move bottle"), SceneSnapshot(0.0, {"external": frame}))
        first = pipeline.update(StreamObservation(1.0, {"external": frame}))
        second = pipeline.update(StreamObservation(2.0, {"external": frame}))
        self.assertIsNone(first.diagnosis)
        self.assertIsNotNone(second.diagnosis)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
