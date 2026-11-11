from __future__ import annotations

import unittest

from failure_detection.pipeline import (
    Frame, PerceptionUpdate, RuleFinding, SceneSnapshot, StreamObservation,
    StreamingRoboSentinel, Task, TrackState,
)
from failure_detection.pipeline.rule_engine import StreamingRuleEngine
from failure_detection.stages import (
    OnlineGroundingSAMTracker, VLMScenePlanner,
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
        status = "confirmed" if context.snapshot.timestamp_sec >= 2.0 else "clear"
        return RuleFinding(self.rule_id, status, "test_failure" if status == "confirmed" else None)


class StreamingPipelineTest(unittest.TestCase):
    def test_configuration_uses_streams(self):
        config, _ = load_project_config(__import__("pathlib").Path("cfg.yaml"))
        self.assertIn("streams", config)
        self.assertEqual(config["stages"]["tracker"]["backend"], "grounding_dino_sam_online")

    def test_rule_report_is_the_final_detection_output(self):
        planner = VLMScenePlanner(lambda _: {"targets": [
            {"target_id": "object", "name": "bottle", "role": "manipulated_object"}
        ]})
        tracker = OnlineGroundingSAMTracker(TrackerBackend())
        pipeline = StreamingRoboSentinel(
            planner, tracker, StreamingRuleEngine([EventRule()])
        )
        frame = Frame("external", 0.0, 0, object())
        pipeline.initialize(Task("task", "move bottle"), SceneSnapshot(0.0, {"external": frame}))
        first = pipeline.update(StreamObservation(1.0, {"external": frame}))
        second = pipeline.update(StreamObservation(2.0, {"external": frame}))
        self.assertFalse(first.rules.failure_detected)
        self.assertTrue(second.rules.failure_detected)
        self.assertEqual(second.rules.findings[0].failure_type, "test_failure")
        value = second.to_dict()
        self.assertNotIn("diagnosis", value)
        self.assertEqual(value["features"]["schema_version"], "2.0.0")
        self.assertEqual(set(value["features"]["robot"]), {
            "tcp_position_m", "tcp_frame", "gripper_position_normalized",
        })


if __name__ == "__main__":
    unittest.main()
