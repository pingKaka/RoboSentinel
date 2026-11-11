import unittest
from unittest.mock import Mock

from failure_detection.backends import box_iou, parse_json, rgb
from failure_detection.pipeline.contracts import (
    Task, TrackingTarget, TrackingPlan, StreamObservation, PerceptionUpdate,
    RuleFinding, RuleReport, Frame,
)
from failure_detection.pipeline import StreamingRoboSentinel
import numpy as np


class BackendTests(unittest.TestCase):
    def test_json_and_color(self):
        self.assertEqual(parse_json('```json\n{"targets": []}\n```'), {"targets": []})
        with self.assertRaises(ValueError):
            parse_json("[]")
        self.assertEqual(rgb(np.array([[[1, 2, 3]]], dtype=np.uint8)).getpixel((0, 0)), (3, 2, 1))
        self.assertEqual(box_iou((0, 0, 2, 2), (0, 0, 2, 2)), 1.0)
        self.assertEqual(box_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_pipeline_passes_each_rule_report_through(self):
        planner, tracker, rules = Mock(), Mock(), Mock()
        planner.plan.return_value = TrackingPlan("complete", (
            TrackingTarget("bottle", "bottle", "manipulated_object"),), "test")
        tracker.update.return_value = PerceptionUpdate(0, "complete", (), "test")
        pipeline = StreamingRoboSentinel(planner, tracker, rules)
        pipeline.initialize(Task("t", "move bottle"), Mock())
        results = []
        for status in ("confirmed", "confirmed", "unknown", "confirmed", "clear", "confirmed"):
            rules.update.return_value = RuleReport(0, "complete", (RuleFinding("r", status),), status == "confirmed")
            frame = Frame("external", 0, 0, np.zeros((20, 30, 3), np.uint8))
            results.append(pipeline.update(StreamObservation(0, {"external": frame})).rules.findings[0].status)
        self.assertEqual(results, ["confirmed", "confirmed", "unknown", "confirmed", "clear", "confirmed"])
        self.assertEqual(rules.update.call_count, 6)


if __name__ == "__main__":
    unittest.main()
