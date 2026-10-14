import unittest
from unittest.mock import Mock

from failure_detection.backends import parse_json, rgb
from failure_detection.rules import TargetLostRule, ControllerFaultRule, create_rules
from failure_detection.pipeline.contracts import (
    Task, TrackingTarget, TrackingPlan, StreamObservation, PerceptionUpdate,
    TrackState, RuleContext, RuleFinding, RuleReport,
)
from failure_detection.pipeline import StreamingRoboSentinel
import numpy as np


def context(now, visible=False, state=None, cameras=("external", "wrist")):
    return RuleContext(Task("t", "move bottle"), TrackingPlan("complete", (
        TrackingTarget("bottle", "bottle", "manipulated_object"),), "test"),
        StreamObservation(now, {}, state), PerceptionUpdate(now, "complete", tuple(
            TrackState("bottle", camera, visible) for camera in cameras), "test"), ())


class BackendTests(unittest.TestCase):
    def test_json_and_color(self):
        self.assertEqual(parse_json('```json\n{"targets": []}\n```'), {"targets": []})
        with self.assertRaises(ValueError):
            parse_json("[]")
        self.assertEqual(rgb(np.array([[[1, 2, 3]]], dtype=np.uint8)).getpixel((0, 0)), (3, 2, 1))

    def test_loss_requires_duration_and_both_cameras(self):
        rule = TargetLostRule(seconds=2, max_gap_seconds=2)
        self.assertEqual(rule.update(context(0)).status, "unknown")
        self.assertEqual(rule.update(context(1)).status, "unknown")
        self.assertEqual(rule.update(context(2)).status, "confirmed")
        self.assertEqual(rule.update(context(3, True)).status, "clear")
        self.assertEqual(rule.update(context(4)).status, "unknown")
        self.assertEqual(rule.update(context(8)).status, "unknown")
        self.assertEqual(rule.update(context(9, cameras=("external",))).status, "unknown")
        self.assertEqual(rule.update(context(10)).status, "unknown")

    def test_one_visible_camera_is_sufficient(self):
        ctx = context(0)
        perception = PerceptionUpdate(0, "complete", (
            TrackState("bottle", "external", False), TrackState("bottle", "wrist", True)), "test")
        ctx = RuleContext(ctx.task, ctx.plan, ctx.observation, perception, ())
        self.assertEqual(TargetLostRule().update(ctx).status, "clear")

    def test_controller_and_registry(self):
        rule = ControllerFaultRule()
        self.assertEqual(rule.update(context(0)).status, "unknown")
        self.assertEqual(rule.update(context(0, state={"controller_state": "protective_stop"})).status, "confirmed")
        self.assertEqual(len(create_rules({"plugins": ["controller_fault", {"name": "target_lost", "options": {"seconds": 3}}]})), 2)

    def test_diagnosis_rearms_only_after_clear(self):
        planner, tracker, rules, diagnoser = Mock(), Mock(), Mock(), Mock()
        planner.plan.return_value = context(0).plan
        tracker.update.return_value = context(0).perception
        pipeline = StreamingRoboSentinel(planner, tracker, rules, diagnoser)
        pipeline.initialize(context(0).task, Mock())
        for status in ("confirmed", "confirmed", "unknown", "confirmed", "clear", "confirmed"):
            rules.update.return_value = RuleReport(0, "complete", (RuleFinding("r", status),), status == "confirmed")
            pipeline.update(context(0).observation)
        self.assertEqual(diagnoser.diagnose.call_count, 2)


if __name__ == "__main__":
    unittest.main()
