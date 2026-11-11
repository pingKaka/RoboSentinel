from __future__ import annotations

import json
from pathlib import Path
import unittest

from failure_detection.features.contracts import (
    CameraEntityFeature, EntityFeature, FeatureContext, FeatureQuality,
    FeatureSnapshot, RobotFeature, SpatialRelationFeature,
)
from failure_detection.pipeline.contracts import Frame, SceneSnapshot, Task, TrackingPlan, TrackingTarget
from failure_detection.pipeline.rule_engine import StreamingRuleEngine
from failure_detection.rule_dsl import (
    SCALAR_FEATURES, DeclarativeRule, RuleSchemaError, parse_rule_package,
)
from failure_detection.stages import VLMScenePlanner


def raw_package(condition=None, duration=0.0, samples=1):
    return {
        "schema_version": "1.0.0",
        "task_type": "pick_and_place",
        "rules": [{
            "rule_id": "grasp_failed",
            "failure_type": "object_not_grasped",
            "constraint_type": "during_execution",
            "description": "A closed gripper should retain the object.",
            "feature_bindings": [
                {"alias": "object_visible", "feature": "entity.visible_any",
                 "entity": "object", "reason": "The object must remain observable."},
                {"alias": "gripper_opening", "feature": "robot.gripper_position_normalized",
                 "reason": "A low opening indicates a grasp attempt."},
            ],
            "activation": {"op": "eq", "feature": "object_visible", "value": True},
            "condition": condition or {"op": "lt", "feature": "gripper_opening", "value": 0.2},
            "duration_sec": duration,
            "minimum_samples": samples,
            "unknown_policy": "unknown",
            "explanation_template": "The gripper closed but the grasp constraint was violated.",
        }],
    }


def snapshot(timestamp, opening, visible=True):
    camera = CameraEntityFeature("external", "visible" if visible else "not_visible",
                                 (0.4, 0.5) if visible else None, 0.1 if visible else None,
                                 0.3 if visible else None, 1 if visible else None,
                                 0.8 if visible else None, 0.9)
    entity = EntityFeature("object", "white bottle", "manipulated_object", (camera,),
                           int(visible), visible, not visible)
    quality = FeatureQuality(True, opening is not None, (), {})
    return FeatureSnapshot("2.0.0", timestamp, None, quality, (entity,), (),
                           RobotFeature((0.1, 0.2, 0.3), "base", opening))


class RuleSchemaTests(unittest.TestCase):
    def test_machine_readable_schema_matches_runtime_catalog(self):
        path = Path(__file__).parents[1] / "rule_dsl" / "rule_schema_v1.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        declared = schema["$defs"]["binding"]["properties"]["feature"]["enum"]
        self.assertEqual(set(declared), SCALAR_FEATURES)

    def test_rejects_unknown_feature_entity_and_operator(self):
        package = raw_package()
        package["rules"][0]["feature_bindings"][0]["feature"] = "entity.box_area"
        with self.assertRaises(RuleSchemaError):
            parse_rule_package(package, {"object"})

        package = raw_package()
        package["rules"][0]["feature_bindings"][0]["entity"] = "invented"
        with self.assertRaises(RuleSchemaError):
            parse_rule_package(package, {"object"})

        package = raw_package()
        package["rules"][0]["condition"]["op"] = "execute_python"
        with self.assertRaises(RuleSchemaError):
            parse_rule_package(package, {"object"})

    def test_persistence_and_evidence_use_real_values(self):
        definition = parse_rule_package(raw_package(duration=1.0, samples=2), {"object"}).rules[0]
        rule = DeclarativeRule(definition)
        first = snapshot(0.0, 0.1)
        second = snapshot(1.0, 0.1)
        self.assertEqual(rule.update(FeatureContext(first, (first,))).status, "unknown")
        finding = rule.update(FeatureContext(second, (first, second)))
        self.assertEqual(finding.status, "confirmed")
        self.assertEqual(finding.failure_type, "object_not_grasped")
        self.assertEqual(finding.evidence["values"]["gripper_opening"], 0.1)
        self.assertIn("gripper_opening", finding.evidence["feature_reasons"])
        self.assertEqual(finding.evidence["thresholds"]["condition"]["value"], 0.2)
        self.assertIn("duration=1.000s", finding.evidence["explanation"])

    def test_missing_feature_is_unknown_not_failure(self):
        definition = parse_rule_package(raw_package(), {"object"}).rules[0]
        finding = DeclarativeRule(definition).update(FeatureContext(snapshot(0, None), (snapshot(0, None),)))
        self.assertEqual(finding.status, "unknown")
        self.assertIsNone(finding.evidence["values"]["gripper_opening"])

    def test_changed_and_stable_operators(self):
        changed = {"op": "changed", "feature": "gripper_opening", "delta": 0.5,
                   "direction": "decrease", "window_sec": 1.0}
        definition = parse_rule_package(raw_package(condition=changed), {"object"}).rules[0]
        history = (snapshot(0, 0.9), snapshot(0.5, 0.6), snapshot(1, 0.2))
        self.assertEqual(DeclarativeRule(definition).update(FeatureContext(history[-1], history)).status,
                         "confirmed")

        stable = {"op": "stable", "feature": "gripper_opening", "tolerance": 0.02,
                  "window_sec": 1.0}
        definition = parse_rule_package(raw_package(condition=stable), {"object"}).rules[0]
        history = (snapshot(0, 0.1), snapshot(0.5, 0.11), snapshot(1, 0.1))
        self.assertEqual(DeclarativeRule(definition).update(FeatureContext(history[-1], history)).status,
                         "confirmed")

    def test_planner_regenerates_after_schema_error(self):
        responses = [
            {"targets": [{"target_id": "object", "name": "bottle", "role": "manipulated_object"}],
             "rule_package": {"schema_version": "bad"}},
            {"targets": [{"target_id": "object", "name": "bottle", "role": "manipulated_object"}],
             "rule_package": raw_package()},
        ]
        calls = []

        def infer(payload):
            calls.append(payload)
            return responses[len(calls) - 1]

        planner = VLMScenePlanner(infer, require_rules=True, max_attempts=2)
        frame = Frame("external", 0, 0, object())
        plan = planner.plan(Task("t", "pick bottle"), SceneSnapshot(0, {"external": frame}))
        self.assertEqual(plan.status, "complete")
        self.assertEqual(len(plan.rule_package.rules), 1)
        self.assertIn("validation_error", calls[1])

    def test_rule_engine_compiles_generated_package(self):
        package = parse_rule_package(raw_package(), {"object"})
        engine = StreamingRuleEngine()
        plan = TrackingPlan("complete", (
            TrackingTarget("object", "bottle", "manipulated_object"),
        ), "test")
        engine.initialize(Task("t", "pick bottle"), plan, package)
        current = snapshot(0, 0.1)
        report = engine.update(FeatureContext(current, (current,)))
        self.assertTrue(report.failure_detected)
        self.assertEqual(report.findings[0].rule_id, "grasp_failed")


if __name__ == "__main__":
    unittest.main()
