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
    SCALAR_FEATURES, DeclarativeRule, RuleSchemaError, compile_authoring_plan,
    merge_rule_packages, parse_monitoring_plan, parse_rule_package,
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
    def test_compact_authoring_plan_compiles_to_runtime_bindings(self):
        authored = {
            "targets": [{"target_id": "object", "name": "bottle",
                         "role": "manipulated_object"}],
            "task_type": "pick",
            "rules": [{
                "rule_id": "grasp_failed", "failure_type": "grasp_failed",
                "constraint_type": "during_execution", "description": "Object did not move.",
                "activation": {"feature": "robot.gripper_position_normalized", "op": "changed",
                               "delta": 0.15, "direction": "any", "window_sec": 1.0,
                               "reason": "A gripper action starts checking."},
                "condition": {"feature": "entity.camera.mask_centroid_x", "entity": "object",
                              "camera": "external", "op": "stable", "tolerance": 0.02,
                              "window_sec": 1.0, "reason": "Stable centroid means no motion."},
                "duration_sec": 0.5, "minimum_samples": 3, "unknown_policy": "unknown",
                "explanation_template": "The gripper acted but the object did not move.",
            }],
        }
        plan = compile_authoring_plan(authored, "test")
        rule = plan.rule_package.rules[0]
        self.assertEqual([item.alias for item in rule.feature_bindings],
                         ["evidence_0", "evidence_1"])
        self.assertEqual(rule.activation["feature"], "evidence_0")
        self.assertEqual(rule.condition["feature"], "evidence_1")

    def test_compact_authoring_accepts_explicit_relation_entities(self):
        authored = {
            "targets": [
                {"target_id": "object", "name": "bottle", "role": "manipulated_object"},
                {"target_id": "cup", "name": "black cup", "role": "destination"},
            ],
            "task_type": "place",
            "rules": [{
                "rule_id": "place_failed", "failure_type": "place_failed",
                "constraint_type": "upon_completion", "description": "Bottle is outside cup.",
                "activation": {"feature": "entity.visible_any", "entity": "object", "op": "eq",
                               "value": True, "reason": "Bottle is observable."},
                "condition": {"feature": "relation.mask_iou", "entity_a": "object",
                              "entity_b": "cup", "camera": "external", "op": "lt",
                              "value": 0.5, "reason": "Low overlap means placement failed."},
                "duration_sec": 0.5, "minimum_samples": 3, "unknown_policy": "unknown",
                "explanation_template": "Bottle does not overlap cup.",
            }],
        }
        plan = compile_authoring_plan(authored, "test")
        self.assertEqual(plan.rule_package.rules[0].feature_bindings[1].entities,
                         ("object", "cup"))

    def test_compact_authoring_rejects_same_activation_and_condition(self):
        evidence = {"feature": "entity.visible_any", "entity": "object", "op": "eq",
                    "value": True, "reason": "Object is visible."}
        authored = {
            "targets": [{"target_id": "object", "name": "bottle",
                         "role": "manipulated_object"}],
            "task_type": "pick",
            "rules": [{
                "rule_id": "bad", "failure_type": "bad", "constraint_type": "during_execution",
                "description": "Bad rule.", "activation": evidence,
                "condition": {**evidence, "reason": "Same fact with different prose."},
                "duration_sec": 0.0, "minimum_samples": 1, "unknown_policy": "unknown",
                "explanation_template": "Bad.",
            }],
        }
        with self.assertRaises(ValueError):
            compile_authoring_plan(authored, "test")

    def test_catalog_feature_in_expression_is_canonicalized_when_unambiguous(self):
        package = raw_package()
        package["rules"][0]["activation"]["feature"] = "robot.gripper_position_normalized"
        package["rules"][0]["condition"] = {
            "op": "eq", "feature": "object_visible", "value": True,
        }
        parsed = parse_rule_package(package, {"object"})
        self.assertEqual(parsed.rules[0].activation["feature"], "gripper_opening")

    def test_catalog_feature_in_expression_is_rejected_when_ambiguous(self):
        package = raw_package()
        duplicate = dict(package["rules"][0]["feature_bindings"][1])
        duplicate["alias"] = "second_gripper_motion"
        package["rules"][0]["feature_bindings"].append(duplicate)
        package["rules"][0]["activation"]["feature"] = "robot.gripper_position_normalized"
        with self.assertRaises(RuleSchemaError):
            parse_rule_package(package, {"object"})

    def test_expression_rejects_parameters_from_another_operator(self):
        package = raw_package()
        package["rules"][0]["activation"]["window_sec"] = 1.0
        with self.assertRaises(RuleSchemaError):
            parse_rule_package(package, {"object"})

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
        self.assertEqual(calls[1]["previous_output"], responses[0])

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

    def test_manual_mode_does_not_call_vlm(self):
        raw = {
            "targets": [{"target_id": "object", "name": "bottle",
                         "role": "manipulated_object"}],
            "rule_package": raw_package(),
        }
        manual = parse_monitoring_plan(raw, "manual")
        planner = VLMScenePlanner(lambda _: self.fail("manual mode called VLM"),
                                  manual_plan=manual, mode="manual")
        frame = Frame("external", 0, 0, object())
        result = planner.plan(Task("t", "pick bottle"), SceneSnapshot(0, {"external": frame}))
        self.assertEqual(result.rule_package.rules[0].rule_id, "grasp_failed")

    def test_merged_mode_preserves_targets_and_appends_rules(self):
        manual_raw = {
            "targets": [{"target_id": "object", "name": "bottle",
                         "role": "manipulated_object"}],
            "rule_package": raw_package(),
        }
        manual = parse_monitoring_plan(manual_raw, "manual", max_rules=3)
        generated_package = raw_package()
        generated_package["rules"][0]["rule_id"] = "target_lost"
        generated_package["rules"][0]["failure_type"] = "target_unobservable"
        calls = []

        def infer(payload):
            calls.append(payload)
            return {"targets": manual_raw["targets"], "rule_package": generated_package}

        planner = VLMScenePlanner(infer, manual_plan=manual, mode="merged",
                                  require_rules=True, max_rules=3)
        frame = Frame("external", 0, 0, object())
        result = planner.plan(Task("t", "pick bottle"), SceneSnapshot(0, {"external": frame}))
        self.assertEqual([item.rule_id for item in result.rule_package.rules],
                         ["target_lost", "grasp_failed"])
        self.assertEqual(calls[0]["fixed_targets"], manual_raw["targets"])

    def test_merge_rejects_duplicate_rule_ids(self):
        package = parse_rule_package(raw_package(), {"object"})
        with self.assertRaises(ValueError):
            merge_rule_packages(package, package, 6)


if __name__ == "__main__":
    unittest.main()
