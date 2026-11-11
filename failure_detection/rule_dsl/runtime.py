from __future__ import annotations

import math
from typing import Any

from ..features.contracts import FeatureContext, FeatureSnapshot
from ..pipeline.contracts import RuleFinding
from .contracts import FeatureBinding, RuleDefinition


UNKNOWN = object()


def _entity(snapshot: FeatureSnapshot, target_id: str):
    return snapshot.entity(target_id)


def resolve_feature(snapshot: FeatureSnapshot, binding: FeatureBinding):
    feature = binding.feature
    if feature.startswith("robot."):
        if feature == "robot.gripper_position_normalized":
            return snapshot.robot.gripper_position_normalized
        position = snapshot.robot.tcp_position_m
        if position is None:
            return UNKNOWN
        return position[{"robot.tcp_x": 0, "robot.tcp_y": 1, "robot.tcp_z": 2}[feature]]

    if feature.startswith("entity."):
        entity = _entity(snapshot, binding.entity or "")
        if entity is None:
            return UNKNOWN
        simple = {
            "entity.visible_any": entity.visible_any,
            "entity.missing_all_cameras": entity.missing_all_cameras,
            "entity.visible_camera_count": entity.visible_camera_count,
        }
        if feature in simple:
            return UNKNOWN if simple[feature] is None else simple[feature]
        camera = next((item for item in entity.cameras if item.camera == binding.camera), None)
        if camera is None:
            return UNKNOWN
        if feature == "entity.camera.visibility":
            return camera.visibility
        values = {
            "entity.camera.mask_area_ratio": camera.mask_area_ratio,
            "entity.camera.mask_perimeter_normalized": camera.mask_perimeter_normalized,
            "entity.camera.mask_component_count": camera.mask_component_count,
            "entity.camera.mask_compactness": camera.mask_compactness,
            "entity.camera.tracking_quality": camera.tracking_quality,
        }
        if feature in {"entity.camera.mask_centroid_x", "entity.camera.mask_centroid_y"}:
            value = camera.mask_centroid_xy
            return UNKNOWN if value is None else value[feature.endswith("_y")]
        value = values.get(feature)
        return UNKNOWN if value is None else value

    requested_a, requested_b = binding.entities or ("", "")
    relation = next((item for item in snapshot.relations
                     if item.camera == binding.camera
                     and {item.entity_a, item.entity_b} == {requested_a, requested_b}), None)
    if relation is None:
        return UNKNOWN
    reverse = relation.entity_a != requested_a
    values = {
        "relation.centroid_distance": relation.centroid_distance,
        "relation.mask_iou": relation.mask_iou,
        "relation.mask_intersection_ratio": relation.mask_intersection_ratio,
        "relation.mask_coverage_a": relation.mask_coverage_b if reverse else relation.mask_coverage_a,
        "relation.mask_coverage_b": relation.mask_coverage_a if reverse else relation.mask_coverage_b,
        "relation.delta_x": -relation.delta_xy[0] if reverse else relation.delta_xy[0],
        "relation.delta_y": -relation.delta_xy[1] if reverse else relation.delta_xy[1],
    }
    return values[feature]


def _comparable_number(value: Any):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class ExpressionEvaluator:
    def __init__(self, bindings: tuple[FeatureBinding, ...]):
        self.bindings = {item.alias: item for item in bindings}

    def evaluate(self, expression: dict[str, Any], context: FeatureContext):
        op = expression["op"]
        if op in {"all", "any"}:
            values = [self.evaluate(item, context) for item in expression["args"]]
            if op == "all":
                return False if False in values else UNKNOWN if UNKNOWN in values else True
            return True if True in values else UNKNOWN if UNKNOWN in values else False
        binding = self.bindings[expression["feature"]]
        if op in {"changed", "stable"}:
            return self._temporal(op, binding, expression, context)
        actual = resolve_feature(context.snapshot, binding)
        if actual is UNKNOWN:
            return UNKNOWN
        expected = expression.get("value")
        if op == "eq":
            return actual == expected
        number = _comparable_number(actual)
        if number is None:
            return UNKNOWN
        if op == "gt":
            return number > float(expected)
        if op == "lt":
            return number < float(expected)
        return float(expression["min"]) <= number <= float(expression["max"])

    @staticmethod
    def _temporal(op, binding, expression, context):
        now = context.snapshot.timestamp_sec
        start = now - float(expression["window_sec"])
        samples = []
        for snapshot in context.history:
            if snapshot.timestamp_sec < start or snapshot.timestamp_sec > now:
                continue
            value = resolve_feature(snapshot, binding)
            number = _comparable_number(value)
            if number is not None:
                samples.append((snapshot.timestamp_sec, number))
        if len(samples) < 2 or samples[-1][0] - samples[0][0] < float(expression["window_sec"]):
            return UNKNOWN
        change = samples[-1][1] - samples[0][1]
        if op == "stable":
            values = [item[1] for item in samples]
            return max(values) - min(values) <= float(expression["tolerance"])
        delta = float(expression["delta"])
        direction = expression.get("direction", "any")
        return abs(change) >= delta if direction == "any" else change >= delta if direction == "increase" else change <= -delta


class DeclarativeRule:
    def __init__(self, definition: RuleDefinition):
        self.definition = definition
        self.rule_id = definition.rule_id
        self.evaluator = ExpressionEvaluator(definition.feature_bindings)
        self._started_at: float | None = None
        self._sample_count = 0

    def update(self, context: FeatureContext) -> RuleFinding:
        now = context.snapshot.timestamp_sec
        activation = self.evaluator.evaluate(self.definition.activation, context)
        condition = self.evaluator.evaluate(self.definition.condition, context) if activation is True else UNKNOWN
        values = {binding.alias: resolve_feature(context.snapshot, binding)
                  for binding in self.definition.feature_bindings}
        serializable_values = {key: None if value is UNKNOWN else value for key, value in values.items()}
        evidence = {
            "constraint": self.definition.description,
            "constraint_type": self.definition.constraint_type,
            "activation": None if activation is UNKNOWN else activation,
            "condition": None if condition is UNKNOWN else condition,
            "values": serializable_values,
            "feature_reasons": {item.alias: item.reason for item in self.definition.feature_bindings},
            "thresholds": {"activation": self.definition.activation, "condition": self.definition.condition},
        }
        if activation is UNKNOWN or activation is True and condition is UNKNOWN:
            self._reset()
            unavailable = [key for key, value in values.items() if value is UNKNOWN]
            evidence["unavailable_features"] = unavailable
            evidence["explanation"] = (
                "Required monitoring evidence is unavailable: " + ", ".join(unavailable)
                if unavailable else "The activation or condition cannot yet be evaluated."
            )
            return RuleFinding(self.rule_id, "unknown", evidence=evidence)
        if activation is False or condition is False:
            self._reset()
            evidence["explanation"] = "Constraint is inactive or currently satisfied."
            return RuleFinding(self.rule_id, "clear", evidence=evidence)

        if self._started_at is None:
            self._started_at = now
        self._sample_count += 1
        elapsed = max(0.0, now - self._started_at)
        evidence.update({"duration_sec": elapsed, "sample_count": self._sample_count})
        if elapsed < self.definition.duration_sec or self._sample_count < self.definition.minimum_samples:
            evidence["explanation"] = "Potential violation is waiting for temporal confirmation."
            return RuleFinding(self.rule_id, "unknown", evidence=evidence)
        evidence["explanation"] = (
            f"{self.definition.explanation_template} Observed values={serializable_values}; "
            f"condition={self.definition.condition}; duration={elapsed:.3f}s; "
            f"samples={self._sample_count}."
        )
        return RuleFinding(self.rule_id, "confirmed", self.definition.failure_type, 1.0, evidence)

    def _reset(self):
        self._started_at = None
        self._sample_count = 0


def compile_rules(definitions: tuple[RuleDefinition, ...]):
    return tuple(DeclarativeRule(item) for item in definitions)
