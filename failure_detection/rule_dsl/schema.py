from __future__ import annotations

import math
import re
from typing import Any

from .contracts import RULE_SCHEMA_VERSION, FeatureBinding, RuleDefinition, RulePackage


SCALAR_FEATURES = {
    "entity.visible_any",
    "entity.missing_all_cameras",
    "entity.visible_camera_count",
    "entity.camera.visibility",
    "entity.camera.mask_centroid_x",
    "entity.camera.mask_centroid_y",
    "entity.camera.mask_area_ratio",
    "entity.camera.mask_perimeter_normalized",
    "entity.camera.mask_component_count",
    "entity.camera.mask_compactness",
    "entity.camera.tracking_quality",
    "relation.delta_x",
    "relation.delta_y",
    "relation.centroid_distance",
    "relation.mask_iou",
    "relation.mask_intersection_ratio",
    "relation.mask_coverage_a",
    "relation.mask_coverage_b",
    "robot.tcp_x",
    "robot.tcp_y",
    "robot.tcp_z",
    "robot.gripper_position_normalized",
}
ENTITY_FEATURES = {item for item in SCALAR_FEATURES if item.startswith("entity.")}
RELATION_FEATURES = {item for item in SCALAR_FEATURES if item.startswith("relation.")}
COMPARISON_OPERATORS = {"eq", "gt", "lt", "between"}
TEMPORAL_OPERATORS = {"changed", "stable"}
BOOLEAN_OPERATORS = {"all", "any"}
OPERATORS = COMPARISON_OPERATORS | TEMPORAL_OPERATORS | BOOLEAN_OPERATORS
TASK_TYPES = {"pick", "transport", "place", "pick_and_place", "other"}
CONSTRAINT_TYPES = {"during_execution", "upon_completion"}
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RuleSchemaError(ValueError):
    pass


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuleSchemaError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuleSchemaError(f"{label} must be a nonempty string")
    return value.strip()


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RuleSchemaError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuleSchemaError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise RuleSchemaError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise RuleSchemaError(f"{label} must be >= {minimum}")
    return result


def _validate_expression(value: Any, aliases: set[str], label: str) -> dict[str, Any]:
    expression = _mapping(value, label)
    op = expression.get("op")
    if op not in OPERATORS:
        raise RuleSchemaError(f"{label}.op must be one of {sorted(OPERATORS)}")
    if op in BOOLEAN_OPERATORS:
        unknown = set(expression) - {"op", "args"}
        if unknown:
            raise RuleSchemaError(f"{label} has unsupported fields {sorted(unknown)}")
        args = expression.get("args")
        if not isinstance(args, list) or not args:
            raise RuleSchemaError(f"{label}.args must be a nonempty list")
        for index, item in enumerate(args):
            _validate_expression(item, aliases, f"{label}.args[{index}]")
        return dict(expression)

    alias = expression.get("feature")
    if alias not in aliases:
        raise RuleSchemaError(f"{label}.feature references unknown binding {alias!r}")
    allowed = {"op", "feature"}
    if op in {"eq", "gt", "lt"}:
        allowed.add("value")
    elif op == "between":
        allowed.update({"min", "max"})
    elif op == "changed":
        allowed.update({"delta", "direction", "window_sec"})
    elif op == "stable":
        allowed.update({"tolerance", "window_sec"})
    unknown = set(expression) - allowed
    if unknown:
        raise RuleSchemaError(f"{label} has unsupported fields {sorted(unknown)}")
    if op in {"eq", "gt", "lt"} and "value" not in expression:
        raise RuleSchemaError(f"{label}.value is required")
    if op in {"gt", "lt"}:
        _number(expression.get("value"), f"{label}.value")
    if op == "between":
        lower = _number(expression.get("min"), f"{label}.min")
        upper = _number(expression.get("max"), f"{label}.max")
        if lower > upper:
            raise RuleSchemaError(f"{label}.min must not exceed max")
    if op in TEMPORAL_OPERATORS:
        _number(expression.get("window_sec"), f"{label}.window_sec", 0.0)
        _number(expression.get("tolerance" if op == "stable" else "delta"),
                f"{label}.{'tolerance' if op == 'stable' else 'delta'}", 0.0)
        direction = expression.get("direction", "any")
        if op == "changed" and direction not in {"any", "increase", "decrease"}:
            raise RuleSchemaError(f"{label}.direction is invalid")
    return dict(expression)


def _canonicalize_expression_features(value: Any, feature_aliases: dict[str, list[str]]) -> Any:
    """Repair only an unambiguous catalog-name-for-alias transcription error."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if result.get("op") in BOOLEAN_OPERATORS:
        args = result.get("args")
        if isinstance(args, list):
            result["args"] = [
                _canonicalize_expression_features(item, feature_aliases) for item in args
            ]
        return result
    feature = result.get("feature")
    candidates = feature_aliases.get(feature, [])
    if len(candidates) == 1:
        result["feature"] = candidates[0]
    return result


def _expression_aliases(value: dict[str, Any]) -> set[str]:
    if value.get("op") in BOOLEAN_OPERATORS:
        return set().union(*(_expression_aliases(item) for item in value["args"]))
    return {value["feature"]}


def parse_rule_package(value: Any, target_ids: set[str], max_rules: int = 6) -> RulePackage:
    package = _mapping(value, "rule package")
    if package.get("schema_version") != RULE_SCHEMA_VERSION:
        raise RuleSchemaError(f"schema_version must be {RULE_SCHEMA_VERSION!r}")
    task_type = package.get("task_type")
    if task_type not in TASK_TYPES:
        raise RuleSchemaError(f"task_type must be one of {sorted(TASK_TYPES)}")
    raw_rules = package.get("rules")
    if not isinstance(raw_rules, list) or not 1 <= len(raw_rules) <= max_rules:
        raise RuleSchemaError(f"rules must contain 1 to {max_rules} entries")

    rules = []
    rule_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        label = f"rules[{index}]"
        item = _mapping(raw_rule, label)
        rule_id = _text(item.get("rule_id"), f"{label}.rule_id")
        failure_type = _text(item.get("failure_type"), f"{label}.failure_type")
        if not _IDENTIFIER.fullmatch(rule_id) or not _IDENTIFIER.fullmatch(failure_type):
            raise RuleSchemaError(f"{label} identifiers must use lowercase snake_case")
        if rule_id in rule_ids:
            raise RuleSchemaError(f"duplicate rule_id {rule_id!r}")
        rule_ids.add(rule_id)
        constraint_type = item.get("constraint_type")
        if constraint_type not in CONSTRAINT_TYPES:
            raise RuleSchemaError(f"{label}.constraint_type is invalid")

        raw_bindings = item.get("feature_bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise RuleSchemaError(f"{label}.feature_bindings must be nonempty")
        bindings = []
        aliases: set[str] = set()
        feature_aliases: dict[str, list[str]] = {}
        for binding_index, raw_binding in enumerate(raw_bindings):
            binding_label = f"{label}.feature_bindings[{binding_index}]"
            binding = _mapping(raw_binding, binding_label)
            alias = _text(binding.get("alias"), f"{binding_label}.alias")
            feature = binding.get("feature")
            if not _IDENTIFIER.fullmatch(alias) or alias in aliases:
                raise RuleSchemaError(f"{binding_label}.alias is invalid or duplicated")
            if feature not in SCALAR_FEATURES:
                raise RuleSchemaError(f"{binding_label}.feature is not in the feature catalog")
            aliases.add(alias)
            feature_aliases.setdefault(feature, []).append(alias)
            entity = binding.get("entity")
            entities = binding.get("entities")
            camera = binding.get("camera")
            if feature in ENTITY_FEATURES:
                if entity not in target_ids:
                    raise RuleSchemaError(f"{binding_label}.entity is not a tracking target")
                if ".camera." in feature and (not isinstance(camera, str) or not camera):
                    raise RuleSchemaError(f"{binding_label}.camera is required")
            elif feature in RELATION_FEATURES:
                if (not isinstance(entities, list) or len(entities) != 2
                        or entities[0] == entities[1] or any(v not in target_ids for v in entities)):
                    raise RuleSchemaError(f"{binding_label}.entities must contain two tracking targets")
                if not isinstance(camera, str) or not camera:
                    raise RuleSchemaError(f"{binding_label}.camera is required")
                entities = (entities[0], entities[1])
            bindings.append(FeatureBinding(
                alias, feature, _text(binding.get("reason"), f"{binding_label}.reason"),
                entity if isinstance(entity, str) else None, entities,
                camera if isinstance(camera, str) else None,
            ))

        activation_value = _canonicalize_expression_features(item.get("activation"), feature_aliases)
        condition_value = _canonicalize_expression_features(item.get("condition"), feature_aliases)
        activation = _validate_expression(activation_value, aliases, f"{label}.activation")
        condition = _validate_expression(condition_value, aliases, f"{label}.condition")
        unused_aliases = aliases - _expression_aliases(activation) - _expression_aliases(condition)
        if unused_aliases:
            raise RuleSchemaError(f"{label}.feature_bindings has unused aliases {sorted(unused_aliases)}")
        duration = _number(item.get("duration_sec"), f"{label}.duration_sec", 0.0)
        minimum_samples = item.get("minimum_samples")
        if (isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int)
                or not 1 <= minimum_samples <= 300):
            raise RuleSchemaError(f"{label}.minimum_samples must be a positive integer")
        if item.get("unknown_policy") != "unknown":
            raise RuleSchemaError(f"{label}.unknown_policy must be 'unknown'")
        rules.append(RuleDefinition(
            rule_id, failure_type, constraint_type,
            _text(item.get("description"), f"{label}.description"), tuple(bindings),
            activation, condition, duration, minimum_samples, "unknown",
            _text(item.get("explanation_template"), f"{label}.explanation_template"),
        ))
    return RulePackage(RULE_SCHEMA_VERSION, task_type, tuple(rules))
