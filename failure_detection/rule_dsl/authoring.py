"""Compile the compact VLM authoring format into the runtime rule DSL."""
from __future__ import annotations

from typing import Any

from .contracts import RULE_SCHEMA_VERSION
from .plans import parse_monitoring_plan
from .schema import BOOLEAN_OPERATORS, SCALAR_FEATURES


def _compile_evidence(value: Any, bindings: list[dict[str, Any]], counter: list[int]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("evidence must be an object")
    op = value.get("op")
    if op in BOOLEAN_OPERATORS:
        args = value.get("args")
        if not isinstance(args, list) or not args:
            raise ValueError(f"{op} evidence requires nonempty args")
        return {"op": op, "args": [_compile_evidence(item, bindings, counter) for item in args]}

    feature = value.get("feature")
    if feature == "entity.camera.mask_iou":
        feature = "relation.mask_iou"
    if feature not in SCALAR_FEATURES:
        raise ValueError(f"unknown feature {feature!r}")
    alias = f"evidence_{counter[0]}"
    counter[0] += 1
    binding = {"alias": alias, "feature": feature, "reason": value.get("reason")}
    for key in ("entity", "entities", "camera"):
        if key in value:
            binding[key] = value[key]
    if "entities" not in binding and "entity_a" in value and "entity_b" in value:
        binding["entities"] = [value["entity_a"], value["entity_b"]]
    bindings.append(binding)
    expression = {"op": op, "feature": alias}
    for key in ("value", "min", "max", "delta", "direction",
                "window_sec", "tolerance"):
        if key in value:
            expression[key] = value[key]
    return expression


def compile_authoring_plan(value: Any, planner: str, max_rules: int = 6):
    """Translate direct evidence declarations and run the normal strict parser."""
    if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
        raise ValueError("authoring plan requires targets and rules")
    compiled_rules = []
    for raw in value["rules"]:
        if not isinstance(raw, dict):
            raise ValueError("authoring rule must be an object")
        bindings: list[dict[str, Any]] = []
        counter = [0]
        activation_source = raw.get("activation")
        condition_source = raw.get("condition")
        if isinstance(activation_source, dict) and isinstance(condition_source, dict):
            activation_fact = {key: item for key, item in activation_source.items()
                               if key != "reason"}
            condition_fact = {key: item for key, item in condition_source.items()
                              if key != "reason"}
            if activation_fact == condition_fact:
                raise ValueError("activation and condition must be different observable facts")
        activation = _compile_evidence(activation_source, bindings, counter)
        condition = _compile_evidence(condition_source, bindings, counter)
        compiled_rules.append({
            **{key: raw.get(key) for key in (
                "rule_id", "failure_type", "constraint_type", "description",
                "duration_sec", "minimum_samples", "unknown_policy",
                "explanation_template",
            )},
            "feature_bindings": bindings,
            "activation": activation,
            "condition": condition,
        })
    compiled = {
        "targets": value.get("targets"),
        "rule_package": {
            "schema_version": RULE_SCHEMA_VERSION,
            "task_type": value.get("task_type"),
            "rules": compiled_rules,
        },
    }
    return parse_monitoring_plan(compiled, planner, max_rules)
