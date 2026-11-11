from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RULE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureBinding:
    alias: str
    feature: str
    reason: str
    entity: str | None = None
    entities: tuple[str, str] | None = None
    camera: str | None = None


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    failure_type: str
    constraint_type: str
    description: str
    feature_bindings: tuple[FeatureBinding, ...]
    activation: dict[str, Any]
    condition: dict[str, Any]
    duration_sec: float
    minimum_samples: int
    unknown_policy: str
    explanation_template: str


@dataclass(frozen=True)
class RulePackage:
    schema_version: str
    task_type: str
    rules: tuple[RuleDefinition, ...]

