"""Versioned standard features exposed to monitoring rules."""

from .catalog import FEATURE_CATALOG, feature_catalog
from .contracts import (
    FEATURE_SCHEMA_VERSION,
    CameraEntityFeature,
    CameraQuality,
    EntityFeature,
    FeatureContext,
    FeatureDefinition,
    FeatureQuality,
    FeatureSnapshot,
    RobotFeature,
    SpatialRelationFeature,
)
from .extractor import StandardFeatureExtractor

__all__ = [
    "FEATURE_CATALOG",
    "FEATURE_SCHEMA_VERSION",
    "CameraEntityFeature",
    "CameraQuality",
    "EntityFeature",
    "FeatureContext",
    "FeatureDefinition",
    "FeatureQuality",
    "FeatureSnapshot",
    "RobotFeature",
    "SpatialRelationFeature",
    "StandardFeatureExtractor",
    "feature_catalog",
]
