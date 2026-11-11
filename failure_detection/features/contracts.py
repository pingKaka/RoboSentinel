from __future__ import annotations

from dataclasses import dataclass


FEATURE_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class FeatureDefinition:
    path: str
    dtype: str
    unit: str | None
    nullable: bool
    description: str
    enum: tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraQuality:
    camera: str
    valid: bool
    width_px: int | None
    height_px: int | None


@dataclass(frozen=True)
class FeatureQuality:
    vision_valid: bool
    robot_state_valid: bool
    cameras: tuple[CameraQuality, ...]
    timing_sec: dict[str, float]


@dataclass(frozen=True)
class CameraEntityFeature:
    camera: str
    visibility: str
    mask_centroid_xy: tuple[float, float] | None
    mask_area_ratio: float | None
    mask_perimeter_normalized: float | None
    mask_component_count: int | None
    mask_compactness: float | None
    tracking_quality: float | None


@dataclass(frozen=True)
class EntityFeature:
    target_id: str
    name: str
    role: str
    cameras: tuple[CameraEntityFeature, ...]
    visible_camera_count: int
    visible_any: bool | None
    missing_all_cameras: bool | None


@dataclass(frozen=True)
class SpatialRelationFeature:
    camera: str
    entity_a: str
    entity_b: str
    delta_xy: tuple[float, float]
    centroid_distance: float
    mask_iou: float
    mask_intersection_ratio: float
    mask_coverage_a: float
    mask_coverage_b: float


@dataclass(frozen=True)
class RobotFeature:
    tcp_position_m: tuple[float, float, float] | None
    tcp_frame: str | None
    gripper_position_normalized: float | None


@dataclass(frozen=True)
class FeatureSnapshot:
    schema_version: str
    timestamp_sec: float
    delta_sec: float | None
    quality: FeatureQuality
    entities: tuple[EntityFeature, ...]
    relations: tuple[SpatialRelationFeature, ...]
    robot: RobotFeature

    def entity(self, target_id: str) -> EntityFeature | None:
        return next((item for item in self.entities if item.target_id == target_id), None)


@dataclass(frozen=True)
class FeatureContext:
    snapshot: FeatureSnapshot
    history: tuple[FeatureSnapshot, ...]
