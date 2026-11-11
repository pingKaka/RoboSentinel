from __future__ import annotations

from dataclasses import asdict

from .contracts import FeatureDefinition


FEATURE_CATALOG = (
    FeatureDefinition("schema_version", "string", None, False, "Standard feature schema version."),
    FeatureDefinition("timestamp_sec", "number", "s", False, "Monotonic observation timestamp."),
    FeatureDefinition("delta_sec", "number", "s", True, "Time since the previous feature snapshot."),
    FeatureDefinition("quality.vision_valid", "boolean", None, False, "Tracker and camera data are usable."),
    FeatureDefinition("quality.robot_state_valid", "boolean", None, False, "At least one standard robot feature is present."),
    FeatureDefinition("quality.cameras.*.camera", "string", None, False, "Camera identifier."),
    FeatureDefinition("quality.cameras.*.valid", "boolean", None, False, "Camera frame dimensions are valid."),
    FeatureDefinition("quality.cameras.*.width_px", "integer", "px", True, "Source frame width."),
    FeatureDefinition("quality.cameras.*.height_px", "integer", "px", True, "Source frame height."),
    FeatureDefinition("quality.timing_sec.*", "number", "s", True, "Input timing or synchronization measurement."),
    FeatureDefinition("entities.*.target_id", "string", None, False, "Stable entity identifier."),
    FeatureDefinition("entities.*.name", "string", None, False, "Visual entity name."),
    FeatureDefinition("entities.*.role", "string", None, False, "Task role of the entity."),
    FeatureDefinition("entities.*.cameras.*.camera", "string", None, False, "Camera identifier."),
    FeatureDefinition(
        "entities.*.cameras.*.visibility", "string", None, False,
        "Tri-state visibility.", ("visible", "not_visible", "unknown"),
    ),
    FeatureDefinition("entities.*.cameras.*.mask_centroid_xy", "vector2", "normalized", True, "Centroid of foreground mask pixels."),
    FeatureDefinition("entities.*.cameras.*.mask_area_ratio", "number", "ratio", True, "Foreground mask pixels divided by frame pixels."),
    FeatureDefinition("entities.*.cameras.*.mask_perimeter_normalized", "number", "normalized", True, "Mask contour perimeter divided by frame diagonal."),
    FeatureDefinition("entities.*.cameras.*.mask_component_count", "integer", "count", True, "Number of connected foreground components."),
    FeatureDefinition("entities.*.cameras.*.mask_compactness", "number", "ratio", True, "Mask compactness 4*pi*area/perimeter^2."),
    FeatureDefinition("entities.*.cameras.*.tracking_quality", "number", "ratio", True, "Tracker quality in [0, 1]."),
    FeatureDefinition("entities.*.visible_camera_count", "integer", "count", False, "Number of cameras seeing the entity."),
    FeatureDefinition("entities.*.visible_any", "boolean", None, True, "Visible in at least one camera."),
    FeatureDefinition("entities.*.missing_all_cameras", "boolean", None, True, "Known absent from every camera."),
    FeatureDefinition("relations.*.camera", "string", None, False, "Camera containing both entities."),
    FeatureDefinition("relations.*.entity_a", "string", None, False, "First entity identifier."),
    FeatureDefinition("relations.*.entity_b", "string", None, False, "Second entity identifier."),
    FeatureDefinition("relations.*.delta_xy", "vector2", "normalized", False, "Center of entity B minus entity A."),
    FeatureDefinition("relations.*.centroid_distance", "number", "normalized", False, "Same-camera center distance."),
    FeatureDefinition("relations.*.mask_iou", "number", "ratio", False, "Exact same-camera binary-mask IoU."),
    FeatureDefinition("relations.*.mask_intersection_ratio", "number", "ratio", False, "Mask intersection pixels divided by frame pixels."),
    FeatureDefinition("relations.*.mask_coverage_a", "number", "ratio", False, "Mask intersection divided by entity A mask area."),
    FeatureDefinition("relations.*.mask_coverage_b", "number", "ratio", False, "Mask intersection divided by entity B mask area."),
    FeatureDefinition("robot.tcp_position_m", "vector3", "m", True, "TCP xyz position."),
    FeatureDefinition("robot.tcp_frame", "string", None, True, "Coordinate frame of the TCP position."),
    FeatureDefinition("robot.gripper_position_normalized", "number", "normalized", True, "Gripper opening position in [0, 1]."),
)


def feature_catalog() -> list[dict]:
    return [asdict(item) for item in FEATURE_CATALOG]
