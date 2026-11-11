from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import cv2
import numpy as np

from .contracts import (
    FEATURE_SCHEMA_VERSION,
    CameraEntityFeature,
    CameraQuality,
    EntityFeature,
    FeatureQuality,
    FeatureSnapshot,
    RobotFeature,
    SpatialRelationFeature,
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _dimensions(frame) -> tuple[int, int] | None:
    shape = getattr(frame.image, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (width, height) if width > 0 and height > 0 else None


def _binary_mask(mask, dimensions):
    if mask is None or dimensions is None:
        return None
    array = np.asarray(mask)
    array = np.squeeze(array)
    width, height = dimensions
    if array.ndim != 2 or array.shape != (height, width):
        return None
    binary = np.ascontiguousarray(array > 0, dtype=np.uint8)
    if not np.any(binary):
        return None
    return binary


def _mask_geometry(mask):
    if mask is None:
        return None, None, None, None, None
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    area_px = len(xs)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_px = sum(cv2.arcLength(contour, True) for contour in contours)
    component_count = max(0, cv2.connectedComponents(mask, connectivity=8)[0] - 1)
    compactness = 4 * math.pi * area_px / (perimeter_px * perimeter_px) if perimeter_px else 1.0
    return (
        ((float(xs.mean()) + 0.5) / width, (float(ys.mean()) + 0.5) / height),
        area_px / (width * height),
        perimeter_px / math.hypot(width, height),
        component_count,
        min(1.0, compactness),
    )


def _relation(camera, entity_a, mask_a, entity_b, mask_b):
    center_a = _mask_geometry(mask_a)[0]
    center_b = _mask_geometry(mask_b)[0]
    dx, dy = center_b[0] - center_a[0], center_b[1] - center_a[1]
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = area_a + area_b - intersection
    return SpatialRelationFeature(
        camera, entity_a, entity_b, (dx, dy), math.hypot(dx, dy),
        intersection / union if union else 0.0, intersection / mask_a.size,
        intersection / area_a if area_a else 0.0,
        intersection / area_b if area_b else 0.0,
    )


class StandardFeatureExtractor:
    def __init__(self, required_cameras=("external", "wrist")) -> None:
        self.required_cameras = tuple(required_cameras)
        self._previous_timestamp = None

    def reset(self) -> None:
        self._previous_timestamp = None

    def extract(self, plan, observation, perception) -> FeatureSnapshot:
        now = float(observation.timestamp_sec)
        delta = None if self._previous_timestamp is None else now - self._previous_timestamp
        if delta is not None and (not math.isfinite(delta) or delta < 0):
            delta = None
        self._previous_timestamp = now

        camera_names = sorted(set(self.required_cameras) | set(observation.frames))
        dimensions = {camera: (_dimensions(observation.frames[camera])
                               if camera in observation.frames else None)
                      for camera in camera_names}
        cameras = tuple(CameraQuality(camera, size is not None,
                                      size[0] if size else None, size[1] if size else None)
                        for camera, size in sorted(dimensions.items()))
        tracks = {(track.target_id, track.camera): track for track in perception.tracks}
        entities = []
        masks = {}
        for target in plan.targets:
            camera_features = []
            for camera in sorted(dimensions):
                track = tracks.get((target.target_id, camera))
                if (perception.status != "complete" or dimensions[camera] is None or track is None
                        or not track.observation_valid):
                    visibility, mask, quality = "unknown", None, None
                elif track.visible:
                    mask = _binary_mask(track.mask, dimensions[camera])
                    visibility = "visible" if mask is not None else "unknown"
                    quality = _finite(track.quality) if mask is not None else None
                else:
                    visibility, mask = "not_visible", None
                    quality = _finite(track.quality)
                if quality is not None and not 0 <= quality <= 1:
                    quality = None
                centroid, area, perimeter, components, compactness = _mask_geometry(mask)
                camera_features.append(CameraEntityFeature(
                    camera, visibility, centroid, area, perimeter, components, compactness, quality,
                ))
                if mask is not None:
                    masks[(camera, target.target_id)] = mask
            states = [item.visibility for item in camera_features]
            visible_count = states.count("visible")
            if visible_count:
                visible_any = True
            elif states and all(state == "not_visible" for state in states):
                visible_any = False
            else:
                visible_any = None
            entities.append(EntityFeature(
                target.target_id, target.name, target.role, tuple(camera_features), visible_count,
                visible_any, None if visible_any is None else not visible_any,
            ))

        relations = []
        for camera in sorted(dimensions):
            visible_ids = [target.target_id for target in plan.targets if (camera, target.target_id) in masks]
            for entity_a, entity_b in combinations(visible_ids, 2):
                relations.append(_relation(camera, entity_a, masks[(camera, entity_a)],
                                           entity_b, masks[(camera, entity_b)]))

        robot = self._robot_feature(observation.robot_state)
        timing = {str(key): number for key, value in observation.synchronization.items()
                  if (number := _finite(value)) is not None and str(key).endswith("_sec")}
        quality = FeatureQuality(
            perception.status == "complete" and bool(cameras) and all(camera.valid for camera in cameras),
            robot.tcp_position_m is not None or robot.gripper_position_normalized is not None,
            cameras, timing,
        )
        return FeatureSnapshot(
            FEATURE_SCHEMA_VERSION, now, delta, quality, tuple(entities), tuple(relations), robot,
        )

    @staticmethod
    def _robot_feature(state) -> RobotFeature:
        state = state or {}
        position = state.get("tcp_position")
        frame = state.get("tcp_position_frame")
        if position is None:
            position = state.get("tcp_pose")
            frame = frame or state.get("tcp_pose_frame")
        if position is None:
            position = state.get("teleop_ee_pose")
            frame = frame or state.get("teleop_ee_pose_frame")
        try:
            xyz = tuple(_finite(value) for value in position[:3])
            tcp = xyz if len(xyz) == 3 and all(value is not None for value in xyz) else None
        except (TypeError, IndexError):
            tcp = None
        gripper = _finite(state.get("gripper_position_normalized"))
        if gripper is not None and not 0 <= gripper <= 1:
            gripper = None
        return RobotFeature(tcp, str(frame) if tcp is not None and frame else None, gripper)
