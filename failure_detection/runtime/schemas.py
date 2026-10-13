from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..state.base import RobotStateMessage
from ..state.features import RobotStateFeatures


@dataclass(frozen=True)
class VisionObservation:
    timestamp_sec: float
    external_frame_index: int | None = None
    wrist_frame_index: int | None = None
    # Detector-specific values stay opaque until a calibrated plugin consumes them.
    features: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedObservation:
    timestamp_sec: float
    robot_state: RobotStateMessage | None
    robot_features: RobotStateFeatures | None
    vision: VisionObservation | None
    synchronization_error_sec: dict[str, float]
    valid: dict[str, bool]
