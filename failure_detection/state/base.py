from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotStateMessage:
    timestamp_sec: float
    sequence_id: int
    actual_tcp_position: np.ndarray
    actual_tcp_quaternion: np.ndarray
    actual_tcp_velocity: np.ndarray
    target_tcp_position: np.ndarray | None = None
    target_tcp_velocity: np.ndarray | None = None
    joint_position: np.ndarray | None = None
    joint_velocity: np.ndarray | None = None
    gripper_width: float | None = None
    gripper_command: float | None = None
    gripper_force: float | None = None
    tcp_wrench: np.ndarray | None = None
    controller_state: str | None = None
    active_skill: str | None = None


@dataclass(frozen=True)
class RobotStateSequence:
    messages: tuple[RobotStateMessage, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Robot state sequence is empty")
        timestamps = np.asarray([item.timestamp_sec for item in self.messages])
        if not np.isfinite(timestamps).all():
            raise ValueError("Robot state timestamps must be finite")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Robot state timestamps must be strictly increasing")

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray([item.timestamp_sec for item in self.messages], dtype=np.float64)


class RobotStateAdapter(Protocol):
    def load(self) -> RobotStateSequence:
        ...
