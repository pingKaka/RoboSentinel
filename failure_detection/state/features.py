from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import RobotStateMessage


@dataclass(frozen=True)
class RobotStateFeatures:
    actual_linear_speed: float
    actual_angular_speed: float
    commanded_linear_speed: float | None
    commanded_angular_speed: float | None
    position_tracking_error: float | None
    gripper_position_error: float | None
    force_norm: float | None
    torque_norm: float | None


def extract_robot_features(state: RobotStateMessage) -> RobotStateFeatures:
    actual_velocity = np.asarray(state.actual_tcp_velocity, dtype=np.float64)
    commanded = state.target_tcp_velocity
    position_error = None
    if state.target_tcp_position is not None:
        position_error = float(
            np.linalg.norm(state.target_tcp_position - state.actual_tcp_position)
        )
    gripper_error = None
    if state.gripper_command is not None and state.gripper_width is not None:
        gripper_error = abs(state.gripper_command - state.gripper_width)
    force_norm = torque_norm = None
    if state.tcp_wrench is not None:
        force_norm = float(np.linalg.norm(state.tcp_wrench[:3]))
        torque_norm = float(np.linalg.norm(state.tcp_wrench[3:]))
    commanded_linear = commanded_angular = None
    if commanded is not None:
        commanded_linear = float(np.linalg.norm(commanded[:3]))
        commanded_angular = float(np.linalg.norm(commanded[3:]))
    return RobotStateFeatures(
        actual_linear_speed=float(np.linalg.norm(actual_velocity[:3])),
        actual_angular_speed=float(np.linalg.norm(actual_velocity[3:])),
        commanded_linear_speed=commanded_linear,
        commanded_angular_speed=commanded_angular,
        position_tracking_error=position_error,
        gripper_position_error=gripper_error,
        force_norm=force_norm,
        torque_norm=torque_norm,
    )
