from __future__ import annotations

import bisect

import numpy as np

from ..state.base import RobotStateMessage, RobotStateSequence
from ..state.features import extract_robot_features
from .schemas import AlignedObservation, VisionObservation
from .buffer import TimeSeriesBuffer


def _lerp(left: np.ndarray, right: np.ndarray, ratio: float) -> np.ndarray:
    return left + ratio * (right - left)


def _normalized_lerp(left: np.ndarray, right: np.ndarray, ratio: float) -> np.ndarray:
    # Sufficient for high-rate adjacent samples; flip to the shortest quaternion arc.
    if float(np.dot(left, right)) < 0:
        right = -right
    value = _lerp(left, right, ratio)
    return value / max(1e-12, float(np.linalg.norm(value)))


class StateSynchronizer:
    def __init__(self, sequence: RobotStateSequence, max_delta_sec: float = 0.05) -> None:
        if max_delta_sec <= 0:
            raise ValueError("max_delta_sec must be positive")
        self.sequence = sequence
        self.max_delta_sec = max_delta_sec
        self.timestamps = sequence.timestamps.tolist()

    def align(
        self, timestamp_sec: float, vision: VisionObservation | None = None
    ) -> AlignedObservation:
        state, delta = self._interpolate(timestamp_sec)
        valid = delta <= self.max_delta_sec
        return AlignedObservation(
            timestamp_sec=timestamp_sec,
            robot_state=state if valid else None,
            robot_features=extract_robot_features(state) if valid and state else None,
            vision=vision,
            synchronization_error_sec={"robot_state": delta},
            valid={"robot_state": valid, "vision": vision is not None},
        )

    def _interpolate(self, timestamp_sec: float) -> tuple[RobotStateMessage, float]:
        messages = self.sequence.messages
        position = bisect.bisect_left(self.timestamps, timestamp_sec)
        if position == 0:
            state = messages[0]
            return state, abs(state.timestamp_sec - timestamp_sec)
        if position == len(messages):
            state = messages[-1]
            return state, abs(state.timestamp_sec - timestamp_sec)
        left, right = messages[position - 1], messages[position]
        span = right.timestamp_sec - left.timestamp_sec
        ratio = (timestamp_sec - left.timestamp_sec) / span
        nearest = left if ratio <= 0.5 else right

        def optional_vector(
            left_value: np.ndarray | None, right_value: np.ndarray | None
        ) -> np.ndarray | None:
            if left_value is None or right_value is None:
                return left_value if ratio <= 0.5 else right_value
            return _lerp(left_value, right_value, ratio)

        def optional_float(left_value: float | None, right_value: float | None) -> float | None:
            if left_value is None or right_value is None:
                return left_value if ratio <= 0.5 else right_value
            return left_value + ratio * (right_value - left_value)

        state = RobotStateMessage(
            timestamp_sec=timestamp_sec,
            sequence_id=nearest.sequence_id,
            actual_tcp_position=_lerp(left.actual_tcp_position, right.actual_tcp_position, ratio),
            actual_tcp_quaternion=_normalized_lerp(
                left.actual_tcp_quaternion, right.actual_tcp_quaternion, ratio
            ),
            actual_tcp_velocity=_lerp(left.actual_tcp_velocity, right.actual_tcp_velocity, ratio),
            target_tcp_position=optional_vector(left.target_tcp_position, right.target_tcp_position),
            target_tcp_velocity=optional_vector(left.target_tcp_velocity, right.target_tcp_velocity),
            joint_position=optional_vector(left.joint_position, right.joint_position),
            joint_velocity=optional_vector(left.joint_velocity, right.joint_velocity),
            gripper_width=optional_float(left.gripper_width, right.gripper_width),
            gripper_command=optional_float(left.gripper_command, right.gripper_command),
            gripper_force=optional_float(left.gripper_force, right.gripper_force),
            tcp_wrench=optional_vector(left.tcp_wrench, right.tcp_wrench),
            controller_state=nearest.controller_state,
            active_skill=nearest.active_skill,
        )
        delta = min(timestamp_sec - left.timestamp_sec, right.timestamp_sec - timestamp_sec)
        return state, delta


class OnlineSynchronizer:
    """Synchronize live high-rate robot states with the latest vision result."""

    def __init__(self, buffer_seconds: float = 15.0, max_delta_sec: float = 0.05) -> None:
        self.max_delta_sec = max_delta_sec
        self.robot_states = TimeSeriesBuffer[RobotStateMessage](
            buffer_seconds, lambda item: item.timestamp_sec
        )
        self.vision = TimeSeriesBuffer[VisionObservation](
            buffer_seconds, lambda item: item.timestamp_sec
        )

    def push_robot_state(self, state: RobotStateMessage) -> None:
        self.robot_states.append(state)

    def push_vision(self, vision: VisionObservation) -> None:
        self.vision.append(vision)

    def snapshot(self, timestamp_sec: float) -> AlignedObservation:
        state, state_delta = self.robot_states.nearest(timestamp_sec)
        vision, vision_delta = self.vision.nearest(timestamp_sec)
        state_valid = state is not None and state_delta <= self.max_delta_sec
        vision_valid = vision is not None and vision_delta <= self.max_delta_sec
        return AlignedObservation(
            timestamp_sec=timestamp_sec,
            robot_state=state if state_valid else None,
            robot_features=extract_robot_features(state) if state_valid and state else None,
            vision=vision if vision_valid else None,
            synchronization_error_sec={
                "robot_state": state_delta,
                "vision": vision_delta,
            },
            valid={"robot_state": state_valid, "vision": vision_valid},
        )
