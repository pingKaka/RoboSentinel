"""Read-only UR5e state adapter using an existing RTDE receive object."""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import RobotStateSample


class UR5eRTDEStateProvider:
    """Map UR-specific observations into RoboSentinel's canonical schema.

    The provider deliberately reuses the receiver owned by the robot driver. It
    never opens a second RTDE connection and never calls a control method.
    """

    name = "ur5e_rtde"

    def __init__(self, receiver=None, fallback_frame: str = "base"):
        self._receiver = receiver
        self._fallback_frame = fallback_frame

    @classmethod
    def from_robot(cls, robot) -> "UR5eRTDEStateProvider":
        # Driver-private access is intentionally isolated in this one adapter.
        receiver = getattr(robot, "_arm", {}).get("rtde_r")
        config = getattr(robot, "config", None)
        control_space = getattr(config, "control_space", "")
        frame = (getattr(config, control_space + "_reference_frame", "base")
                 if isinstance(control_space, str) else "base")
        if not isinstance(frame, str):
            frame = "base"
        return cls(receiver, "episode_reference" if frame == "tcp" else str(frame))

    def read(self, observation: Mapping[str, Any], timestamp_sec: float) -> RobotStateSample:
        values: dict[str, Any] = {
            "tcp_position": None,
            "tcp_position_frame": None,
            "gripper_position_normalized": self._gripper(observation),
        }
        diagnostics: dict[str, Any] = {"rtde_receiver_available": self._receiver is not None}

        if self._receiver is not None:
            try:
                pose = self._receiver.getActualTCPPose()
                if len(pose) < 3:
                    raise ValueError("actual TCP pose has fewer than three values")
                values["tcp_position"] = [float(value) for value in pose[:3]]
                values["tcp_position_frame"] = "base"
            except Exception as exc:  # Device faults must not stop teleoperation.
                diagnostics["tcp_read_error"] = f"{type(exc).__name__}: {exc}"

        if values["tcp_position"] is None:
            position = [observation.get(f"tcp_pose.{axis}") for axis in ("x", "y", "z")]
            if all(value is not None for value in position):
                try:
                    values["tcp_position"] = [float(value) for value in position]
                    values["tcp_position_frame"] = self._fallback_frame
                    diagnostics["tcp_source"] = "observation_fallback"
                except (TypeError, ValueError) as exc:
                    diagnostics["tcp_fallback_error"] = f"{type(exc).__name__}: {exc}"

        valid = values["tcp_position"] is not None or values["gripper_position_normalized"] is not None
        return RobotStateSample(timestamp_sec, self.name, valid, values, diagnostics)

    @staticmethod
    def _gripper(observation: Mapping[str, Any]) -> float | None:
        value = observation.get("gripper_raw_position")
        if value is None:
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        return normalized if 0 <= normalized <= 1 else None
