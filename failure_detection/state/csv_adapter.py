from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .base import RobotStateMessage, RobotStateSequence


POSITION_COLUMNS = ("ee_x", "ee_y", "ee_z")
QUATERNION_COLUMNS = ("ee_qx", "ee_qy", "ee_qz", "ee_qw")
VELOCITY_COLUMNS = ("ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz")
TARGET_POSITION_COLUMNS = ("target_ee_x", "target_ee_y", "target_ee_z")
TARGET_VELOCITY_COLUMNS = (
    "target_ee_vx", "target_ee_vy", "target_ee_vz",
    "target_ee_wx", "target_ee_wy", "target_ee_wz",
)
WRENCH_COLUMNS = ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")


class CSVStateAdapter:
    """Load a portable robot-state CSV into the runtime contract."""

    def __init__(self, path: Path, timestamp_column: str = "timestamp_sec") -> None:
        self.path = path
        self.timestamp_column = timestamp_column

    def load(self) -> RobotStateSequence:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {self.timestamp_column, *POSITION_COLUMNS, *QUATERNION_COLUMNS}
            missing = sorted(required - fields)
            if missing:
                raise ValueError(f"Robot state CSV is missing columns: {missing}")
            messages = tuple(
                self._message(row, index, fields) for index, row in enumerate(reader)
            )
        return RobotStateSequence(messages, str(self.path))

    def _message(
        self, row: dict[str, str], index: int, fields: set[str]
    ) -> RobotStateMessage:
        quaternion = self._required_vector(row, QUATERNION_COLUMNS)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError(f"Row {index + 2}: end-effector quaternion has zero norm")
        quaternion = quaternion / norm
        velocity = self._optional_vector(row, VELOCITY_COLUMNS, fields)
        return RobotStateMessage(
            timestamp_sec=float(row[self.timestamp_column]),
            sequence_id=index,
            actual_tcp_position=self._required_vector(row, POSITION_COLUMNS),
            actual_tcp_quaternion=quaternion,
            actual_tcp_velocity=velocity if velocity is not None else np.zeros(6),
            target_tcp_position=self._optional_vector(row, TARGET_POSITION_COLUMNS, fields),
            target_tcp_velocity=self._optional_vector(row, TARGET_VELOCITY_COLUMNS, fields),
            gripper_width=self._optional_float(row, "gripper_width", fields),
            gripper_command=self._optional_float(row, "gripper_command", fields),
            gripper_force=self._optional_float(row, "gripper_force", fields),
            tcp_wrench=self._optional_vector(row, WRENCH_COLUMNS, fields),
            controller_state=self._optional_text(row, "controller_state", fields),
            active_skill=self._optional_text(row, "active_skill", fields),
        )

    @staticmethod
    def _required_vector(row: dict[str, str], names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([float(row[name]) for name in names], dtype=np.float64)

    @staticmethod
    def _optional_vector(
        row: dict[str, str], names: tuple[str, ...], fields: set[str]
    ) -> np.ndarray | None:
        if not set(names).issubset(fields) or any(not row.get(name, "").strip() for name in names):
            return None
        return np.asarray([float(row[name]) for name in names], dtype=np.float64)

    @staticmethod
    def _optional_float(row: dict[str, str], name: str, fields: set[str]) -> float | None:
        return float(row[name]) if name in fields and row.get(name, "").strip() else None

    @staticmethod
    def _optional_text(row: dict[str, str], name: str, fields: set[str]) -> str | None:
        return row[name].strip() if name in fields and row.get(name, "").strip() else None
