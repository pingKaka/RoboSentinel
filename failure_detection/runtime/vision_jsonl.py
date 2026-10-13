from __future__ import annotations

import json
from pathlib import Path

from .schemas import VisionObservation


class VisionJSONLAdapter:
    """Replay timestamped outputs produced by an external GPU vision service."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.observations = self._load(path)

    @staticmethod
    def _load(path: Path) -> tuple[VisionObservation, ...]:
        if not path.is_file():
            raise FileNotFoundError(path)
        values = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "timestamp_sec" not in value:
                raise ValueError(f"Invalid vision observation at {path}:{line_number}")
            values.append(VisionObservation(**value))
        if not values:
            raise ValueError(f"No vision observations in {path}")
        if any(
            current.timestamp_sec <= previous.timestamp_sec
            for previous, current in zip(values, values[1:])
        ):
            raise ValueError("Vision observation timestamps must be strictly increasing")
        return tuple(values)

    def nearest(self, timestamp_sec: float, max_delta_sec: float) -> VisionObservation | None:
        value = min(self.observations, key=lambda item: abs(item.timestamp_sec - timestamp_sec))
        return value if abs(value.timestamp_sec - timestamp_sec) <= max_delta_sec else None
