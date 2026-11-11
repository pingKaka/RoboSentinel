"""Canonical contract between robot-specific readers and RoboSentinel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RobotStateSample:
    """One non-controlling robot-state sample.

    ``values`` uses the canonical wire names consumed by the standard feature
    extractor. Provider-specific details belong in ``diagnostics``.
    """

    timestamp_sec: float
    source: str
    valid: bool
    values: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp_sec": float(self.timestamp_sec),
            "valid": bool(self.valid),
            **dict(self.values),
            "diagnostics": dict(self.diagnostics),
        }


class RobotStateProvider(Protocol):
    """Read-only provider implemented once for each robot integration."""

    name: str

    def read(self, observation: Mapping[str, Any], timestamp_sec: float) -> RobotStateSample:
        """Return a sample without sending commands or raising device errors."""
