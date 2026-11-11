"""No-hardware provider used by video-only and disabled configurations."""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import RobotStateSample


class NullRobotStateProvider:
    name = "none"

    def read(self, observation: Mapping[str, Any], timestamp_sec: float) -> RobotStateSample:
        return RobotStateSample(timestamp_sec, self.name, False, diagnostics={"reason": "disabled"})
