"""Provider selection kept outside the transport and feature pipeline."""
from __future__ import annotations

from .null import NullRobotStateProvider
from .ur5e import UR5eRTDEStateProvider


def create_robot_state_provider(settings=None, robot=None):
    settings = dict(settings or {})
    provider = str(settings.get("provider", "ur5e_rtde")).lower()
    if not settings.get("enabled", True) or provider in {"none", "null"}:
        return NullRobotStateProvider()
    if provider == "ur5e_rtde":
        if robot is None:
            raise ValueError("ur5e_rtde robot-state provider requires a robot instance")
        return UR5eRTDEStateProvider.from_robot(robot)
    raise ValueError(f"Unknown robot-state provider: {provider}")
