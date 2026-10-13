from __future__ import annotations

from typing import Protocol

from ..pipeline.contracts import PerceptionUpdate, SceneSnapshot, StreamObservation, TrackingPlan


class GroundedSAMBackend(Protocol):
    """Backend should use GroundingDINO for seeds and SAM memory for frame updates."""

    def initialize(self, plan: TrackingPlan, scene: SceneSnapshot) -> None: ...
    def update(self, observation: StreamObservation) -> PerceptionUpdate: ...


class OnlineGroundingSAMTracker:
    name = "grounding_dino_sam_online"

    def __init__(self, backend: GroundedSAMBackend) -> None:
        self.backend = backend

    def initialize(self, plan: TrackingPlan, scene: SceneSnapshot) -> None:
        self.backend.initialize(plan, scene)

    def update(self, observation: StreamObservation) -> PerceptionUpdate:
        result = self.backend.update(observation)
        if not isinstance(result, PerceptionUpdate):
            raise TypeError("Tracking backend must return PerceptionUpdate")
        return result
