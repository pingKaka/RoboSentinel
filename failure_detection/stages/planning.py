from __future__ import annotations

from collections.abc import Callable

from ..pipeline.contracts import SceneSnapshot, Task, TrackingPlan, TrackingTarget


class VLMScenePlanner:
    name = "vlm_scene_planner"

    def __init__(self, infer: Callable[[dict], dict]) -> None:
        self.infer = infer

    def plan(self, task: Task, scene: SceneSnapshot) -> TrackingPlan:
        result = self.infer({
            "task": task.description,
            "frames": {name: frame.image for name, frame in scene.frames.items()},
        })
        values = result.get("targets", [])
        if not isinstance(values, list) or not values:
            return TrackingPlan("invalid", (), self.name)
        try:
            targets = tuple(TrackingTarget(
                str(item["target_id"]), str(item["name"]), str(item["role"]),
                dict(item.get("attributes", {})),
            ) for item in values)
        except (KeyError, TypeError, ValueError):
            return TrackingPlan("invalid", (), self.name)
        return TrackingPlan("complete", targets, self.name)
