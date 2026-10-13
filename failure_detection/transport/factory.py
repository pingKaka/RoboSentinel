from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..pipeline import StreamingRoboSentinel


PipelineFactory = Callable[[dict[str, Any]], StreamingRoboSentinel]


def load_pipeline_factory(value: str | None) -> PipelineFactory:
    if not value or ":" not in value:
        raise ValueError(
            "deployment.pipeline_factory must be set to 'package.module:function'"
        )
    module_name, function_name = value.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"Pipeline factory is not callable: {value}")
    return factory
