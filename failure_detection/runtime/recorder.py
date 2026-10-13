from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLEventRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, value: dict[str, Any]) -> None:
        self._handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JSONLEventRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
