from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_VERSION = 4


def load_project_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Project config must be a mapping")
    required = {"version", "task", "streams", "models", "stages", "runtime", "outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if config["version"] != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version: {config['version']!r}")
    if config["stages"]["rules"].get("plugins") is None:
        raise ValueError("stages.rules.plugins is required")
    return config, path.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path
