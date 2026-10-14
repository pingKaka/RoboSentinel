from __future__ import annotations

from pathlib import Path
from datetime import datetime
import uuid
from typing import Any

import yaml


CONFIG_VERSION = 4
DEFAULT_CONFIG = Path(__file__).resolve().parent / "cfg.yaml"


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
    config["_config_root"] = str(path.parent)
    return config, path.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def client_settings(config, root, mode):
    settings = dict(config.get("client", {}))
    settings["description"] = config["task"]["description"]
    settings["task_id"] = config["task"]["id"]
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    settings["output_dir"] = str(resolve_path(root, settings.get("output_dir", "result/client")) / mode / run_name)
    return settings
