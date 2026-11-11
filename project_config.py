from __future__ import annotations

from pathlib import Path
from datetime import datetime
import uuid
from typing import Any

import yaml

from failure_detection.features.contracts import FEATURE_SCHEMA_VERSION
from failure_detection.rule_dsl import RULE_SCHEMA_VERSION


CONFIG_VERSION = 4
DEFAULT_CONFIG = Path(__file__).resolve().parent / "cfg.yaml"


def _read_config(path: Path, stack=()) -> dict:
    path = path.expanduser().resolve()
    if path in stack:
        raise ValueError(f"Circular config include: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Project config must be a mapping")
    includes = value.pop("include", [])
    if not isinstance(includes, list) or any(not isinstance(p, str) for p in includes):
        raise ValueError("include must be a list of config paths")
    merged = {}
    for name in includes:
        _merge(merged, _read_config(path.parent / name, (*stack, path)))
    _merge(merged, value)
    return merged


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def load_project_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    config = _read_config(path)
    if not isinstance(config, dict):
        raise ValueError("Project config must be a mapping")
    required = {"version", "task", "streams", "models", "stages", "runtime", "outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if config["version"] != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version: {config['version']!r}")
    rule_config = config["stages"]["rules"]
    if rule_config.get("mode") != "generated":
        raise ValueError("stages.rules.mode must be 'generated'")
    if rule_config.get("schema_version") != RULE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rule schema version: {rule_config.get('schema_version')!r}")
    configured_features = config["stages"].get("features", {}).get("schema_version")
    if configured_features != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema version: {configured_features!r}; expected {FEATURE_SCHEMA_VERSION}"
        )
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
