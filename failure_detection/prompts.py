"""Load version-controlled model prompts from the repository prompt directory."""
from __future__ import annotations

from pathlib import Path


def load_prompt(config_root: str | Path, value: str | Path) -> str:
    """Read one UTF-8 prompt file and reject empty prompts early.

    Relative paths are resolved from the project configuration directory. This
    keeps deployment independent of the process working directory.
    """
    root = Path(config_root).expanduser().resolve()
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text
