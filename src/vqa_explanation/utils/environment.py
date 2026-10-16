"""Resolve machine-specific storage without importing GPU or Google libraries."""

import os
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_environment(path: Path, root: Path | None = None) -> dict[str, Path]:
    root = (root or project_root()).resolve()
    path = path if path.is_absolute() else root / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"annotations", "image_root", "run_root"}
    optional = {"image_checksums"}
    if not isinstance(raw, dict) or not required.issubset(raw) or set(raw) - required - optional:
        raise ValueError(
            f"Environment must contain {sorted(required)} and only {sorted(optional)} optionally"
        )
    result = {}
    for name, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid path for {name}")
        expanded = os.path.expandvars(value)
        if "$" in expanded or "%" in expanded:
            raise ValueError(f"Unresolved environment variable in {name}: {value}")
        resolved = Path(expanded).expanduser()
        result[name] = (resolved if resolved.is_absolute() else root / resolved).resolve()
    return result
