"""Fingerprint source and inputs for experiment provenance."""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints(root: Path) -> dict[str, str]:
    """Hash executable source/config, not datasets, notebooks, outputs or secrets."""
    files = [root / name for name in ("AGENTS.md", "Makefile", "pyproject.toml")]
    for folder, suffixes in (
        ("src", {".py", ".html", ".css", ".js"}),
        ("scripts", {".py"}),
        ("configs", {".yaml", ".yml"}),
        ("app", {".html", ".css", ".js"}),
    ):
        files.extend(path for path in (root / folder).rglob("*") if path.suffix in suffixes)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
        if path.is_file() and not path.is_symlink()
    }


def preference_training_fingerprints(root: Path) -> dict[str, str]:
    """Hash the trainer's local import graph, separately from the full release."""
    root = root.resolve()
    pending = ["vqa_explanation.training.preference"]
    visited, files = set(), set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        relative = Path("src", *module.split("."))
        path = root / relative.with_suffix(".py")
        if not path.is_file():
            path = root / relative / "__init__.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trainer dependency: {module}")
        files.add(path)
        parts = module.split(".")
        pending.extend(
            ".".join(parts[:index]) for index in range(1, len(parts))
            if (root / "src" / Path(*parts[:index]) / "__init__.py").is_file()
        )
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                dependencies = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package.split(".")[:len(package.split(".")) - node.level + 1]
                dependency = ".".join(prefix + ([node.module] if node.module else []))
                dependency = dependency if node.level else node.module or ""
                dependencies = [dependency]
                for alias in node.names:
                    submodule = dependency + "." + alias.name
                    relative = root / "src" / Path(*submodule.split("."))
                    if (relative.with_suffix(".py").is_file()
                            or (relative / "__init__.py").is_file()):
                        dependencies.append(submodule)
            else:
                continue
            pending.extend(name for name in dependencies if (
                name == "vqa_explanation" or name.startswith("vqa_explanation.")
            ))
    files.update(root / name for name in (
        "scripts/train_preference.py", "requirements-cloud-preference.txt",
    ))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)}


def git_state(root: Path) -> dict[str, str | bool | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"commit": None, "dirty": None}
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def collect_provenance(
    root: Path,
    environment: Path,
    annotations: Path,
    training: Path | None = None,
    model: Path | None = None,
    image_checksums: Path | None = None,
) -> dict:
    fingerprints = source_fingerprints(root)
    source_sha256 = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    files = {"environment": environment, "annotations": annotations}
    if training is not None:
        files["training_config"] = training
    if model is not None:
        files["model_config"] = model
    if image_checksums is not None:
        files["image_checksums"] = image_checksums
    return {
        "source_sha256": source_sha256,
        "source_files": fingerprints,
        "input_sha256": {name: sha256_file(path) for name, path in files.items()},
        "git": git_state(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
