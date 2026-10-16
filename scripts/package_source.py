"""Package selected source files in a versioned ZIP with checksums."""

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import source_fingerprints

ROOT_FILES = (
    "AGENTS.md",
    "README.md",
    "TIEN_DO.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-demo.txt",
    "requirements-cloud-preference.txt",
    "requirements-cloud-semantic.txt",
    "Makefile",
    ".gitignore",
    ".gitattributes",
    "artifacts/README.md",
    ".github/workflows/ci.yml",
)
SOURCE_FOLDERS = ("src", "scripts", "configs", "tests", "docs", "notebooks", "app")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".ipynb", ".txt", ".html", ".css", ".js"}
SAMPLE_FILES = ("data/sample/vivqax_sample.jsonl", "data/sample/preference_sample.jsonl")
REQUIRED_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "scripts/run_pipeline.py",
    "src/vqa_explanation/__init__.py",
}
SENSITIVE_STEMS = {
    "credential",
    "credentials",
    "secret",
    "secrets",
    "tokens",
    "api_keys",
    "private",
}
SENSITIVE_CONFIG_PARTS = SENSITIVE_STEMS | {"token"}
ARCHIVE_ROOT = "Vietnamese-VQA-Explanation/"


def credential_like_filename(path: Path) -> bool:
    stem = path.stem.lower()
    if stem in SENSITIVE_STEMS:
        return True
    if path.suffix.lower() in {".yaml", ".yml"}:
        parts = set(re.split(r"[^a-z0-9]+", stem))
        return bool(SENSITIVE_CONFIG_PARTS.intersection(parts)) or (
            "api" in parts and bool({"key", "keys"}.intersection(parts))
        )
    return False


def source_files(root: Path) -> list[Path]:
    """Collect allowed source files; reject symlinks and credential-like filenames."""
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in SAMPLE_FILES:
        if (root / name).is_file() and (root / name).stat().st_size > 128 * 1024:
            raise ValueError(f"Sample fixture exceeds 128 KiB: {name}")
    files.extend(root / name for name in SAMPLE_FILES if (root / name).is_file())
    for folder in SOURCE_FOLDERS:
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if path.suffix == ".ipynb" and "_with_outputs_" in path.stem:
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info")
                   for part in relative.parts):
                continue
            files.append(path)
    missing = REQUIRED_FILES - {path.relative_to(root).as_posix() for path in files}
    if missing:
        raise FileNotFoundError(f"Required source files are missing: {sorted(missing)}")
    selected = {path.relative_to(root).as_posix() for path in files}
    unpackageable = set(source_fingerprints(root)) - selected
    if unpackageable:
        raise ValueError(f"Provenance source is excluded from release: {sorted(unpackageable)}")
    for path in files:
        relative = path.relative_to(root)
        if credential_like_filename(path):
            raise ValueError(f"Refusing to package credential-like file: {relative}")
        linked = any(
            root.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        )
        if linked or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing to package symlink or external file: {relative}")
    return sorted(files)


def verify_source_archive(archive: Path, manifest: dict[str, str]) -> None:
    """Verify ZIP contents, member names and checksums."""
    expected = {ARCHIVE_ROOT + name for name in manifest}
    expected.add(ARCHIVE_ROOT + "SOURCE_MANIFEST.json")
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Source archive members do not match the manifest")
        if handle.testzip() is not None:
            raise ValueError("Source archive failed CRC verification")
        stored_manifest = json.loads(handle.read(ARCHIVE_ROOT + "SOURCE_MANIFEST.json"))
        if stored_manifest != manifest:
            raise ValueError("Source archive manifest differs from the packaged files")
        for relative, digest in manifest.items():
            actual = hashlib.sha256(handle.read(ARCHIVE_ROOT + relative)).hexdigest()
            if actual != digest:
                raise ValueError(f"Source archive hash mismatch: {relative}")


def main():
    root = project_root()
    files = source_files(root)
    manifest = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    name = "source_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = root / "artifacts" / "releases" / name
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / (name + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for p in files:
            handle.write(p, ARCHIVE_ROOT + p.relative_to(root).as_posix())
        handle.writestr(
            ARCHIVE_ROOT + "SOURCE_MANIFEST.json", json.dumps(manifest, indent=2)
        )
    verify_source_archive(archive, manifest)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".sha256").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    print(archive)
    print(f"SHA256 {digest}; {len(manifest)} verified files; not uploaded")


if __name__ == "__main__":
    main()
