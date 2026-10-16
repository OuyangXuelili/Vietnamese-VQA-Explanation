"""Restore a verified source release without importing project dependencies."""

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ARCHIVE_ROOT = "Vietnamese-VQA-Explanation/"


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tree(repo, manifest):
    for relative, expected in manifest.items():
        path = repo / relative
        parts = Path(relative).parts
        if (not path.is_file() or any(repo.joinpath(*parts[:index]).is_symlink()
                                     for index in range(1, len(parts) + 1))
                or digest_file(path) != expected):
            raise ValueError(f"Source tree changed: {relative}")
    manifest_path = repo / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Source manifest is missing or linked")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored != manifest:
        raise ValueError("Source tree manifest changed")
    executable = {path.relative_to(repo).as_posix() for folder in ("src", "scripts", "configs")
                  for path in (repo / folder).rglob("*")
                  if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
                  and "__pycache__" not in path.parts}
    if executable - set(manifest):
        raise ValueError("Unexpected executable files in source tree")


def restore_source(archive_path, expected_sha256, work_root):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected a lowercase source SHA-256")
    if digest_file(archive_path) != expected_sha256:
        raise ValueError("Source ZIP checksum mismatch")
    parent = work_root / expected_sha256[:16]
    repo = parent / ARCHIVE_ROOT.rstrip("/")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(ARCHIVE_ROOT + "SOURCE_MANIFEST.json"))
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Empty or invalid source manifest")
        for relative, expected in manifest.items():
            if (not isinstance(relative, str) or not relative or relative.startswith("/")
                    or "\\" in relative or ":" in relative
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or not isinstance(expected, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("Unsafe source manifest entry")
        expected_names = {ARCHIVE_ROOT + name for name in manifest}
        expected_names.add(ARCHIVE_ROOT + "SOURCE_MANIFEST.json")
        if (len(archive.namelist()) != len(expected_names)
                or set(archive.namelist()) != expected_names):
            raise ValueError("Source ZIP members differ from manifest")
        if archive.testzip() is not None:
            raise ValueError("Source ZIP CRC failure")
        for member in archive.infolist():
            if member.is_dir() or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError("Source ZIP contains a linked or directory member")
        for relative, expected in manifest.items():
            if hashlib.sha256(archive.read(ARCHIVE_ROOT + relative)).hexdigest() != expected:
                raise ValueError(f"Source ZIP file changed: {relative}")
        if parent.exists():
            if parent.is_symlink() or repo.is_symlink():
                raise ValueError("Source root must not be linked")
            verify_tree(repo, manifest)
        else:
            work_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="source-", dir=work_root) as temporary:
                archive.extractall(temporary)
                verify_tree(Path(temporary) / ARCHIVE_ROOT.rstrip("/"), manifest)
                Path(temporary).rename(parent)
    return repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    print(restore_source(args.archive, args.sha256, args.work_root))


if __name__ == "__main__":
    main()
