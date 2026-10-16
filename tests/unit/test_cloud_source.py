"""Restore source without overwriting previous work or trusting unsafe ZIP paths."""

import hashlib
import json
import zipfile

import pytest

from scripts.bootstrap_cloud_source import ARCHIVE_ROOT, digest_file, restore_source


def source_zip(tmp_path, relative="scripts/run.py"):
    content = b"print('source fixture')\n"
    manifest = {relative: hashlib.sha256(content).hexdigest()}
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr(ARCHIVE_ROOT + relative, content)
        archive.writestr(ARCHIVE_ROOT + "SOURCE_MANIFEST.json", json.dumps(manifest))
    return path, digest_file(path)


def test_restore_reuses_only_unchanged_source(tmp_path):
    path, digest = source_zip(tmp_path)
    repo = restore_source(path, digest, tmp_path / "work")
    assert restore_source(path, digest, tmp_path / "work") == repo
    (repo / "scripts/run.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        restore_source(path, digest, tmp_path / "work")
    assert (repo / "scripts/run.py").read_text() == "changed"


@pytest.mark.parametrize("relative", ["../outside.py", "/absolute.py", "a/../../b.py",
                                      "C:/outside.py", "scripts\\run.py"])
def test_unsafe_members_never_extract(tmp_path, relative):
    path, digest = source_zip(tmp_path, relative)
    with pytest.raises(ValueError, match="Unsafe|members differ"):
        restore_source(path, digest, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_bad_checksum_does_not_touch_work_directory(tmp_path):
    path, _ = source_zip(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        restore_source(path, "0" * 64, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_unexpected_executable_blocks_source_reuse(tmp_path):
    path, digest = source_zip(tmp_path)
    repo = restore_source(path, digest, tmp_path / "work")
    (repo / "scripts/extra.py").write_text("pass", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        restore_source(path, digest, tmp_path / "work")
