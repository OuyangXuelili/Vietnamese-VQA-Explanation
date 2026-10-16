import hashlib
import json
import stat
import zipfile

import pytest

from vqa_explanation.training.preference_launch import restore_preference_bundle
from vqa_explanation.utils.run_provenance import sha256_file


def bundle(tmp_path, fault=None):
    files = {f"frozen/{name}": b"{}\n" for name in (
        "accepted.jsonl", "chosen_only.jsonl", "manifest.json", "quality_review.jsonl",
        "decision_record.json", "readiness.json", "text_only.json",
    )}
    files["training_lock.json"] = b"{}\n"
    if fault in {"traversal", "extra_root"}:
        files["evidence/../escaped.json" if fault == "traversal" else "run.py"] = b"{}"
    if fault == "missing":
        files.pop("training_lock.json")
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if fault == "member_hash":
        manifest["training_lock.json"] = "0" * 64
    path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(path, "x") as archive:
        for name, data in files.items():
            if fault == "symlink" and name == "training_lock.json":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, data)
            else:
                archive.writestr(name, data)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))
        if fault == "unlisted":
            archive.writestr("extra.json", "{}")
    return path, sha256_file(path)


def test_restore_is_idempotent_and_does_not_replace_changed_data(tmp_path):
    archive, digest = bundle(tmp_path)
    target = tmp_path / "restored"
    assert restore_preference_bundle(archive, digest, target) == target
    assert restore_preference_bundle(archive, digest, target) == target
    (target / "training_lock.json").write_text("changed")
    with pytest.raises(ValueError, match="Existing preference data differs"):
        restore_preference_bundle(archive, digest, target)
    assert (target / "training_lock.json").read_text() == "changed"


@pytest.mark.parametrize("fault", ["traversal", "extra_root", "missing", "member_hash",
                                  "symlink", "unlisted"])
def test_invalid_archive_never_creates_destination(tmp_path, fault):
    archive, digest = bundle(tmp_path, fault)
    target = tmp_path / "restored"
    with pytest.raises(ValueError):
        restore_preference_bundle(archive, digest, target)
    assert not target.exists()
    assert not (tmp_path / "escaped.json").exists()


def test_outer_hash_mismatch_stops_before_extraction(tmp_path):
    archive, _ = bundle(tmp_path)
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_preference_bundle(archive, "0" * 64, target)
    assert not target.exists()
