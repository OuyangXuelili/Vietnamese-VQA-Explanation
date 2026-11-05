import csv
import json
import zipfile

import pytest
import yaml

from vqa_explanation.data.image_subset import package_image_subset, restore_image_subset
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def subset(tmp_path):
    metadata = tmp_path / "metadata"
    images = tmp_path / "coco/images"
    preferences = tmp_path / "preferences"
    for p in (metadata, images, preferences):
        p.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "validation", "test"), 1):
        name = f"COCO_{'train' if split == 'train' else 'val'}2014_{index:012d}.jpg"
        (images / name).write_bytes(b"offline file fixture " + bytes([index]))
        rows.append({"question_id": str(index), "split": split, "image_name": name,
                     "image_path": "images/" + name})
    all_path = metadata / "all.jsonl"
    all_path.write_text(''.join(json.dumps(r) + '\n' for r in rows), "utf-8")
    for split in ("train", "validation", "test"):
        (metadata / (split + ".jsonl")).write_text(
            ''.join(json.dumps(r) + '\n' for r in rows if r["split"] == split), "utf-8")
    checksums = metadata / "full_image_checksums.csv"
    with checksums.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_name", "sha256"))
        writer.writeheader()
        writer.writerows({"image_name": r["image_name"],
                          "sha256": sha256_file(images / r["image_name"])} for r in rows)
    accepted = preferences / "accepted.jsonl"
    accepted.write_text(json.dumps({"source_question_id": "1"}) + '\n', "utf-8")
    (preferences / "manifest.json").write_text(json.dumps({
        "files": {"accepted.jsonl": sha256_file(accepted)}}), "utf-8")
    config = tmp_path / "regression.yaml"
    config.write_text(yaml.safe_dump({"cases": [{"question_id": "2"}]}), "utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({"annotations": str(all_path),
        "image_root": str(images.parent), "run_root": str(tmp_path / "runs"),
        "image_checksums": str(checksums)}), "utf-8")
    output = tmp_path / "subset.zip"
    result = package_image_subset(environment, preferences, config, output)
    return tmp_path, metadata, rows, output, result


def test_subset_roundtrip_keeps_canonical_metadata_and_only_required_images(subset):
    folder, metadata, rows, archive, result = subset
    destination = folder / "restored"
    report = restore_image_subset(archive, result["sha256"], destination)
    assert report["images"] == 2
    assert ((destination / "metadata/all.jsonl").read_bytes()
            == (metadata / "all.jsonl").read_bytes())
    assert len(list((destination / "images").glob('*.jpg'))) == 2
    assert not (destination / rows[2]["image_path"]).exists()
    assert restore_image_subset(archive, result["sha256"], destination) == report


def test_subset_refuses_to_overwrite_changed_existing_data(subset):
    folder, _, rows, archive, result = subset
    destination = folder / "restored"
    restore_image_subset(archive, result["sha256"], destination)
    image = destination / rows[0]["image_path"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing data differs"):
        restore_image_subset(archive, result["sha256"], destination)
    assert image.read_bytes() == b"changed"


def test_subset_verifies_archive_before_creating_destination(subset):
    folder, _, _, archive, _ = subset
    destination = folder / "restored"
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_image_subset(archive, '0' * 64, destination)
    assert not destination.exists()


def test_subset_rejects_undeclared_path_before_extraction(subset):
    folder, _, _, archive, _ = subset
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr("../outside.txt", "outside")
    destination = folder / "restored"
    with pytest.raises(ValueError, match="manifest or member paths"):
        restore_image_subset(archive, sha256_file(archive), destination)
    assert not destination.exists() and not (folder / "outside.txt").exists()


def test_subset_rejects_corrupt_member_even_with_matching_archive_checksum(subset):
    folder, _, _, archive, _ = subset
    with zipfile.ZipFile(archive) as original:
        files = {name: original.read(name) for name in original.namelist()}
    files["metadata/train.jsonl"] += b"changed"
    changed = folder / "changed.zip"
    with zipfile.ZipFile(changed, "w") as stream:
        for name, payload in files.items():
            stream.writestr(name, payload)
    with pytest.raises(ValueError, match="member checksum"):
        restore_image_subset(changed, sha256_file(changed), folder / "restored")
