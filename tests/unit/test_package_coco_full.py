import csv
import hashlib
import json
import zipfile

import pytest
from PIL import Image

from scripts.package_coco_full import package_dataset, validate_annotations
from scripts.verify_coco_bundle import verify_bundle


def test_package_dataset_uses_split_archives_and_hashes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "images.csv"
    checksums = tmp_path / "full_image_checksums.csv"
    rows = []
    checksum_rows = []
    for split, number in (("train2014", 1), ("val2014", 2), ("val2014", 3)):
        name = f"COCO_{split}_{number:012d}.jpg"
        image = images / name
        Image.new("RGB", (4, 4)).save(image)
        rows.append({"image_name": name, "coco_split": split})
        checksum_rows.append(
            {
                "image_name": name,
                "size_bytes": image.stat().st_size,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    for path, fieldnames, content in (
        (manifest, ["image_name", "coco_split"], rows),
        (checksums, ["image_name", "size_bytes", "sha256"], checksum_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(content)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    records = []
    for index, row in enumerate(rows):
        name = row["image_name"]
        records.append(
            {
                "question_id": f"q{index}",
                "image_id": str(index + 1),
                "image_name": name,
                "image_path": f"images/{name}",
                "question": "Con gì?",
                "answer": "Mèo",
                "explanations": ["Có mèo."],
                "split": ("train", "validation", "test")[index],
            }
        )
    (annotations / "all.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    for split in ("train", "validation", "test"):
        (annotations / f"{split}.jsonl").write_text(
            json.dumps(next(record for record in records if record["split"] == split)) + "\n",
            encoding="utf-8",
        )
    (annotations / "provenance.json").write_text(
        json.dumps({"protocol": "image_disjoint_v1"}), encoding="utf-8"
    )
    output = tmp_path / "bundle"
    report = package_dataset(manifest, checksums, images, annotations, output)
    assert report["status"] == "complete"
    assert report["total_images"] == 3
    assert len(report["archives"]) == 3
    with zipfile.ZipFile(output / "coco_train2014.zip") as archive:
        assert archive.namelist() == [f"images/{rows[0]['image_name']}"]
    assert json.loads((output / "bundle.json").read_text(encoding="utf-8")) == report
    assert verify_bundle(output)["images"] == 3
    extracted = tmp_path / "extracted"
    assert verify_bundle(output, extracted)["status"] == "verified"
    assert (extracted / "images" / rows[0]["image_name"]).is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        verify_bundle(output, extracted)
    with (output / "coco_val2014.zip").open("ab") as archive:
        archive.write(b"tampered")
    with pytest.raises(ValueError, match="size-mismatched"):
        verify_bundle(output)


def test_validate_annotations_rejects_split_file_mismatch(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    image_name = "COCO_train2014_000000000001.jpg"
    record = {
        "question_id": "q1",
        "image_id": "1",
        "image_name": image_name,
        "image_path": f"images/{image_name}",
        "question": "Con gì?",
        "answer": "Mèo",
        "explanations": ["Có mèo."],
        "split": "train",
    }
    (annotations / "all.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (annotations / "train.jsonl").write_text("{}\n", encoding="utf-8")
    for split in ("validation", "test"):
        (annotations / f"{split}.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="train.jsonl differs"):
        validate_annotations(annotations, {image_name})


def test_validate_annotations_rejects_noncanonical_split_and_path_alias(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    image_name = "COCO_train2014_000000000001.jpg"
    row = {
        "question_id": "q1",
        "image_id": "1",
        "image_name": image_name,
        "image_path": f"images/{image_name}",
        "question": "Con gì?",
        "answer": "Mèo",
        "explanations": ["Có mèo."],
        "split": "val",
    }
    (annotations / "all.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Annotation split must"):
        validate_annotations(annotations, {image_name})
    row["split"] = "train"
    row["image_path"] = "images/../images/" + image_name
    (annotations / "all.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image name/path"):
        validate_annotations(annotations, {image_name})
