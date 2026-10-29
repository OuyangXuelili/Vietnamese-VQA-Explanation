import csv
import json

import pytest
from PIL import Image

from scripts.verify_coco_full import verify_images


def test_full_verifier_records_hashes_and_missing(tmp_path):
    manifest = tmp_path / "images.csv"
    names = [f"COCO_val2014_{number:012d}.jpg" for number in (1, 2)]
    with manifest.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["image_name", "coco_split"])
        writer.writeheader()
        writer.writerows({"image_name": name, "coco_split": "val2014"} for name in names)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / names[0])
    checksums = tmp_path / "checksums.csv"
    report_path = tmp_path / "verify.json"
    first = verify_images(manifest, images, checksums, report_path)
    assert first["status"] == "failed"
    assert first["missing_images"] == [names[1]]
    assert not checksums.exists()
    assert first["checksum_manifest_status"] == "not_published"
    Image.new("RGB", (4, 4)).save(images / names[1])
    second = verify_images(manifest, images, checksums, report_path)
    assert second["status"] == "complete"
    assert second["verified_images"] == 2
    assert len(list(csv.DictReader(checksums.open(encoding="utf-8")))) == 2
    assert json.loads(report_path.read_text(encoding="utf-8")) == second


@pytest.mark.parametrize("fault", ["missing", "corrupt", "truncated", "extra", "changed"])
def test_failed_verification_preserves_the_trusted_checksums(tmp_path, fault):
    name = "COCO_val2014_000000000001.jpg"
    manifest = tmp_path / "images.csv"
    manifest.write_text(f"image_name,coco_split\n{name},val2014\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    image = images / name
    Image.new("RGB", (4, 4), "red").save(image)
    checksums, report_path = tmp_path / "checksums.csv", tmp_path / "verify.json"
    assert verify_images(manifest, images, checksums, report_path)["status"] == "complete"
    original = checksums.read_bytes()
    if fault == "missing":
        image.unlink()
    elif fault == "corrupt":
        image.write_bytes(b"broken JPEG")
    elif fault == "truncated":
        image.write_bytes(image.read_bytes()[:-10])
    elif fault == "extra":
        Image.new("RGB", (4, 4)).save(images / "extra.jpg")
    else:
        Image.new("RGB", (4, 4), "blue").save(image)
    result = verify_images(manifest, images, checksums, report_path)
    assert result["status"] == "failed"
    assert checksums.read_bytes() == original
    assert result["checksum_manifest_status"] == (
        "mismatch" if fault == "changed" else "not_published"
    )
    assert list(tmp_path.glob("coco-verify-*")) == []


def test_complete_recheck_keeps_the_same_manifest(tmp_path):
    name = "COCO_val2014_000000000001.jpg"
    manifest = tmp_path / "images.csv"
    manifest.write_text(f"image_name,coco_split\n{name},val2014\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / name)
    checksums, report_path = tmp_path / "checksums.csv", tmp_path / "verify.json"
    verify_images(manifest, images, checksums, report_path)
    before = checksums.stat().st_mtime_ns
    result = verify_images(manifest, images, checksums, report_path)
    assert result["status"] == "complete"
    assert result["checksum_manifest_status"] == "verified"
    assert checksums.stat().st_mtime_ns == before


@pytest.mark.parametrize("fault", ["manifest", "checksums", "image"])
def test_output_paths_cannot_overwrite_inputs(tmp_path, fault):
    manifest = tmp_path / "images.csv"
    manifest.write_bytes(b"keep input")
    images = tmp_path / "images"
    images.mkdir()
    image = images / "input.jpg"
    image.write_bytes(b"keep image")
    checksums = tmp_path / "checksums.csv"
    report = {"manifest": manifest, "checksums": checksums, "image": image}[fault]
    with pytest.raises(ValueError, match="overlap"):
        verify_images(manifest, images, checksums, report)
    assert manifest.read_bytes() == b"keep input"
    assert image.read_bytes() == b"keep image"
    assert not checksums.exists()
