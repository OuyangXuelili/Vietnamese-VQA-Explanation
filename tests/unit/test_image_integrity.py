import csv

import pytest
from PIL import Image

from vqa_explanation.training.image_integrity import verify_record_images
from vqa_explanation.utils.run_provenance import sha256_file


def test_rejects_replaced_valid_jpeg_with_same_size(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "sample.jpg"
    Image.new("RGB", (16, 16), "red").save(image)
    original_size = image.stat().st_size
    checksums = tmp_path / "full_image_checksums.csv"
    with checksums.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["image_name", "size_bytes", "sha256"])
        writer.writerow([image.name, original_size, sha256_file(image)])
    records = [{"image_path": "images/sample.jpg"}]

    report = verify_record_images(records, tmp_path, checksums)
    assert report["verified_images"] == 1
    assert report["verified_bytes"] == original_size

    Image.new("RGB", (16, 16), "blue").save(image)
    with Image.open(image) as opened:
        opened.verify()
    assert image.stat().st_size == original_size
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_record_images(records, tmp_path, checksums)


def test_rejects_image_missing_from_manifest(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "sample.jpg"
    Image.new("RGB", (4, 4), "red").save(image)
    checksums = tmp_path / "full_image_checksums.csv"
    checksums.write_text("image_name,size_bytes,sha256\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from checksum manifest"):
        verify_record_images([{"image_path": "images/sample.jpg"}], tmp_path, checksums)
