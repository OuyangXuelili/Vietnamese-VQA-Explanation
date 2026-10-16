import csv
import json

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
    Image.new("RGB", (4, 4)).save(images / names[1])
    second = verify_images(manifest, images, checksums, report_path)
    assert second["status"] == "complete"
    assert second["verified_images"] == 2
    assert len(list(csv.DictReader(checksums.open(encoding="utf-8")))) == 2
    assert json.loads(report_path.read_text(encoding="utf-8")) == second
