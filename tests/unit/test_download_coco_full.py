import csv
import json

import pytest

from scripts import download_coco_full


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["image_name", "coco_split"])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_rejects_duplicate_and_unsafe_names(tmp_path):
    path = tmp_path / "images.csv"
    valid = {"image_name": "COCO_train2014_000000000001.jpg", "coco_split": "train2014"}
    write_manifest(path, [valid, valid])
    with pytest.raises(ValueError, match="Duplicate"):
        download_coco_full.load_manifest(path)
    write_manifest(path, [{"image_name": "../bad.jpg", "coco_split": "train2014"}])
    with pytest.raises(ValueError, match="Invalid"):
        download_coco_full.load_manifest(path)


def test_full_downloader_batches_and_writes_resume_report(tmp_path, monkeypatch):
    manifest = tmp_path / "images.csv"
    rows = [
        {"image_name": f"COCO_train2014_{number:012d}.jpg", "coco_split": "train2014"}
        for number in range(3)
    ]
    write_manifest(manifest, rows)
    calls = []

    def fake_download(batch, output_dir, workers):
        calls.append((len(batch), output_dir, workers))
        return [
            {"image_name": row["image_name"], "status": "skipped", "error": ""} for row in batch
        ]

    monkeypatch.setattr(download_coco_full, "download_images", fake_download)
    report_path = tmp_path / "report.json"
    report = download_coco_full.download_manifest(
        manifest, tmp_path / "images", report_path, workers=2, batch_size=2
    )
    assert [size for size, _, _ in calls] == [2, 1]
    assert report["status"] == "complete"
    assert report["processed_images"] == report["total_images"] == 3
    assert report["download_status"] == {"skipped": 3}
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_full_downloader_records_failures_for_retry(tmp_path, monkeypatch):
    manifest = tmp_path / "images.csv"
    row = {"image_name": "COCO_val2014_000000000001.jpg", "coco_split": "val2014"}
    write_manifest(manifest, [row])
    monkeypatch.setattr(
        download_coco_full,
        "download_images",
        lambda rows, output_dir, workers: [
            {"image_name": rows[0]["image_name"], "status": "failed", "error": "network"}
        ],
    )
    report_path = tmp_path / "report.json"
    report = download_coco_full.download_manifest(
        manifest, tmp_path / "images", report_path, workers=1, batch_size=1
    )
    assert report["status"] == "failed"
    assert report["processed_images"] == 1
    assert report["failures"][0]["error"] == "network"
