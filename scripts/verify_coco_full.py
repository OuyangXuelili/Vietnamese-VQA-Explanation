"""Verify every required COCO JPEG and record per-file hashes for cloud transfer."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from vqa_explanation.data.coco import load_manifest, sha256_file, valid_jpeg


def verify_images(manifest: Path, image_dir: Path, checksums: Path, report_path: Path) -> dict:
    rows = load_manifest(manifest)
    required = {row["image_name"] for row in rows}
    actual = {path.name for path in image_dir.glob("*.jpg")}
    report = {
        "manifest_sha256": sha256_file(manifest),
        "required_images": len(required),
        "present_jpeg_files": len(actual),
        "extra_jpeg_files": sorted(actual - required),
        "missing_images": [],
        "corrupt_images": [],
        "verified_images": 0,
        "total_bytes": 0,
    }
    checksums.parent.mkdir(parents=True, exist_ok=True)
    temporary = checksums.with_name(checksums.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["image_name", "size_bytes", "sha256"])
        writer.writeheader()
        for row in rows:
            name = row["image_name"]
            path = image_dir / name
            if not path.is_file():
                report["missing_images"].append(name)
            elif not valid_jpeg(path):
                report["corrupt_images"].append(name)
            else:
                size = path.stat().st_size
                writer.writerow(
                    {"image_name": name, "size_bytes": size, "sha256": sha256_file(path)}
                )
                report["verified_images"] += 1
                report["total_bytes"] += size
    os.replace(temporary, checksums)
    report["checksums_sha256"] = sha256_file(checksums)
    report["status"] = (
        "complete"
        if report["verified_images"] == len(required)
        and not report["extra_jpeg_files"]
        and not report["missing_images"]
        and not report["corrupt_images"]
        else "failed"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/vivqax/manifests/images.csv")
    )
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw/coco/images"))
    parser.add_argument(
        "--checksums", type=Path, default=Path("data/processed/vivqax/full_image_checksums.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/vivqax/full_image_verify.json")
    )
    args = parser.parse_args()
    report = verify_images(args.manifest, args.image_dir, args.checksums, args.report)
    print(
        f"{report['status']}: {report['verified_images']}/{report['required_images']} images; "
        f"missing={len(report['missing_images'])} corrupt={len(report['corrupt_images'])} "
        f"extra={len(report['extra_jpeg_files'])}"
    )
    if report["status"] != "complete":
        parser.exit(1, f"See {args.report} for details.\n")


if __name__ == "__main__":
    main()
