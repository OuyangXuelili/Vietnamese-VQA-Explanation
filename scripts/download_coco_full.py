"""Download exactly the COCO images referenced by the normalized ViVQA-X manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from vqa_explanation.data.coco import download_images, load_manifest, sha256_file


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def download_manifest(
    manifest: Path,
    output_dir: Path,
    report_path: Path,
    *,
    workers: int,
    batch_size: int,
) -> dict:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    rows = load_manifest(manifest)
    report = {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "output_dir": str(output_dir.resolve()),
        "total_images": len(rows),
        "processed_images": 0,
        "download_status": {},
        "failures": [],
        "status": "running",
    }
    write_report(report_path, report)
    counts: Counter[str] = Counter()
    for start in range(0, len(rows), batch_size):
        results = download_images(rows[start : start + batch_size], output_dir, workers=workers)
        counts.update(result["status"] for result in results)
        report["failures"].extend(result for result in results if result["status"] == "failed")
        report["processed_images"] += len(results)
        report["download_status"] = dict(sorted(counts.items()))
        write_report(report_path, report)
        print(
            f"{report['processed_images']}/{len(rows)} images; "
            f"downloaded={counts['downloaded']} skipped={counts['skipped']} "
            f"failed={counts['failed']}",
            flush=True,
        )
    report["status"] = "complete" if not report["failures"] else "failed"
    write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/vivqax/manifests/images.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/coco/images"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/vivqax/full_download.json")
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = download_manifest(
        args.manifest,
        args.output_dir,
        args.report,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    if report["status"] != "complete":
        parser.exit(1, f"{len(report['failures'])} downloads failed; rerun to retry.\n")


if __name__ == "__main__":
    main()
