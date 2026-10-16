from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from vqa_explanation.data.coco import download_images, select_smoke_images
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a deterministic COCO smoke subset")
    parser.add_argument("--manifest", default="data/processed/vivqax/manifests/images.csv")
    parser.add_argument("--all-jsonl", default="data/processed/vivqax/all.jsonl")
    parser.add_argument("--output-dir", default="data/raw/coco/images")
    parser.add_argument(
        "--sample-manifest", default="data/processed/vivqax/manifests/smoke_500.csv"
    )
    parser.add_argument("--sample-jsonl", default="data/processed/vivqax/smoke_500.jsonl")
    parser.add_argument("--report", default="data/processed/vivqax/smoke_500_download.json")
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--validation", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    quotas = {"train": args.train, "validation": args.validation, "test": args.test}
    selected = select_smoke_images(read_manifest(Path(args.manifest)), quotas, args.seed)
    write_manifest(Path(args.sample_manifest), selected)

    selected_names = {row["image_name"] for row in selected}
    qa_records = [row for row in read_jsonl(args.all_jsonl) if row["image_name"] in selected_names]
    write_jsonl(args.sample_jsonl, qa_records)

    results = download_images(selected, args.output_dir, workers=args.workers)
    statuses = Counter(result["status"] for result in results)
    failures = [result for result in results if result["status"] == "failed"]
    report = {
        "seed": args.seed,
        "image_quotas": quotas,
        "selected_images": len(selected),
        "selected_qa_pairs": len(qa_records),
        "download_status": dict(sorted(statuses.items())),
        "failures": failures,
    }
    write_json(args.report, report)
    print(
        {
            "seed": report["seed"],
            "selected_images": report["selected_images"],
            "selected_qa_pairs": report["selected_qa_pairs"],
            "download_status": report["download_status"],
        }
    )
    if failures:
        raise SystemExit(f"{len(failures)} image downloads failed; see {args.report}")


if __name__ == "__main__":
    main()
