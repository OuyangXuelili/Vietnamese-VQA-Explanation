from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import write_json, write_jsonl
from vqa_explanation.data.vivqax import RAW_FILES, image_manifest, load_raw_split, normalize_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize ViVQA-X and build COCO image manifests")
    parser.add_argument("--annotation-dir", default="data/raw/vivqax/annotations")
    parser.add_argument("--output-dir", default="data/processed/vivqax")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"Output already exists; choose a new dataset version: {output_dir}")
    manifest_dir = output_dir / "manifests"

    normalized: list[dict] = []
    split_counts: dict[str, int] = {}
    for split in RAW_FILES:
        records = [normalize_record(row, split) for row in load_raw_split(annotation_dir, split)]
        split_counts[split] = len(records)
        normalized.extend(records)
    if not normalized:
        raise ValueError("ViVQA-X annotations are empty")
    audit = audit_records(normalized, verify_images=False)
    if audit["errors"] or audit["counts"].get("duplicate_question_ids", 0):
        raise ValueError("ViVQA-X annotations failed the record/ID audit")
    manifest = image_manifest(normalized)

    manifest_dir.mkdir(parents=True)
    offset = 0
    for split in RAW_FILES:
        records = normalized[offset : offset + split_counts[split]]
        offset += split_counts[split]
        write_jsonl(output_dir / f"{split}.jsonl", records)
    write_jsonl(output_dir / "all.jsonl", normalized)

    with (manifest_dir / "images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    names_by_coco_split: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        names_by_coco_split[row["coco_split"]].append(row["image_name"])
    for coco_split, names in sorted(names_by_coco_split.items()):
        (manifest_dir / f"required_{coco_split}.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )

    overlap_counts = Counter(row["dataset_splits"] for row in manifest)
    stats = {
        "qa_pairs_by_split": split_counts,
        "qa_pairs_total": len(normalized),
        "unique_images_total": len(manifest),
        "unique_images_by_coco_split": {
            key: len(value) for key, value in sorted(names_by_coco_split.items())
        },
        "image_dataset_split_combinations": dict(sorted(overlap_counts.items())),
        "audit": audit,
        "annotation_sha256": {
            filename: sha256(annotation_dir / filename) for filename in RAW_FILES.values()
        },
    }
    write_json(output_dir / "stats.json", stats)
    print(f"Prepared {len(normalized):,} QA pairs and {len(manifest):,} unique images")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
