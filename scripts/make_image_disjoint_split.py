"""Create a versioned image-disjoint ViVQA-X split without changing original annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.coco import COCO_NAME_PATTERN
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.data.vivqax import image_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_image_disjoint(records: list[dict]) -> tuple[list[dict], list[str]]:
    name_by_id: dict[str, str] = {}
    id_by_name: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("split") not in ("train", "validation", "test"):
            raise ValueError("Every source record needs a canonical train/validation/test split")
        if not isinstance(row.get("image_name"), str) or not COCO_NAME_PATTERN.fullmatch(
            row["image_name"]
        ):
            raise ValueError("Every source record needs a canonical COCO image_name")
        if row.get("image_path") != f"images/{row['image_name']}":
            raise ValueError("image_path must match image_name")
        image_id, name = str(row["image_id"]), str(row["image_name"])
        if image_id in name_by_id and name_by_id[image_id] != name:
            raise ValueError(f"Image ID maps to multiple filenames: {image_id}")
        if name in id_by_name and id_by_name[name] != image_id:
            raise ValueError(f"Image filename maps to multiple IDs: {name}")
        name_by_id[image_id] = name
        id_by_name[name] = image_id
    images = {
        split: {str(row["image_id"]) for row in records if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(not image_ids for image_ids in images.values()):
        raise ValueError("All three original splits must be non-empty")
    if images["train"] & (images["validation"] | images["test"]):
        raise ValueError("Train overlaps a held-out split; review the source data first")
    shared = sorted(images["validation"] & images["test"])
    # Keep the untouched official test set; remove its shared images from validation only.
    derived = [
        row
        for row in records
        if not (row["split"] == "validation" and str(row["image_id"]) in shared)
    ]
    audit = audit_records(derived)
    names_by_split = {
        split: {row["image_name"] for row in derived if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    name_overlap = (
        names_by_split["train"] & names_by_split["validation"]
        or names_by_split["train"] & names_by_split["test"]
        or names_by_split["validation"] & names_by_split["test"]
    )
    if (
        audit["errors"]
        or audit["counts"].get("duplicate_question_ids", 0)
        or any(audit["image_overlap_by_split"].values())
        or any(audit["image_path_overlap_by_split"].values())
        or name_overlap
    ):
        raise ValueError("Derived split failed the record/overlap audit")
    return derived, shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/vivqax/all.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/vivqax/image_disjoint_v1")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(
            f"Output already exists; do not overwrite a dataset version: {args.output_dir}"
        )
    records = list(read_jsonl(args.input))
    derived, shared = derive_image_disjoint(records)
    args.output_dir.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        write_jsonl(args.output_dir / f"{split}.jsonl", (r for r in derived if r["split"] == split))
    write_jsonl(args.output_dir / "all.jsonl", derived)
    manifest = image_manifest(derived)
    with (args.output_dir / "images.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json(
        args.output_dir / "provenance.json",
        {
            "protocol": "image_disjoint_v1",
            "original_input": str(args.input.resolve()),
            "original_sha256": sha256_file(args.input),
            "policy": "Keep original train/test; exclude test-shared images from validation",
            "shared_image_ids": shared,
            "removed_validation_questions": len(records) - len(derived),
            "original_counts": {
                split: sum(r["split"] == split for r in records)
                for split in ("train", "validation", "test")
            },
            "derived_counts": {
                split: sum(r["split"] == split for r in derived)
                for split in ("train", "validation", "test")
            },
            "audit": audit_records(derived),
        },
    )
    print(
        f"Created {args.output_dir}: {len(derived)} questions, "
        f"excluded {len(shared)} shared validation images / {len(records) - len(derived)} questions"
    )


if __name__ == "__main__":
    main()
