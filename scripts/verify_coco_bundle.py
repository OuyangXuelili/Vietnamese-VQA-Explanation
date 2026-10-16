"""Verify a transferred ViVQA-X bundle and optionally extract it safely."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from vqa_explanation.data.coco import sha256_file

IMAGE_ENTRY = re.compile(r"images/COCO_(train2014|val2014)_\d{12}\.jpg\Z")
EXPECTED_ARCHIVES = {
    "coco_train2014.zip",
    "coco_val2014.zip",
    "annotations_image_disjoint_v1.zip",
}
METADATA_ENTRIES = {
    "metadata/images.csv",
    "metadata/full_image_checksums.csv",
    "metadata/all.jsonl",
    "metadata/train.jsonl",
    "metadata/validation.jsonl",
    "metadata/test.jsonl",
    "metadata/provenance.json",
}


def verify_bundle(bundle_dir: Path, extract_root: Path | None = None) -> dict:
    bundle_file = bundle_dir / "bundle.json"
    digest_file = bundle_dir / "bundle.sha256"
    if not bundle_file.is_file() or not digest_file.is_file():
        raise FileNotFoundError("bundle.json and bundle.sha256 are required")
    expected_bundle_digest = digest_file.read_text(encoding="utf-8").split()[0]
    if sha256_file(bundle_file) != expected_bundle_digest:
        raise ValueError("bundle.json SHA-256 mismatch")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    if (
        bundle.get("format") != "vivqax_image_disjoint_v1_bundle"
        or bundle.get("status") != "complete"
    ):
        raise ValueError("Unexpected or incomplete bundle manifest")
    archives = bundle.get("archives", {})
    if set(archives) != EXPECTED_ARCHIVES:
        raise ValueError("Unexpected archive set")
    total_images = 0
    for name in sorted(archives):
        path = bundle_dir / name
        if not path.is_file() or path.stat().st_size != archives[name]["bytes"]:
            raise ValueError(f"Archive missing or size-mismatched: {name}")
        if sha256_file(path) != archives[name]["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {name}")
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            if len(entries) != len(set(entries)):
                raise ValueError(f"Duplicate archive entries: {name}")
            if name.startswith("coco_"):
                split = name.removeprefix("coco_").removesuffix(".zip")
                if len(entries) != archives[name]["images"] or any(
                    not (match := IMAGE_ENTRY.fullmatch(entry)) or match.group(1) != split
                    for entry in entries
                ):
                    raise ValueError(f"Unsafe or incomplete image entries: {name}")
                total_images += len(entries)
            elif set(entries) != METADATA_ENTRIES:
                raise ValueError("Metadata archive entries differ from protocol")
    if total_images != bundle["total_images"]:
        raise ValueError("Image count differs from bundle manifest")

    if extract_root is not None:
        if extract_root.exists():
            raise FileExistsError(f"Extraction target already exists: {extract_root}")
        extract_root.parent.mkdir(parents=True, exist_ok=True)
        required = (
            bundle["total_source_image_bytes"]
            + archives["annotations_image_disjoint_v1.zip"]["bytes"]
        )
        if shutil.disk_usage(extract_root.parent).free < required + 256 * 1024 * 1024:
            raise OSError("Not enough free space to extract the bundle")
        extract_root.mkdir()
        for name in sorted(archives):
            with zipfile.ZipFile(bundle_dir / name) as archive:
                archive.extractall(extract_root)
        if len(list((extract_root / "images").glob("*.jpg"))) != total_images:
            raise ValueError("Extracted image count mismatch")
    return {
        "status": "verified",
        "images": total_images,
        "extracted_to": str(extract_root) if extract_root else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle_dir, args.extract_to), ensure_ascii=False))


if __name__ == "__main__":
    main()
