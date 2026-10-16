"""Package the verified ViVQA-X image set for transfer to a Colab runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.coco import load_manifest, sha256_file
from vqa_explanation.data.io import read_jsonl

FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
METADATA_FILES = ("all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "provenance.json")
DATASET_SPLITS = ("train", "validation", "test")


def read_checksums(path: Path) -> dict[str, tuple[int, str]]:
    checksums: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"image_name", "size_bytes", "sha256"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Checksum CSV must contain image_name, size_bytes and sha256")
        for row in reader:
            name = row["image_name"]
            if name in checksums:
                raise ValueError(f"Duplicate image checksum: {name}")
            size = int(row["size_bytes"])
            digest = row["sha256"].lower()
            if size <= 0 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid checksum entry: {name}")
            checksums[name] = (size, digest)
    return checksums


def write_entry(archive: zipfile.ZipFile, source: Path, name: str, expected: str) -> None:
    entry = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    entry.compress_type = zipfile.ZIP_STORED
    entry.external_attr = 0o644 << 16
    digest = hashlib.sha256()
    with source.open("rb") as input_file, archive.open(entry, "w", force_zip64=True) as output:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Source changed since verification: {source}")


def verify_archive(path: Path, expected: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError(f"Archive entries differ from manifest: {path}")
        for name, digest in expected.items():
            actual = hashlib.sha256()
            with archive.open(name) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    actual.update(chunk)
            if actual.hexdigest() != digest:
                raise ValueError(f"Archive content hash mismatch: {name}")


def validate_annotations(annotations_dir: Path, image_names: set[str]) -> None:
    """Reject a bundle whose all/split annotations are inconsistent or leaky."""
    all_records = list(read_jsonl(annotations_dir / "all.jsonl"))
    if not all_records:
        raise ValueError("Annotation manifest is empty")
    audit = audit_records(all_records)
    if (
        audit["errors"]
        or audit["counts"].get("duplicate_question_ids", 0)
        or any(audit["image_overlap_by_split"].values())
        or any(audit["image_path_overlap_by_split"].values())
    ):
        raise ValueError("Annotation manifest failed the record/split audit")

    actual_names: set[str] = set()
    for row in all_records:
        if row.get("split") not in DATASET_SPLITS:
            raise ValueError("Annotation split must be train, validation or test")
        name = row.get("image_name")
        if (
            not isinstance(name, str)
            or name not in image_names
            or row.get("image_path") != f"images/{name}"
        ):
            raise ValueError("Annotation image name/path differs from image manifest")
        actual_names.add(name)
    if actual_names != image_names:
        raise ValueError("Annotation images and image manifest differ")

    for split in DATASET_SPLITS:
        expected = [row for row in all_records if row["split"] == split]
        actual = list(read_jsonl(annotations_dir / f"{split}.jsonl"))
        if actual != expected:
            raise ValueError(f"{split}.jsonl differs from all.jsonl")


def package_dataset(
    manifest_path: Path,
    checksums_path: Path,
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
) -> dict:
    if manifest_path.name != "images.csv" or checksums_path.name != "full_image_checksums.csv":
        raise ValueError("Expected canonical manifest/checksum filenames for this bundle format")
    rows = load_manifest(manifest_path)
    checksums = read_checksums(checksums_path)
    names = {row["image_name"] for row in rows}
    if set(checksums) != names:
        raise ValueError("Checksum CSV and image manifest contain different image names")
    metadata_paths = [annotations_dir / name for name in METADATA_FILES]
    for path in [manifest_path, checksums_path, *metadata_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for row in rows:
        image = images_dir / row["image_name"]
        if not image.is_file() or image.stat().st_size != checksums[row["image_name"]][0]:
            raise ValueError(f"Missing or size-mismatched image: {image}")
    validate_annotations(annotations_dir, names)
    provenance = json.loads((annotations_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("protocol") != "image_disjoint_v1":
        raise ValueError("Expected image_disjoint_v1 annotation provenance")
    if output_dir.exists():
        raise FileExistsError(f"Dataset bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    required = sum(size for size, _ in checksums.values())
    if shutil.disk_usage(output_dir.parent).free < required + 256 * 1024 * 1024:
        raise OSError("Not enough free space to create the dataset archives")
    output_dir.mkdir(parents=True)

    bundles: dict[str, dict] = {}
    for coco_split in ("train2014", "val2014"):
        selected = [row for row in rows if row["coco_split"] == coco_split]
        archive_path = output_dir / f"coco_{coco_split}.zip"
        expected = {
            f"images/{row['image_name']}": checksums[row["image_name"]][1] for row in selected
        }
        with zipfile.ZipFile(archive_path, "x", allowZip64=True) as archive:
            for index, row in enumerate(selected, 1):
                name = row["image_name"]
                write_entry(archive, images_dir / name, f"images/{name}", checksums[name][1])
                if index % 2000 == 0:
                    print(f"{coco_split}: packed {index}/{len(selected)}", flush=True)
        verify_archive(archive_path, expected)
        bundles[archive_path.name] = {
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
            "images": len(selected),
        }
        print(f"Verified {archive_path.name}: {len(selected)} images", flush=True)

    metadata = [manifest_path, checksums_path, *metadata_paths]
    metadata_archive = output_dir / "annotations_image_disjoint_v1.zip"
    metadata_expected = {f"metadata/{path.name}": sha256_file(path) for path in metadata}
    with zipfile.ZipFile(metadata_archive, "x", allowZip64=True) as archive:
        for path in metadata:
            write_entry(
                archive, path, f"metadata/{path.name}", metadata_expected[f"metadata/{path.name}"]
            )
    verify_archive(metadata_archive, metadata_expected)
    bundles[metadata_archive.name] = {
        "sha256": sha256_file(metadata_archive),
        "bytes": metadata_archive.stat().st_size,
        "files": len(metadata),
    }
    report = {
        "format": "vivqax_image_disjoint_v1_bundle",
        "manifest_sha256": sha256_file(manifest_path),
        "image_checksums_sha256": sha256_file(checksums_path),
        "total_images": len(rows),
        "total_source_image_bytes": required,
        "archives": bundles,
        "status": "complete",
    }
    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "bundle.sha256").write_text(
        f"{sha256_file(bundle_path)}  {bundle_path.name}\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/vivqax/manifests/images.csv")
    )
    parser.add_argument(
        "--checksums", type=Path, default=Path("data/processed/vivqax/full_image_checksums.csv")
    )
    parser.add_argument("--images", type=Path, default=Path("data/raw/coco/images"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/processed/vivqax/image_disjoint_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/datasets/vivqax_image_disjoint_v1")
    )
    args = parser.parse_args()
    result = package_dataset(
        args.manifest, args.checksums, args.images, args.annotations, args.output
    )
    print(f"Dataset bundle complete: {args.output}; {result['total_images']} images")


if __name__ == "__main__":
    main()
