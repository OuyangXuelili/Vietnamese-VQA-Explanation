"""Verify training images against the fixed, per-file checksum manifest."""

import csv
import re
from collections.abc import Iterable
from pathlib import Path

from vqa_explanation.utils.run_provenance import sha256_file


def verify_record_images(
    records: Iterable[dict], image_root: Path, checksums_csv: Path
) -> dict[str, int | str]:
    """Hash each distinct referenced image before a training or resume session."""
    root = image_root.resolve()
    if not checksums_csv.is_file():
        raise ValueError(f"Image checksum manifest is missing: {checksums_csv}")

    expected: dict[str, tuple[int, str]] = {}
    with checksums_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image_name", "size_bytes", "sha256"]:
            raise ValueError("Image checksum manifest has unexpected columns")
        for row in reader:
            name = row["image_name"]
            digest = row["sha256"]
            try:
                size = int(row["size_bytes"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid image size in checksum manifest: {name}") from error
            if (
                not name
                or Path(name).name != name
                or name in expected
                or size <= 0
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest or "")
            ):
                raise ValueError(f"Invalid image checksum entry: {name}")
            expected[name] = (size, digest.lower())

    verified: set[Path] = set()
    verified_bytes = 0
    for record in records:
        path = (root / record["image_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {record['image_path']}")
        if path in verified:
            continue
        name = path.name
        if record.get("image_name", name) != name or name not in expected:
            raise ValueError(f"Image is absent from checksum manifest: {record['image_path']}")
        expected_size, expected_digest = expected[name]
        actual_size = path.stat().st_size
        if actual_size != expected_size or sha256_file(path) != expected_digest:
            raise ValueError(f"Image checksum mismatch: {record['image_path']}")
        verified.add(path)
        verified_bytes += actual_size

    return {
        "verified_images": len(verified),
        "verified_bytes": verified_bytes,
        "checksum_manifest_sha256": sha256_file(checksums_csv),
    }
