from __future__ import annotations

import csv
import hashlib
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image

COCO_IMAGE_BASE_URL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org"
COCO_NAME_PATTERN = re.compile(r"COCO_(train2014|val2014)_\d{12}\.jpg\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"image_name", "coco_split"}.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain image_name and coco_split columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    names: set[str] = set()
    for row in rows:
        name = row["image_name"]
        match = COCO_NAME_PATTERN.fullmatch(name)
        if not match or match.group(1) != row["coco_split"]:
            raise ValueError(f"Invalid COCO image name/split: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate image in manifest: {name}")
        names.add(name)
    return sorted(rows, key=lambda row: row["image_name"])


def select_smoke_images(
    rows: Iterable[dict[str, str]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    """Select an image-disjoint, deterministic subset for pipeline smoke tests."""
    groups: dict[str, list[dict[str, str]]] = {split: [] for split in quotas}
    for row in rows:
        split = row["dataset_splits"]
        if split in groups:
            groups[split].append(row)

    selected: list[dict[str, str]] = []
    for offset, (split, size) in enumerate(quotas.items()):
        candidates = sorted(groups[split], key=lambda item: item["image_name"])
        if len(candidates) < size:
            raise ValueError(f"not enough exclusive {split} images: {len(candidates)} < {size}")
        rng = random.Random(seed + offset)
        selected.extend(rng.sample(candidates, size))
    return sorted(selected, key=lambda item: (item["dataset_splits"], item["image_name"]))


def coco_image_url(row: dict[str, str]) -> str:
    return f"{COCO_IMAGE_BASE_URL}/{row['coco_split']}/{row['image_name']}"


def valid_jpeg(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return False
    try:
        with Image.open(source) as image:
            if image.format != "JPEG":
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def download_image(
    row: dict[str, str],
    output_dir: str | Path,
    timeout: int = 30,
    retries: int = 3,
) -> dict[str, str]:
    destination = Path(output_dir) / row["image_name"]
    if valid_jpeg(destination):
        return {"image_name": row["image_name"], "status": "skipped", "error": ""}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(coco_image_url(row), headers={"User-Agent": "vivqa-nle/0.1"})
    last_error = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                temporary.write_bytes(response.read())
            if not valid_jpeg(temporary):
                raise ValueError("downloaded file is not a valid JPEG")
            temporary.replace(destination)
            return {"image_name": row["image_name"], "status": "downloaded", "error": ""}
        except Exception as exc:  # network failures need to be reported per image
            temporary.unlink(missing_ok=True)
            last_error = str(exc)
            if attempt + 1 < retries:
                time.sleep(1.5**attempt)
    return {"image_name": row["image_name"], "status": "failed", "error": last_error}


def download_images(
    rows: Iterable[dict[str, str]],
    output_dir: str | Path,
    workers: int = 8,
) -> list[dict[str, str]]:
    records = list(rows)
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_image, row, output_dir): row for row in records}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["image_name"])
