from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coco import COCO_NAME_PATTERN
from .schema import normalize_identifier

RAW_FILES = {
    "train": "ViVQA-X_train.json",
    "validation": "ViVQA-X_val.json",
    "test": "ViVQA-X_test.json",
}


def load_raw_split(annotation_dir: str | Path, split: str) -> list[dict[str, Any]]:
    if split not in RAW_FILES:
        raise ValueError(f"unsupported split: {split}")
    source = Path(annotation_dir) / RAW_FILES[split]
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {source}")
    return payload


def normalize_record(raw: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in RAW_FILES:
        raise ValueError(f"unsupported split: {split}")
    if not isinstance(raw, dict):
        raise ValueError("raw sample must be a JSON object")
    image_name = str(raw.get("image_name", "")).strip()
    if not COCO_NAME_PATTERN.fullmatch(image_name):
        raise ValueError(f"invalid COCO image_name: {image_name!r}")
    return {
        "question_id": normalize_identifier(raw.get("question_id"), "question_id"),
        "image_id": normalize_identifier(raw.get("image_id"), "image_id"),
        "image_name": image_name,
        "image_path": f"images/{image_name}",
        "question": raw.get("question"),
        "answer": raw.get("answer"),
        "explanations": raw.get("explanation"),
        "question_type": raw.get("question_type"),
        "answer_type": raw.get("answer_type"),
        "split": split,
    }


def image_manifest(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    splits_by_image: dict[str, set[str]] = defaultdict(set)
    image_ids: dict[str, str] = {}
    names_by_id: dict[str, str] = {}
    for record in records:
        name = record["image_name"]
        image_id = record["image_id"]
        if not COCO_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid COCO image_name: {name!r}")
        if name in image_ids and image_ids[name] != image_id:
            raise ValueError(f"image_name maps to multiple image IDs: {name}")
        if image_id in names_by_id and names_by_id[image_id] != name:
            raise ValueError(f"image_id maps to multiple image names: {image_id}")
        splits_by_image[name].add(record["split"])
        image_ids[name] = image_id
        names_by_id[image_id] = name

    rows = []
    for name in sorted(splits_by_image):
        coco_split = "train2014" if "_train2014_" in name else "val2014"
        rows.append(
            {
                "image_name": name,
                "image_id": image_ids[name],
                "coco_split": coco_split,
                "dataset_splits": ",".join(sorted(splits_by_image[name])),
            }
        )
    return rows
