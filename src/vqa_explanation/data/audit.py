from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from .schema import VQASample


def audit_records(
    records: Iterable[dict],
    verify_images: bool = False,
    image_root: str | Path | None = None,
) -> dict:
    counts = Counter()
    errors: list[dict] = []
    images_by_split: dict[str, set[str]] = defaultdict(set)
    paths_by_split: dict[str, set[str | Path]] = defaultdict(set)
    seen_questions: set[str] = set()
    root = Path(image_root).resolve() if image_root is not None else None

    for index, raw in enumerate(records):
        counts["records"] += 1
        try:
            sample = VQASample.from_dict(raw)
            sample.validate_ids()
        except (TypeError, ValueError) as exc:
            errors.append({"index": index, "error": str(exc)})
            counts["invalid"] += 1
            continue
        counts[f"split_{sample.split}"] += 1
        counts["explanations"] += len(sample.explanations)
        images_by_split[sample.split].add(sample.image_id)
        if sample.question_id in seen_questions:
            counts["duplicate_question_ids"] += 1
        seen_questions.add(sample.question_id)
        image_path = Path(sample.image_path)
        if root is not None:
            image_path = (root / image_path).resolve()
        # Resolve aliases when a root is supplied, even without decoding images.
        paths_by_split[sample.split].add(image_path if root is not None else sample.image_path)
        if verify_images:
            if root is None:
                raise ValueError("image_root is required for image verification")
            if not image_path.is_relative_to(root):
                counts["unsafe_image_paths"] += 1
            elif not image_path.is_file():
                counts["missing_images"] += 1
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, ValueError, Image.DecompressionBombError):
                    counts["corrupt_images"] += 1

    overlaps = {}
    path_overlaps = {}
    split_names = sorted(images_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            shared = images_by_split[left] & images_by_split[right]
            overlaps[f"{left}__{right}"] = len(shared)
            path_overlaps[f"{left}__{right}"] = len(
                paths_by_split[left] & paths_by_split[right]
            )

    return {
        "counts": dict(sorted(counts.items())),
        "unique_images_by_split": {k: len(v) for k, v in sorted(images_by_split.items())},
        "image_overlap_by_split": overlaps,
        "image_path_overlap_by_split": path_overlaps,
        "errors": errors,
    }
