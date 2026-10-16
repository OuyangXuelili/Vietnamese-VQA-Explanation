from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def image_ids_by_split(records: Iterable[dict]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split = "validation" if row.get("split") == "val" else row.get("split")
        result[str(split)].add(str(row.get("image_id")))
    return dict(result)


def assert_no_image_leakage(records: Iterable[dict]) -> None:
    grouped = image_ids_by_split(records)
    names = sorted(grouped)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = grouped[left] & grouped[right]
            if shared:
                example = sorted(shared)[:5]
                raise ValueError(f"image leakage between {left} and {right}: {example}")

