from __future__ import annotations

import argparse

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized ViVQA-X JSONL data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--image-root")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on invalid data or split overlap"
    )
    args = parser.parse_args()
    report = audit_records(
        read_jsonl(args.input),
        verify_images=args.verify_images,
        image_root=args.image_root,
    )
    write_json(args.output, report)
    print(f"Audit written to {args.output}")
    print(report["counts"])
    if args.strict and (
        not report["counts"].get("records", 0)
        or any(
            report["counts"].get(k, 0)
            for k in (
                "invalid",
                "duplicate_question_ids",
                "missing_images",
                "corrupt_images",
                "unsafe_image_paths",
            )
        )
        or any(report["image_overlap_by_split"].values())
        or any(report["image_path_overlap_by_split"].values())
    ):
        raise SystemExit("Dataset audit failed; inspect the saved report")


if __name__ == "__main__":
    main()
