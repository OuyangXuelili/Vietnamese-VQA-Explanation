from __future__ import annotations

import argparse
import json

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.preference.freeze import freeze_pairs
from vqa_explanation.preference.validate import validate_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate pair schema or freeze reviewed train pairs"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--annotations", help="Canonical all.jsonl (required for freeze)")
    parser.add_argument("--image-root", help="Root containing canonical image paths")
    parser.add_argument("--output-dir", help="New directory for matched M2/M3 exports")
    args = parser.parse_args()
    freeze_args = (args.annotations, args.image_root, args.output_dir)
    if any(freeze_args):
        if not all(freeze_args):
            parser.error("Freeze requires --annotations, --image-root and --output-dir together")
        report = freeze_pairs(args.input, *freeze_args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report = validate_pairs(read_jsonl(args.input))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["counts"].get("invalid", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
