"""Select a reproducible, train-only question pool for preference annotation."""

import argparse
import json

from vqa_explanation.preference.selection import freeze_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-selection", action="append", default=[],
                        help="Frozen question pools to exclude by image, repeatable")
    args = parser.parse_args()
    report = freeze_selection(
        args.annotations, args.image_root, args.output_dir, args.count, args.seed,
        args.exclude_selection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
