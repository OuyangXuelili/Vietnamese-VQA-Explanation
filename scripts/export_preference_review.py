"""Convert explicitly reviewed notes; never generate human labels."""

import argparse
import json
from pathlib import Path

from vqa_explanation.preference.bundle import verify_bundle
from vqa_explanation.preference.review import export_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, inputs, predictions = verify_bundle(args.bundle, args.expected_sha256)
    manifest = json.loads(data["candidates.jsonl.manifest.json"])
    print(
        export_review(
            args.notes,
            inputs,
            predictions,
            manifest,
            args.expected_sha256,
            args.annotations,
            args.image_root,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
