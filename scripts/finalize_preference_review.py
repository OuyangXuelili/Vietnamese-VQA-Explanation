"""Audit a saved review and optionally export/freeze its accepted subset."""

import argparse
import json
from pathlib import Path

from vqa_explanation.preference.handoff import build_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pairs", type=int, default=1000,
                        help="Planning target, not an accepted count or training budget")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--expected-pairs", type=int,
                        help="Exact accepted count declared before freeze")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    try:
        report = build_handoff(
            args.review_dir, args.notes, args.pilot, args.output_dir,
            target_pairs=args.target_pairs, freeze=args.freeze,
            expected_pairs=args.expected_pairs, bundle=args.bundle,
            expected_bundle_sha256=args.expected_bundle_sha256,
            annotations=args.annotations, image_root=args.image_root,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Review handoff failed: {error}\n")
    print(json.dumps({key: report[key] for key in (
        "status", "review_counts", "freeze_requested", "freeze_blockers",
        "frozen_bundle", "frozen_pairs", "training_gate_open", "gpu_training_started",
    )}, ensure_ascii=True, indent=2))
    if report["status"] == "invalid":
        return 1
    return 2 if args.freeze and report["freeze_blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
