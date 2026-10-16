"""Export M0/M1 validation figures and fixed diagnostic examples."""

import argparse
from pathlib import Path

from vqa_explanation.evaluation.validation_report import build_validation_report
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--m0-semantic", type=Path)
    parser.add_argument("--m1-semantic", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.m0_semantic) != bool(args.m1_semantic):
        parser.error("Use both semantic reports or neither")
    env = load_environment(args.environment)
    semantic = (args.m0_semantic, args.m1_semantic) if args.m0_semantic else None
    report = build_validation_report(
        args.m0, args.m1, args.annotations, env["image_root"], args.output_dir,
        annotations_sha256=args.annotations_sha256, semantic_paths=semantic)
    print(f"Saved {report['n']} validation questions and figures: {args.output_dir}")


if __name__ == "__main__":
    main()
