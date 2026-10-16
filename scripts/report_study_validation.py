"""Build the four-arm validation report from saved predictions and training records."""

import argparse
from pathlib import Path

from vqa_explanation.evaluation.study_report import build_study_report
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    environment = load_environment(args.environment)
    report = build_study_report(
        args.results, args.annotations, environment["image_root"], args.output_dir,
        study_config=args.study_config, semantic_dir=args.semantic_dir)
    print(f"Verified {report['n']} questions in each of four arms: {args.output_dir}")


if __name__ == "__main__":
    main()
