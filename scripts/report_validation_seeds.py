"""Summarize the complete validation reports for the three declared seeds."""

import argparse
from pathlib import Path

from vqa_explanation.evaluation.seed_report import build_seed_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in (42, 43, 44):
        parser.add_argument(f"--seed{seed}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_seed_report({s: getattr(args, f"seed{s}") for s in (42, 43, 44)}, args.output_dir)
    print(f"Verified three-seed validation summary: {args.output_dir}")


if __name__ == "__main__":
    main()
