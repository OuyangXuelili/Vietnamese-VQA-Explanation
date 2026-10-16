"""Freeze a test sample, prepare blinded pages, or score independent ratings."""

import argparse
from pathlib import Path

from vqa_explanation.data.io import write_json
from vqa_explanation.evaluation.blind_review import (
    freeze_sample,
    prepare_blind_review,
    summarize_blind_review,
)
from vqa_explanation.utils.environment import load_environment


def assignments(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate input: {name}")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-sample")
    freeze.add_argument("--annotations", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--count", type=int, default=200)
    freeze.add_argument("--seed", type=int, default=42)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--annotations", type=Path, required=True)
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--environment", type=Path, required=True)
    prepare.add_argument("--prediction", action="append", required=True, help="EXPERIMENT_ID=PATH")
    prepare.add_argument("--output-dir", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--rating", action="append", required=True, help="rater1=PATH")
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-sample":
        freeze_sample(args.annotations, args.output, count=args.count, seed=args.seed)
    elif args.command == "prepare":
        env = load_environment(args.environment)
        prepare_blind_review(args.annotations, args.sample, assignments(args.prediction),
                             env["image_root"], args.output_dir)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, summarize_blind_review(args.bundle, assignments(args.rating)))


if __name__ == "__main__":
    main()
