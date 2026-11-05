"""Prepare validation practice pages or freeze the rubric after complete ratings."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.rubric_calibration import freeze_rubric, prepare_calibration
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--results", type=Path, required=True)
    prepare.add_argument("--environment", type=Path, required=True)
    prepare.add_argument("--study-config", type=Path, required=True)
    prepare.add_argument("--rubric", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--rating", action="append", required=True, help="rater1=PATH")
    freeze.add_argument("--rubric", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        environment = load_environment(args.environment)
        result = prepare_calibration(
            args.results, environment["annotations"].parent / "validation.jsonl",
            environment["annotations"], environment["image_root"], args.study_config,
            args.rubric, args.output_dir)
    else:
        ratings = {}
        for value in args.rating:
            rater, path = value.split("=", 1)
            if rater in ratings:
                parser.error(f"Duplicate rater: {rater}")
            ratings[rater] = Path(path)
        result = freeze_rubric(args.calibration, ratings, args.rubric, args.output_dir)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"sources", "files"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
