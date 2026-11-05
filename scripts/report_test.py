"""Inspect progress or score the eight planned final-test exports."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.test_report import build_test_report
from vqa_explanation.inference.study_test import IDENTITIES


def paths_by_identity(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in IDENTITIES or not path or name in result:
            raise ValueError("Use one distinct planned IDENTITY=PATH per input")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="IDENTITY=DIR override")
    parser.add_argument("--semantic", action="append", default=[], help="IDENTITY=JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if not args.inspect and args.output_dir is None:
        parser.error("Scoring requires a new --output-dir")
    try:
        runs = {name: args.runs_root / name for name in IDENTITIES}
        runs.update(paths_by_identity(args.run))
        report = build_test_report(
            args.plan, args.plan_sha256, args.annotations, runs,
            semantic_paths=paths_by_identity(args.semantic) if args.semantic else None,
            output_dir=args.output_dir, inspect_only=args.inspect)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report if args.inspect else {
        "status": report["status"], "split": report["split"], "questions_per_identity": report["n"],
        "human_evaluation_complete": False, "output": str(args.output_dir)},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
