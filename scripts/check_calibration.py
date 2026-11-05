"""Inspect calibration drafts and list disagreements after independent completion."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.calibration_progress import (
    inspect_calibration,
    save_calibration_progress,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rating", action="append", default=[], help="rater1=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ratings = {}
    for value in args.rating:
        name, path = value.split("=", 1)
        if not name or not path or name in ratings:
            parser.error("Use one distinct rater=PATH per rating file")
        ratings[name] = Path(path)
    report = inspect_calibration(args.calibration, ratings or None)
    save_calibration_progress(report, args.output_dir)
    print(json.dumps({"status": report["status"], "raters": {
        key: {name: row[name] for name in ("expected", "complete", "remaining")}
        for key, row in report["raters"].items()},
        "rubric_frozen": False, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
