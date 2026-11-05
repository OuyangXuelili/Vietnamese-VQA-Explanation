"""Record a pretest instruction freeze; leave human ratings unconfirmed."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.guidelines import freeze_guidelines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "calibration", "reference", "guidelines", "annotations",
                 "test-runs-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    result = freeze_guidelines(
        args.plan, args.plan_sha256, args.calibration, args.reference, args.guidelines,
        args.annotations, args.test_runs_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
