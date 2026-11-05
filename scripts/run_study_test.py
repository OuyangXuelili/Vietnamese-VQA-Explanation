"""Inspect or resume one identity from the frozen final-test plan."""

import argparse
import json
from pathlib import Path

from vqa_explanation.inference.study_test import IDENTITIES, run_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--identity", choices=IDENTITIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rubric", type=Path, help="Directory containing the frozen lock.json")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--recover-from", type=Path,
                        help="Restore a closed snapshot into a new directory without inference")
    args = parser.parse_args()
    try:
        result = run_test(args.environment, args.plan, args.plan_sha256, args.identity,
                          args.output_dir, adapter=args.adapter, rubric=args.rubric,
                          inspect_only=args.inspect, recover_from=args.recover_from)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
