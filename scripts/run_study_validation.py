"""Export M0–M3 validation predictions without changing training outputs."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from vqa_explanation.inference.study_validation import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--run-group", type=Path, required=True)
    parser.add_argument("--m1-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-config", type=Path,
                        help="Locked main study; omit only for the historical 186-pair run")
    parser.add_argument("--relocate-from", type=Path,
                        help="Copy verified old exports to a new directory after a storage move")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(FileLock(str(args.output_dir) + ".lock", timeout=0))
        if args.relocate_from is not None:
            locks.enter_context(FileLock(str(args.relocate_from.resolve()) + ".lock", timeout=0))
        result = run_validation(args.environment, args.run_group, args.m1_adapter, args.output_dir,
                                inspect_only=args.inspect, relocate_from=args.relocate_from,
                                study_config=args.study_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
