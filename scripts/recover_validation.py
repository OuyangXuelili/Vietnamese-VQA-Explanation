"""Recover closed validation snapshots into a separate, traceable attempt."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vqa_explanation.inference.study_validation import recover_validation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("environment", "run-group", "m1-adapter", "source-dir", "output-dir",
                 "study-config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        for directory in sorted({args.source_dir.resolve(), args.output_dir.resolve()}):
            locks.enter_context(FileLock(str(directory) + ".lock", timeout=0))
        result = recover_validation(
            args.environment, args.run_group, args.m1_adapter, args.source_dir, args.output_dir,
            study_config=args.study_config, inspect_only=args.inspect,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
