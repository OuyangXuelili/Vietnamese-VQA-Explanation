"""Reuse or repair old pairs and screen new train-only pairs on a GPU."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vqa_explanation.preference.expansion import (  # noqa: E402
    package_metadata,
    restore_metadata,
    run_expansion,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    package = commands.add_parser("package", help="Package six metadata files, not images")
    for name in ("old-pairs", "candidates", "selection", "output"):
        package.add_argument("--" + name, type=Path, required=True)
    restore = commands.add_parser("restore", help="Verify and unpack a metadata bundle")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--sha256", required=True)
    restore.add_argument("--work-root", type=Path, required=True)
    run = commands.add_parser("run", help="Screen or resume the image-based expansion")
    for name in ("annotations", "image-root", "old-pairs", "candidates", "selection",
                 "m1-adapter", "m1-selection", "output-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--target", type=int, default=1000)
    run.add_argument("--max-items", type=int, default=0,
                     help="Stop after this many additional decisions; zero processes the pool")
    run.add_argument("--inspect", action="store_true")
    run.add_argument("--teacher-id", help="Screening model only; M1 candidates stay unchanged")
    run.add_argument("--teacher-revision", help="Immutable 40-character screening model commit")
    args = parser.parse_args()
    if args.mode == "package":
        print(json.dumps(package_metadata(args.old_pairs, args.candidates, args.selection,
                                          args.output), indent=2))
        return
    if args.mode == "restore":
        print(restore_metadata(args.archive, args.sha256, args.work_root))
        return
    if args.max_items < 0:
        parser.error("max-items must be non-negative")
    if bool(args.teacher_id) != bool(args.teacher_revision):
        parser.error("teacher-id and teacher-revision must be supplied together")
    args.teacher = ({"id": args.teacher_id, "revision": args.teacher_revision}
                    if args.teacher_id else None)
    print(json.dumps(run_expansion(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
