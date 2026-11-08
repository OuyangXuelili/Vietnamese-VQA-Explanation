"""Start, verify and hand off a bounded task in this checkout."""

import argparse
import json
from pathlib import Path

from filelock import Timeout

from vqa_explanation.utils import task_harness
from vqa_explanation.utils.environment import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Record scope and observable acceptance criteria")
    start.add_argument("--title", required=True)
    start.add_argument("--kind", choices=task_harness.KINDS, required=True)
    start.add_argument("--accept", action="append", required=True)

    check = commands.add_parser("check", help="Execute existing checks and save the real outcome")
    check.add_argument("task", type=Path)
    check.add_argument("--profile", choices=task_harness.PROFILES, default="full")
    check.add_argument("--test", action="append", default=[])
    check.add_argument("--source-only", action="store_true")

    review = commands.add_parser("review", help="Record a criterion review with hashed evidence")
    review.add_argument("task", type=Path)
    review.add_argument("--criterion", type=int, required=True)
    review.add_argument("--evidence", type=Path, action="append", required=True)
    review.add_argument("--note", required=True)
    review.add_argument("--result", choices=("passed", "failed", "blocked"), default="passed")

    finish = commands.add_parser("finish", help="Validate completion or save an unfinished handoff")
    finish.add_argument("task", type=Path)
    finish.add_argument("--status", choices=task_harness.STATES, required=True)
    finish.add_argument("--summary", required=True)
    finish.add_argument("--next-step", default="")

    status = commands.add_parser("status", help="Read context pointers and recent task handoffs")
    status.add_argument("task", type=Path, nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != project_root().resolve():
        parser.error("Python imports another checkout. Use this checkout's environment first.")
    try:
        if args.command == "start":
            result = {"task": str(task_harness.start(root, args.title, args.kind, args.accept)),
                      "status": "in_progress"}
        elif args.command == "check":
            result = task_harness.check(root, args.task, args.profile, args.test, args.source_only)
        elif args.command == "review":
            result = task_harness.review(root, args.task, args.criterion, args.evidence,
                                         args.note, args.result)
        elif args.command == "finish":
            output = task_harness.finish(root, args.task, args.status, args.summary, args.next_step)
            result = {"handoff": str(output), "status": args.status}
        else:
            result = task_harness.inspect(root, args.task)
    except (ValueError, OSError, Timeout, KeyError, TypeError) as error:
        parser.exit(1, f"Task harness: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result.get("returncode", 0)


if __name__ == "__main__":
    raise SystemExit(main())
