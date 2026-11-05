"""Inspect or run the live-demo GPU acceptance checks."""

import argparse
import json
from pathlib import Path

from vqa_explanation.demo.acceptance import (
    check_output_path,
    inspect_demo,
    load_demo_config,
    run_demo_check,
)
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    config = load_demo_config(args.config)
    environment = load_environment(args.environment)
    plan = inspect_demo(*(config[key] for key in (
        "predictions", "manifest", "annotations")), environment["image_root"],
        config["selection"], args.adapter, replay_count=config["examples"])
    check_output_path(args.output_dir, plan)
    if args.inspect:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        if not args.expected_plan_sha256:
            parser.error("Inspect first and provide --expected-plan-sha256")
        report = run_demo_check(plan, args.output_dir, expected_sha256=args.expected_plan_sha256)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
