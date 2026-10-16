"""Run matched M2/M3 stages or inspect their inputs without a GPU."""

import argparse
import json
from pathlib import Path

from vqa_explanation.training.preference import run_stage
from vqa_explanation.training.preference_data import adapter_fingerprint, validate_study_arm


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-adapter", type=Path)
    parser.add_argument(
        "--environment", type=Path, default=Path("configs/environments/local_full.yaml")
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--study-config", type=Path,
        default=Path("configs/experiments/study_budget_v2.yaml"),
        help="Declared paired comparison; stored in the run identity",
    )
    parser.add_argument("--check-comparison", action="store_true",
                        help="Check both arm configs without data or a GPU")
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen25_vl_3b.yaml")
    )
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--m1-adapter", type=Path)
    parser.add_argument("--m1-selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--gpu-gate", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    if args.inspect_adapter:
        print(json.dumps(adapter_fingerprint(args.inspect_adapter), indent=2))
        return
    if args.check_comparison:
        if args.config is None:
            parser.error("config is required to check the comparison")
        print(json.dumps(validate_study_arm(args.config, args.study_config), indent=2))
        return
    if any(
        getattr(args, name) is None
        for name in ("config", "preferences", "m1_adapter", "m1_selection")
    ):
        parser.error("config, preferences, m1-adapter and m1-selection are required")
    if args.pilot_steps < 0 or args.pilot_steps > 10:
        parser.error("pilot-steps must be 0 (full stage) or 1–10")
    run_stage(args)


if __name__ == "__main__":
    main()
