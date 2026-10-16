"""Launch inspection, one GPU gate, or main training from an explicit record."""

import argparse
import json
from pathlib import Path

from vqa_explanation.training.preference import run_stage
from vqa_explanation.training.preference_launch import prepare_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("m2", "m3"), required=True)
    parser.add_argument("--mode", choices=("inspect", "gate", "train"), default="inspect")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-gate", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    report, stage = prepare_launch(
        args.launch_record, args.project_dir, args.arm, args.mode,
        args.output_dir, args.gpu_gate, args.resume_from_checkpoint,
    )
    if stage is None or args.mode == "inspect":
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0 if args.mode == "inspect" else 2
    run_stage(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
