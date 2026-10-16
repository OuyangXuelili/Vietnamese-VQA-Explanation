"""Prepare, run or score a fixed validation image-permutation diagnostic."""

import argparse
import json
from pathlib import Path

from vqa_explanation.data.io import write_json
from vqa_explanation.evaluation.image_dependence import (
    prepare_images,
    run_image_check,
    score_image_check,
)
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "score", "report"))
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--study-config", type=Path)
    args = parser.parse_args()
    env = load_environment(args.environment)
    if args.mode == "report":
        if not all((args.evidence, args.study_config, args.output_dir)):
            parser.error("report requires evidence, study-config and output-dir")
        from vqa_explanation.evaluation.image_check_report import build_image_report

        result = build_image_report(args.bundle, args.evidence, env["annotations"],
                                    env["image_root"], args.study_config, args.output_dir)
    elif args.mode == "prepare":
        result = prepare_images(env["annotations"], env["image_root"], args.bundle,
                                args.count, args.seed)
    else:
        if args.output_dir is None:
            parser.error("run and score require output-dir")
        if args.mode == "run":
            if args.experiment_id is None:
                parser.error("run requires experiment-id")
            run_image_check(args.bundle, env["annotations"], env["image_root"],
                            args.output_dir, args.experiment_id, args.adapter)
        result = score_image_check(args.bundle, env["annotations"], env["image_root"],
                                   args.output_dir)
        path = args.output_dir / "image_dependence.json"
        if path.exists():
            if json.loads(path.read_text("utf-8")) != result:
                raise ValueError("Preserve the previous diagnostic result")
        else:
            write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
