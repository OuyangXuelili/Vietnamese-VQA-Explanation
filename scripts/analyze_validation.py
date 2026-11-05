"""Build a verified three-seed validation diagnostic report and offline comparison page."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.validation_diagnostics_bundle import build_diagnostics
from vqa_explanation.utils.environment import load_environment, project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/evaluation/validation_diagnostics.yaml",
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    report = build_diagnostics(
        args.config, load_environment(args.environment), args.output_dir, resamples=args.resamples
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "schema",
                    "n",
                    "n_images",
                    "unique_inference_runs",
                    "images_verified",
                    "answer_stability",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
