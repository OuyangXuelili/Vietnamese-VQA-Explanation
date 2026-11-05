"""Inspect or execute fixed numerical regression cases on a CUDA GPU."""

import argparse
import json
from pathlib import Path

from vqa_explanation.inference.stability import prepare_stability, run_stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path,
                        default=Path("configs/environments/local_full.yaml"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/evaluation/numerical_stability.yaml"))
    parser.add_argument("--model-config", type=Path,
                        default=Path("configs/model/qwen25_vl_3b.yaml"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    env, plan = prepare_stability(args.environment, args.config, args.model_config)
    if not args.execute:
        print(json.dumps({"status": "inputs_verified", "plan": plan,
                          "gpu_execution_verified": False}, ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--execute requires a new --output-dir")
    report = run_stability(env, plan, args.output_dir, adapter=args.adapter)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
