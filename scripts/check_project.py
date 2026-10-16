"""CPU-only project gate: lint, tests, and optionally dataset checks."""

import argparse
import json
import subprocess
import sys

from vqa_explanation.utils.environment import project_root


def commands(with_data: bool, full_data: bool) -> list[list[str]]:
    root = project_root()
    checks = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    if with_data or full_data:
        environment = (
            "configs/environments/local_full.yaml"
            if full_data
            else ("configs/environments/local.yaml")
        )
        base = [sys.executable, str(root / "scripts/run_pipeline.py")]
        checks.append([*base, "audit", "--environment", environment])
        preflight = [*base, "preflight", "--environment", environment]
        if full_data:
            preflight.append("--all-train-records")
        checks.append(preflight)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true", help="Audit/preflight smoke data")
    parser.add_argument("--full-data", action="store_true", help="Audit/preflight full data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.with_data and args.full_data:
        parser.error("Choose either --with-data or --full-data")
    checks = commands(args.with_data, args.full_data)
    if args.dry_run:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0
    root = project_root()
    for index, command in enumerate(checks, start=1):
        print(f"[{index}/{len(checks)}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode != 0:
            print(f"Project gate failed at check {index}; return code {result.returncode}")
            return result.returncode
    print("Project gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
