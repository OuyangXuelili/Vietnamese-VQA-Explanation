"""Export M0/M1 on the frozen validation split, then compare their predictions."""

import argparse
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock, Timeout

from scripts.compare_runs import read_expected_validation_annotations
from scripts.run_m0 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    read_jsonl,
    select_records,
    validate_selected_records,
)
from vqa_explanation.inference.export import export_inputs, validate_saved_predictions
from vqa_explanation.training.sft_full import validate_checkpoint
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.run_provenance import sha256_file

VALIDATION_SHA256 = "7f0b2b7d374d386030ddea50bc57d3e8e6c76e257e9fc4d1203bfb331b5f5b6e"


def build_commands(
    environment: dict, checkpoint: Path, output: Path, m0_predictions: Path | None = None,
) -> list[list[str]]:
    root = project_root()
    annotations = environment["annotations"].parent / "validation.jsonl"
    shared = [
        sys.executable, "-u", str(root / "scripts/run_m0.py"),
        "--annotations", str(annotations), "--image-root", str(environment["image_root"]),
        "--split", "validation", "--limit", "0", "--seed", "42",
        "--model-id", DEFAULT_MODEL, "--revision", DEFAULT_REVISION,
        "--min-pixels", "200704", "--max-pixels", "200704",
        "--max-new-tokens", "128", "--load-in-4bit", "--resume",
    ]
    commands = [
        [*shared, "--experiment-id", "M0_PRETRAINED", "--output", str(output / "m0.jsonl")],
        [*shared, "--experiment-id", "M1_SFT", "--adapter", str(checkpoint),
         "--output", str(output / "m1.jsonl")],
        [sys.executable, "-u", str(root / "scripts/compare_runs.py"),
         str(output / "m0.jsonl"), str(output / "m1.jsonl"),
         "--output", str(output / "full_validation.json"),
         "--expected-validation-annotations", str(annotations),
         "--expected-validation-sha256", VALIDATION_SHA256],
    ]
    if m0_predictions is not None:
        commands = commands[1:]
        commands[-1][3] = str(m0_predictions)
    return commands


def cache_identity(predictions: Path) -> dict:
    manifest = predictions.with_suffix(predictions.suffix + ".manifest.json")
    return {
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "manifest_sha256": sha256_file(manifest),
        "validator_sha256": sha256_file(Path(__file__)),
    }


def validate_m0_cache(environment: dict, predictions: Path) -> dict:
    """Require a complete export made with the same inputs, source and environment."""
    identity = cache_identity(predictions)
    annotations = environment["annotations"].parent / "validation.jsonl"
    records = select_records(read_jsonl(annotations), "validation", 0, 42)
    args = SimpleNamespace(
        annotations=annotations, model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION,
        experiment_id="M0_PRETRAINED", adapter=None, split="validation", limit=0,
        seed=42, min_pixels=200704, max_pixels=200704, max_new_tokens=128,
        load_in_4bit=True,
    )
    image_hashes = {
        row["question_id"]: sha256_file(environment["image_root"] / row["image_path"])
        for row in records
    }
    manifest = json.loads(
        predictions.with_suffix(predictions.suffix + ".manifest.json").read_text("utf-8")
    )
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported M0 cache manifest")
    if manifest.get("inputs") != export_inputs(args, records, image_hashes):
        raise ValueError(
            "M0 cache inputs/source/environment changed; use the pinned evaluation setup"
        )
    if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"]:
        raise ValueError("Missing inference protocol in M0 cache")
    saved = read_jsonl(predictions)
    completed = validate_saved_predictions(saved, records, manifest)
    if not records or len(completed) != len(records):
        raise ValueError(f"M0 cache is incomplete: {len(completed)}/{len(records)} predictions")
    with predictions.open("rb") as source:
        source.seek(-1, 2)
        if source.read(1) != b"\n":
            raise ValueError("M0 cache has an incomplete final line")
    if cache_identity(predictions) != identity:
        raise ValueError("M0 cache changed during validation")
    return identity


def validate_inputs(environment: dict, checkpoint: Path) -> None:
    references, _ = read_expected_validation_annotations(
        environment["annotations"].parent / "validation.jsonl",
        expected_sha256=VALIDATION_SHA256,
    )
    validate_selected_records(list(references.values()), environment["image_root"])
    validate_checkpoint(checkpoint)
    adapter = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(adapter, dict) or adapter.get("base_model_name_or_path") != DEFAULT_MODEL:
        raise ValueError("M1 adapter does not reference the pinned Qwen base model")


def export_command_for_state(command: list[str]) -> list[str]:
    """Start a missing export; keep strict resume checks for existing artifacts."""
    if "--resume" not in command:
        return list(command)
    output = Path(command[command.index("--output") + 1])
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        return list(command)
    return [argument for argument in command if argument != "--resume"]


def execute(commands: list[list[str]], output: Path) -> None:
    root = project_root()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), environment.get("PYTHONPATH", "")]
    )
    with (output / "console.log").open("a", encoding="utf-8") as log:
        for stage, command in enumerate(commands, start=1):
            command = export_command_for_state(command)
            heading = f"\n[{stage}/{len(commands)}] {json.dumps(command)}\n"
            log.write(heading)
            log.flush()
            print(heading, end="", flush=True)
            with subprocess.Popen(
                command, cwd=root, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f"Validation stage {stage} failed ({code}); inspect console.log")


def run(args) -> None:
    environment = load_environment(args.environment)
    checkpoint, output = args.checkpoint.resolve(), args.output_dir.resolve()
    validate_inputs(environment, checkpoint)
    cached = getattr(args, "m0_predictions", None)
    cached = cached.resolve() if cached is not None else None
    if cached is not None and cached.is_relative_to(output):
        raise ValueError("M0 cache must be outside the new output directory")
    commands = build_commands(environment, checkpoint, output, cached)
    if args.dry_run:
        if cached is not None:
            validate_m0_cache(environment, cached)
        print(json.dumps(commands, indent=2))
        return
    if output.exists() and not args.resume:
        raise ValueError("Output directory exists; inspect it and use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    cache_lock = FileLock(str(cached) + ".lock", timeout=0) if cached else nullcontext()
    with FileLock(str(output / "validation.lock"), timeout=0), cache_lock:
        if (output / "full_validation.json").exists():
            raise ValueError(
                "Comparison report already exists; inspect it without rerunning this job"
            )
        plan = output / "commands.json"
        identity = validate_m0_cache(environment, cached) if cached else None
        cache_plan = output / "m0_cache.json"
        if cache_plan.exists():
            if identity is None or json.loads(cache_plan.read_text("utf-8")) != identity:
                raise ValueError("Saved M0 cache changed; preserve this job and inspect it")
        elif identity is not None and plan.exists():
            raise ValueError("Existing cached job has no cache provenance; inspect it")
        if plan.exists():
            if json.loads(plan.read_text(encoding="utf-8")) != commands:
                raise ValueError("Validation commands changed; choose a new output directory")
        else:
            if list(output.glob("*.jsonl")):
                raise ValueError("Existing predictions have no job plan; preserve and inspect them")
            if identity is not None and not cache_plan.exists():
                with cache_plan.open("x", encoding="utf-8") as handle:
                    json.dump(identity, handle, indent=2)
            with plan.open("x", encoding="utf-8") as handle:
                json.dump(commands, handle, indent=2)
        execute(commands, output)
        if cached is not None and cache_identity(cached) != identity:
            raise ValueError("M0 cache changed during execution; do not use this report")
    print(f"Full validation finished: {output / 'full_validation.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--m0-predictions", type=Path,
        help="Reuse a complete, verified M0 export without copying or generating it again",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check inputs and print commands; no GPU"
    )
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, Timeout) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
