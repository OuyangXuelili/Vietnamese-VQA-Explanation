"""Rehearse an M1 optimizer-pilot pause at step 10 and resume to step 20."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from vqa_explanation.training.sft_full import validate_checkpoint
from vqa_explanation.utils.environment import load_environment, project_root

STAGE = "sft-optimizer-pilot"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _new_run(root: Path, environment: Path, run_root: Path, extra: list[str]) -> Path:
    before = {path.resolve() for path in run_root.glob(f"{STAGE}_*") if path.is_dir()}
    command = [
        sys.executable,
        str(root / "scripts/run_pipeline.py"),
        STAGE,
        "--environment",
        str(environment),
        *extra,
    ]
    result = subprocess.run(command, cwd=root, check=False)
    created = sorted(
        path.resolve()
        for path in run_root.glob(f"{STAGE}_*")
        if path.is_dir() and path.resolve() not in before
    )
    _require(len(created) == 1, f"Expected one new {STAGE} run; found {created}")
    run = created[0]
    _require(result.returncode == 0, f"Pilot command failed ({result.returncode}); inspect {run}")
    print(f"Verified new run path: {run}", flush=True)
    return run


def _execution(run: Path, status: str) -> dict:
    execution = _read_object(run / "execution.json")
    _require(execution.get("stage") == STAGE, f"Wrong stage in {run}")
    _require(execution.get("status") == status, f"Expected {status} execution in {run}")
    _require(execution.get("returncode") == 0, f"Nonzero return code in {run}")
    return execution


def _adapter(path: Path) -> Path:
    _require(path.is_dir(), f"Missing adapter directory: {path}")
    config = path / "adapter_config.json"
    weights = [path / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    _require(config.is_file() and config.stat().st_size > 0, f"Missing adapter config: {path}")
    _require(
        any(file.is_file() and file.stat().st_size > 0 for file in weights),
        f"Missing adapter weights: {path}",
    )
    return path.resolve()


def _lora_b_payloads(checkpoint: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Read saved LoRA-B tensor bytes, not the safetensors file's variable header."""
    path = checkpoint / "adapter_model.safetensors"
    _require(path.is_file(), f"Rehearsal requires safetensors adapter weights: {path}")
    content = path.read_bytes()
    _require(len(content) >= 8, f"Invalid safetensors adapter: {path}")
    header_size = int.from_bytes(content[:8], "little")
    _require(0 < header_size <= 16 * 1024 * 1024, f"Invalid safetensors header: {path}")
    data_start = 8 + header_size
    _require(data_start <= len(content), f"Truncated safetensors header: {path}")
    header = json.loads(content[8:data_start])
    _require(isinstance(header, dict), f"Invalid safetensors metadata: {path}")
    tensors = {}
    for name, entry in header.items():
        if "lora_B" not in name:
            continue
        _require(isinstance(entry, dict), f"Invalid LoRA-B tensor metadata: {name}")
        offsets = entry.get("data_offsets")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(offset) is int for offset in offsets)
            and 0 <= offsets[0] < offsets[1] <= len(content) - data_start
            and isinstance(shape, list)
            and all(type(dimension) is int and dimension > 0 for dimension in shape)
            and isinstance(dtype, str)
            and dtype,
            f"Invalid LoRA-B tensor metadata: {name}",
        )
        tensors[name] = (
            dtype,
            tuple(shape),
            content[data_start + offsets[0] : data_start + offsets[1]],
        )
    _require(tensors, f"No LoRA-B tensors in checkpoint: {path}")
    return tensors


def _lora_values(dtype: str, payload: bytes):
    """Decode the floating-point dtypes PEFT saves, without importing Torch."""
    import numpy as np

    if dtype == "BF16":
        half_words = np.frombuffer(payload, dtype="<u2")
        return (half_words.astype("<u4") << 16).view("<f4")
    numpy_dtype = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}.get(dtype)
    _require(numpy_dtype is not None, f"Unsupported LoRA-B tensor dtype: {dtype}")
    return np.frombuffer(payload, dtype=numpy_dtype)


def _changed_lora_b_tensors(first: Path, second: Path) -> tuple[int, int, float]:
    import numpy as np

    previous = _lora_b_payloads(first)
    resumed = _lora_b_payloads(second)
    _require(previous.keys() == resumed.keys(), "LoRA-B tensor names changed across resume")
    max_delta = 0.0
    changed = 0
    for name in previous:
        _require(
            previous[name][:2] == resumed[name][:2],
            f"LoRA-B tensor shape or dtype changed across resume: {name}",
        )
        dtype, shape, old_bytes = previous[name]
        new_bytes = resumed[name][2]
        old_values = _lora_values(dtype, old_bytes)
        new_values = _lora_values(dtype, new_bytes)
        _require(
            old_values.size == new_values.size == math.prod(shape)
            and np.isfinite(old_values).all()
            and np.isfinite(new_values).all(),
            f"Non-finite or incorrectly sized LoRA-B tensor: {name}",
        )
        with np.errstate(over="ignore"):
            delta = float(
                np.max(np.abs(new_values.astype(np.float64) - old_values.astype(np.float64)))
            )
        _require(math.isfinite(delta), f"Non-finite LoRA-B weight change: {name}")
        if delta > 0:
            changed += 1
            max_delta = max(max_delta, delta)
    _require(changed > 0, "No saved LoRA-B tensor changed after resumed optimizer steps")
    return changed, len(previous), max_delta


def _finite_metric(payload: dict, section: str, key: str) -> None:
    metrics = payload.get(section)
    value = metrics.get(key) if isinstance(metrics, dict) else None
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"Missing or non-finite {section}.{key}",
    )


def rehearse(environment: Path, root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    environment = (environment if environment.is_absolute() else root / environment).resolve()
    run_root = load_environment(environment, root)["run_root"]

    first = _new_run(root, environment, run_root, ["--stop-after-step", "10"])
    first_execution = _execution(first, "paused")
    session = _read_object(first / "training/session_result.json")
    checkpoint_10 = (first / "training/checkpoint-10").resolve()
    _require(session.get("status") == "paused", "First pilot did not pause")
    _require(session.get("global_step") == 10, "First pilot did not reach step 10")
    _require(session.get("target_global_step") == 20, "Pilot target changed during pause")
    _require(session.get("optimizer_steps_this_run") == 10, "First pilot optimizer steps differ")
    _require(
        Path(session.get("checkpoint", "")).resolve() == checkpoint_10,
        "Wrong pause checkpoint",
    )
    _require(
        isinstance(session.get("max_lora_weight_change"), (int, float))
        and math.isfinite(session["max_lora_weight_change"])
        and session["max_lora_weight_change"] > 0,
        "First pilot has no measured LoRA update",
    )
    _finite_metric(session, "train_metrics", "train_loss")
    validate_checkpoint(checkpoint_10)

    second = _new_run(root, environment, run_root, ["--resume-from-checkpoint", str(checkpoint_10)])
    second_execution = _execution(second, "passed")
    _require(first != second, "Resume reused the paused run directory")
    _require(
        Path(second_execution.get("resume_from_checkpoint", "")).resolve() == checkpoint_10,
        "Second run did not record the source checkpoint",
    )
    first_provenance = first_execution.get("provenance", {})
    second_provenance = second_execution.get("provenance", {})
    _require(
        isinstance(first_provenance, dict)
        and first_provenance.get("source_sha256")
        and first_provenance == second_provenance,
        "Pilot provenance changed between sessions",
    )
    result = _read_object(second / "training/training_result.json")
    for key, expected in (
        ("status", "passed"),
        ("experiment_id", "M1_SFT_PILOT"),
        ("global_step", 20),
        ("target_global_step", 20),
        ("resumed_from_global_step", 10),
        ("optimizer_steps_this_run", 10),
    ):
        _require(result.get(key) == expected, f"Unexpected {key} in resumed pilot")
    _finite_metric(result, "train_metrics", "train_loss")
    _finite_metric(result, "final_validation", "eval_loss")
    checkpoint_20 = validate_checkpoint(second / "training/checkpoint-20")
    changed_lora_b, lora_b_count, max_lora_b_delta = _changed_lora_b_tensors(
        checkpoint_10, checkpoint_20
    )
    adapter = _adapter(second / "training/adapter")
    _require(
        Path(result.get("adapter", "")).resolve() == adapter,
        "Resumed pilot reported a different adapter",
    )

    report_path = second / "resume_rehearsal.json"
    report = {
        "status": "passed",
        "experiment_id": "M1_SFT_PILOT",
        "paused_run": str(first),
        "completed_run": str(second),
        "checkpoint_10": str(checkpoint_10),
        "checkpoint_20": str(checkpoint_20),
        "resumed_optimizer_steps": 10,
        "changed_lora_b_tensors": changed_lora_b,
        "lora_b_tensor_count": lora_b_count,
        "max_lora_b_weight_change": max_lora_b_delta,
        "adapter": str(adapter),
        "source_sha256": first_provenance["source_sha256"],
    }
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"M1 resume rehearsal passed: {report_path}", flush=True)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    try:
        rehearse(args.environment)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.exit(1, f"M1 resume rehearsal failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
