"""Local/cloud audit, preflight, SFT readiness, smoke and full M1 entry point."""

import argparse
import ctypes
import errno
import json
import math
import os
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vqa_explanation.training.sft_full import validate_checkpoint
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.run_provenance import collect_provenance, utc_now

HEARTBEAT_INTERVAL_SECONDS = 30
INTERRUPTED_RUN_GRACE = timedelta(minutes=10)


def write_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_alive(pid: int) -> bool | None:
    """Return None when process liveness cannot be established safely."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            if result == 0x00000102:  # WAIT_TIMEOUT
                return True
            return False if result == 0 else None  # WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return False if error.errno == errno.ESRCH else None
    return True


def verify_interrupted_run(previous_run: Path, previous: dict) -> None:
    """Require opt-in, a stale heartbeat and no live local process for recovery."""
    runner = previous.get("runner")
    if not isinstance(runner, dict) or not isinstance(runner.get("id"), str) or not runner["id"]:
        raise ValueError("Running run lacks runner identity; cannot prove interruption")
    if not isinstance(runner.get("host"), str) or not runner["host"]:
        raise ValueError("Running run lacks host identity; cannot prove interruption")
    if any(type(runner.get(key)) is not int or runner[key] <= 0 for key in ("pid", "child_pid")):
        raise ValueError("Running run lacks valid runner and child PIDs")
    if runner["host"] == socket.gethostname():
        for key in ("pid", "child_pid"):
            alive = process_alive(runner[key])
            if alive is not False:
                raise ValueError(f"Cannot recover running run: {key} is live or unverified")
    heartbeat_path = previous_run / "runner_heartbeat.json"
    if not heartbeat_path.is_file():
        raise ValueError("Running run lacks a heartbeat; cannot prove interruption")
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    if not isinstance(heartbeat, dict) or heartbeat.get("runner_id") != runner["id"]:
        raise ValueError("Running run heartbeat does not match runner identity")
    try:
        last_seen = datetime.fromisoformat(heartbeat["updated_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Running run has an invalid heartbeat timestamp") from error
    if last_seen.tzinfo is None:
        raise ValueError("Running run heartbeat must use a timezone-aware timestamp")
    age = datetime.now(timezone.utc) - last_seen
    if age < INTERRUPTED_RUN_GRACE:
        raise ValueError("Running run heartbeat is recent; cannot recover an active run")


def heartbeat_loop(path: Path, runner_id: str, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
        write_metadata(path, {"runner_id": runner_id, "updated_at_utc": utc_now()})


def verify_resume(
    checkpoint: Path,
    run_root: Path,
    provenance: dict,
    stage: str = "sft-full",
    *,
    recover_interrupted: bool = False,
) -> Path:
    """Resume into a new run only from an incomplete, provenance-matched M1 run."""
    checkpoint = checkpoint.resolve()
    run_root = run_root.resolve()
    if not checkpoint.is_relative_to(run_root) or checkpoint.parent.name != "training":
        raise ValueError("Resume checkpoint must belong to an M1 run under run_root")
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name[11:].isdigit():
        raise ValueError("Resume source must be a numbered Trainer checkpoint")
    previous_run = checkpoint.parent.parent
    metadata_path = previous_run / "execution.json"
    if not metadata_path.is_file():
        raise ValueError("Previous run has no execution.json")
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(previous, dict):
        raise ValueError("Previous execution.json must be a JSON object")
    if previous.get("stage") != stage:
        raise ValueError(f"Only a stopped, incomplete {stage} run can be resumed")
    if previous.get("status") == "running" and recover_interrupted:
        verify_interrupted_run(previous_run, previous)
    elif previous.get("status") not in {"paused", "failed", "interrupted_or_failed"}:
        raise ValueError(f"Only a stopped, incomplete {stage} run can be resumed")
    old = previous.get("provenance")
    if not isinstance(old, dict) or not isinstance(old.get("input_sha256"), dict):
        raise ValueError("Previous run lacks valid provenance")
    if old.get("source_sha256") != provenance["source_sha256"]:
        raise ValueError("Source code changed since the checkpoint; start a new experiment")
    for key in ("environment", "annotations", "training_config", "model_config", "image_checksums"):
        old_hash = old["input_sha256"].get(key)
        new_hash = provenance["input_sha256"].get(key)
        if not isinstance(old_hash, str) or not old_hash or old_hash != new_hash:
            raise ValueError(f"Resume input changed: {key}")
    return validate_checkpoint(checkpoint)


def verify_training_artifacts(run: Path, stop_after_step: int | None) -> str:
    """Validate saved M1 results after the child process exits."""
    training = run / "training"
    if stop_after_step is not None:
        report_path = training / "session_result.json"
        if not report_path.is_file():
            raise ValueError("M1 pause completed without session_result.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("status") != "paused":
            raise ValueError("M1 pause did not report status=paused")
        if type(report.get("global_step")) is not int or report["global_step"] != stop_after_step:
            raise ValueError("M1 pause report has the wrong optimizer step")
        checkpoint = validate_checkpoint(training / f"checkpoint-{stop_after_step}")
        if not isinstance(report.get("checkpoint"), str) or (
            Path(report["checkpoint"]).resolve() != checkpoint
        ):
            raise ValueError("M1 pause report points to a different checkpoint")
        return "paused"

    report_path = training / "training_result.json"
    if not report_path.is_file():
        raise ValueError("M1 training completed without training_result.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError("M1 training did not report status=passed")
    step = report.get("global_step")
    target = report.get("target_global_step")
    if type(step) is not int or type(target) is not int or step <= 0 or step != target:
        raise ValueError("M1 training report has an incomplete optimizer step count")
    adapter = training / "adapter"
    if not isinstance(report.get("adapter"), str) or Path(report["adapter"]).resolve() != (
        adapter.resolve()
    ):
        raise ValueError("M1 training report points to a different adapter")
    adapter_config = adapter / "adapter_config.json"
    if not adapter_config.is_file() or adapter_config.stat().st_size == 0 or not any(
        (adapter / name).is_file() and (adapter / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError("M1 training completed without saved adapter weights")
    return "passed"


def verify_smoke_artifacts(run: Path) -> None:
    """Require saved training evidence before accepting a smoke run."""
    training = run / "training"
    report_path = training / "smoke_result.json"
    if not report_path.is_file():
        raise ValueError("SFT smoke completed without smoke_result.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError("SFT smoke did not report status=passed")
    step = report.get("global_step")
    target = report.get("target_global_step")
    if type(step) is not int or type(target) is not int or step <= 0 or step != target:
        raise ValueError("SFT smoke report has an incomplete optimizer step count")
    delta = report.get("max_lora_weight_change")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(delta)
        or delta <= 0
    ):
        raise ValueError("SFT smoke report has no LoRA weight update")
    adapter = training / "adapter"
    if not isinstance(report.get("adapter"), str) or Path(report["adapter"]).resolve() != (
        adapter.resolve()
    ):
        raise ValueError("SFT smoke report points to a different adapter")
    adapter_config = adapter / "adapter_config.json"
    if not adapter_config.is_file() or adapter_config.stat().st_size == 0 or not any(
        (adapter / name).is_file() and (adapter / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError("SFT smoke completed without saved adapter weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "preflight",
            "token-profile",
            "sft-batch-pilot",
            "sft-smoke",
            "sft-full-preflight",
            "sft-optimizer-pilot",
            "sft-full",
        ],
    )
    parser.add_argument("--environment", type=Path, default=Path("configs/environments/local.yaml"))
    parser.add_argument(
        "--all-train-records",
        action="store_true",
        help="Preflight every train question, not just the smoke subset",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--recover-interrupted-run",
        action="store_true",
        help=(
            "Allow resume from a stale 'running' run only after a 10-minute heartbeat gap "
            "and process-liveness checks; use only after confirming its Colab runtime stopped"
        ),
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Pause M1 at a checkpoint-aligned optimizer step for a later resume",
    )
    args = parser.parse_args()
    if args.all_train_records and args.stage != "preflight":
        parser.error("--all-train-records is only valid for preflight")
    train_stages = ("sft-optimizer-pilot", "sft-full")
    if args.resume_from_checkpoint and args.stage not in train_stages:
        parser.error("--resume-from-checkpoint is only valid for M1 training stages")
    if args.recover_interrupted_run and not args.resume_from_checkpoint:
        parser.error("--recover-interrupted-run requires --resume-from-checkpoint")
    if args.stop_after_step is not None and args.stage not in train_stages:
        parser.error("--stop-after-step is only valid for M1 training stages")
    root = project_root()
    environment = args.environment
    if not environment.is_absolute():
        environment = root / environment
    paths = load_environment(args.environment)
    run = paths["run_root"] / (
        args.stage + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    common = ["--annotations", str(paths["annotations"]), "--image-root", str(paths["image_root"])]
    if args.stage == "audit":
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/audit_dataset.py"),
            "--input",
            str(paths["annotations"]),
            "--image-root",
            str(paths["image_root"]),
            "--verify-images",
            "--strict",
            "--output",
            str(run / "audit.json"),
        ]
    elif args.stage == "token-profile":
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/profile_sft_tokens.py"),
            *common,
            "--config",
            str(root / "configs/training/sft_readiness.yaml"),
            "--model-config",
            str(root / "configs/model/qwen25_vl_3b.yaml"),
            "--output",
            str(run / "token_profile.json"),
        ]
    elif args.stage == "sft-batch-pilot":
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/pilot_sft_batch.py"),
            *common,
            "--config",
            str(root / "configs/training/sft_batch_pilot.yaml"),
            "--model-config",
            str(root / "configs/model/qwen25_vl_3b.yaml"),
            "--output",
            str(run / "batch_pilot.json"),
        ]
    elif args.stage in ("sft-full-preflight", "sft-optimizer-pilot", "sft-full"):
        full_config = (
            "sft_optimizer_pilot.yaml" if args.stage == "sft-optimizer-pilot" else "sft.yaml"
        )
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/train_sft_full.py"),
            *common,
            "--config",
            str(root / "configs/training" / full_config),
            "--model-config",
            str(root / "configs/model/qwen25_vl_3b.yaml"),
            "--output-dir",
            str(run / "training"),
        ]
        if "image_checksums" in paths:
            command.extend(["--image-checksums", str(paths["image_checksums"])])
        if args.stage == "sft-full-preflight":
            command.append("--preflight-only")
        if args.resume_from_checkpoint:
            command.extend(["--resume-from-checkpoint", str(args.resume_from_checkpoint.resolve())])
        if args.stop_after_step is not None:
            command.extend(["--stop-after-step", str(args.stop_after_step)])
    else:
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/train_sft.py"),
            *common,
            "--config",
            str(root / "configs/training/sft_smoke.yaml"),
            "--model-config",
            str(root / "configs/model/qwen25_vl_3b.yaml"),
            "--output-dir",
            str(run / "training"),
        ]
        if args.stage == "preflight":
            command.append("--preflight-only")
            if args.all_train_records:
                command.append("--all-train-records")
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    if not paths["annotations"].is_file() or not paths["image_root"].is_dir():
        parser.error("Dataset missing: prepare/download data first; see docs/local_cloud.md")
    if "image_checksums" in paths and not paths["image_checksums"].is_file():
        parser.error("Image checksum manifest missing; verify the dataset bundle first")
    if args.stage in ("sft-full-preflight", "sft-optimizer-pilot", "sft-full"):
        if "image_checksums" not in paths:
            parser.error("Full M1 stages require image_checksums in the environment profile")
    config_for_stage = {
        "token-profile": "sft_readiness.yaml",
        "sft-batch-pilot": "sft_batch_pilot.yaml",
        "sft-full-preflight": "sft.yaml",
        "sft-optimizer-pilot": "sft_optimizer_pilot.yaml",
        "sft-full": "sft.yaml",
    }
    training_config = root / "configs/training" / config_for_stage.get(args.stage, "sft_smoke.yaml")
    model_config = root / "configs/model/qwen25_vl_3b.yaml"
    provenance = collect_provenance(
        root,
        environment,
        paths["annotations"],
        training=training_config if args.stage != "audit" else None,
        model=model_config if args.stage != "audit" else None,
        image_checksums=paths.get("image_checksums"),
    )
    checkpoint = None
    if args.resume_from_checkpoint:
        try:
            checkpoint = verify_resume(
                args.resume_from_checkpoint,
                paths["run_root"],
                provenance,
                args.stage,
                recover_interrupted=args.recover_interrupted_run,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    run.mkdir(parents=True, exist_ok=False)
    metadata = {
        "stage": args.stage,
        "started_at_utc": utc_now(),
        "paths": {k: str(v) for k, v in paths.items()},
        "command": command,
        "provenance": provenance,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "runner": {"id": uuid.uuid4().hex, "host": socket.gethostname(), "pid": os.getpid()},
        "status": "running",
    }
    metadata_path = run / "execution.json"
    write_metadata(metadata_path, metadata)
    heartbeat_path = run / "runner_heartbeat.json"
    write_metadata(
        heartbeat_path, {"runner_id": metadata["runner"]["id"], "updated_at_utc": utc_now()}
    )
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(heartbeat_path, metadata["runner"]["id"], heartbeat_stop),
        daemon=True,
    )
    heartbeat_thread.start()
    print(f"Run: {run}", flush=True)
    try:
        with (run / "console.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                metadata["runner"]["child_pid"] = process.pid
                write_metadata(metadata_path, metadata)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                result = process.wait()
        status = "passed" if result == 0 else "failed"
        if result == 0 and args.stage in train_stages:
            try:
                status = verify_training_artifacts(run, args.stop_after_step)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                metadata["artifact_validation_error"] = str(error)
                status = "failed"
                result = 1
        elif result == 0 and args.stage == "sft-smoke":
            try:
                verify_smoke_artifacts(run)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                metadata["artifact_validation_error"] = str(error)
                status = "failed"
                result = 1
        metadata.update(status=status, returncode=result)
    except BaseException as exc:
        metadata["status"] = "interrupted_or_failed"
        metadata["error_type"] = type(exc).__name__
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
        metadata["finished_at_utc"] = utc_now()
        write_metadata(metadata_path, metadata)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
