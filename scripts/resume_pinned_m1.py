"""Resume an immutable M1 release into a writable output root."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import sys
import threading
import uuid
from collections import Counter
from pathlib import Path

BASE_BUDGET = "vqa_nle_budget_v2"
EXTENDED_BUDGET = "vqa_nle_budget_v2_m1_1200"
EPOCH_BUDGET = "vqa_nle_m1_epoch1_v1"
EPOCH_STEPS = 3683
BUDGET_CEILINGS = {BASE_BUDGET: 1000, EXTENDED_BUDGET: 1200, EPOCH_BUDGET: EPOCH_STEPS}


def verify_source(root, fingerprints):
    for name, expected in fingerprints.items():
        file = (root / name).resolve(strict=True)
        if not file.is_relative_to(root):
            raise ValueError(f"Source path escapes release: {name}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source changed: {name}")


def validate_stop(state, stop, *, budget_protocol=BASE_BUDGET):
    """Return a session pause, or None for the original one-epoch finish."""
    ceiling = BUDGET_CEILINGS.get(budget_protocol)
    if ceiling is None:
        raise ValueError("Unknown M1 budget protocol")
    if budget_protocol == EXTENDED_BUDGET and state["global_step"] < 1000:
        raise ValueError("The 1200-step extension must resume from step 1000 or later")
    if budget_protocol == EPOCH_BUDGET:
        if (state["global_step"] < 1200 or state["max_steps"] != EPOCH_STEPS
                or type(state.get("num_train_epochs")) is not int
                or state["num_train_epochs"] != 1
                or state.get("save_steps") != 100 or state.get("eval_steps") != 100):
            raise ValueError(
                "Epoch extension requires step >=1200 and the original one-epoch schedule"
            )
        if type(stop) is int and stop == EPOCH_STEPS and state["global_step"] < stop:
            return None
    if (type(stop) is not int or stop > ceiling
            or not state["global_step"] < stop < state["max_steps"]
            or stop % state["save_steps"]):
        raise ValueError("Stop must be a later checkpoint-aligned step before the full run ends")
    return stop


def training_command(root, paths, run, checkpoint, session_stop):
    command = [
        sys.executable, "-u", str(root / "scripts/train_sft_full.py"),
        "--annotations", str(paths["annotations"]), "--image-root", str(paths["image_root"]),
        "--config", str(root / "configs/training/sft.yaml"),
        "--model-config", str(root / "configs/model/qwen25_vl_3b.yaml"),
        "--image-checksums", str(paths["image_checksums"]),
        "--output-dir", str(run / "training"), "--resume-from-checkpoint", str(checkpoint),
    ]
    if session_stop is not None:
        command += ["--stop-after-step", str(session_stop)]
    return command


def verify_epoch_completion(run, target):
    """Distinguish the last optimizer checkpoint from the validation-selected adapter."""
    import torch
    from safetensors.torch import load_file

    training = run / "training"
    result = json.loads((training / "training_result.json").read_text("utf-8"))
    state = json.loads((training / "trainer_state.json").read_text("utf-8"))
    if (target != EPOCH_STEPS or result.get("status") != "passed"
            or result.get("global_step") != target or result.get("target_global_step") != target
            or state.get("global_step") != target or state.get("max_steps") != target
            or not math.isclose(state.get("epoch", -1), 1.0, abs_tol=1e-9)):
        raise ValueError("M1 did not complete the original one-epoch schedule")
    best_path = result.get("best_model_checkpoint")
    if not isinstance(best_path, str) or state.get("best_model_checkpoint") != best_path:
        raise ValueError("Epoch result and Trainer disagree on the selected checkpoint")
    best_metric = result.get("best_eval_loss")
    loss = result.get("final_validation", {}).get("eval_loss")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (best_metric, loss)):
        raise ValueError("Epoch result has no finite validation losses")
    if not math.isclose(best_metric, state.get("best_metric", -1), abs_tol=1e-9):
        raise ValueError("Epoch result and Trainer disagree on the best loss")
    best = Path(best_path).resolve(strict=True)
    if best.parent.name != "training" or not best.name.startswith("checkpoint-"):
        raise ValueError("Selected model is not a numbered M1 checkpoint")
    best_state = json.loads((best / "trainer_state.json").read_text("utf-8"))
    selected_step = best_state["global_step"]
    if (type(selected_step) is not int or not 0 < selected_step <= target
            or best.name != f"checkpoint-{selected_step}"):
        raise ValueError("Selected checkpoint has an invalid optimizer step")
    selected_file = training / "adapter/adapter_model.safetensors"
    best_file = best / "adapter_model.safetensors"
    selected = load_file(str(selected_file), device="cpu")
    expected = load_file(str(best_file), device="cpu")
    if not selected or selected.keys() != expected.keys() or any(
        not torch.equal(weight, expected[name]) or not bool(torch.isfinite(weight).all())
        for name, weight in selected.items()
    ):
        raise ValueError("Saved adapter differs from the validation-selected checkpoint")
    return {"epoch": state["epoch"], "final_optimizer_step": target,
            "selected_checkpoint": str(best), "selected_step": selected_step,
            "best_eval_loss": best_metric, "selected_eval_loss": loss,
            "selected_adapter_sha256": hashlib.sha256(selected_file.read_bytes()).hexdigest(),
            "selected_checkpoint_sha256": hashlib.sha256(best_file.read_bytes()).hexdigest()}


def resume_run_root(previous, checkpoint, configured_root):
    """Accept a relocated checkpoint only under its recorded output root."""
    checkpoint = checkpoint.resolve()
    configured_root = configured_root.resolve()
    if checkpoint.is_relative_to(configured_root):
        return configured_root
    relocation = previous.get("output_relocation")
    if not isinstance(relocation, dict):
        raise ValueError("Checkpoint outside configured run_root has no relocation record")
    output = relocation.get("output_root")
    original = relocation.get("original_run_root")
    paths = previous.get("paths")
    if (not isinstance(output, str) or not Path(output).is_absolute()
            or not isinstance(original, str) or not Path(original).is_absolute()
            or not isinstance(paths, dict) or paths.get("run_root") != original):
        raise ValueError("Checkpoint has an inconsistent relocation record")
    recorded_root = Path(output).resolve()
    if checkpoint.parent.parent.parent != recorded_root:
        raise ValueError("Checkpoint does not belong to its recorded output root")
    return recorded_root


def verify_no_followup(checkpoint, output_root):
    """A repeated launch must not fork the same checkpoint accidentally."""
    for metadata_path in output_root.glob("sft-full_*/execution.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        resumed = metadata.get("resume_from_checkpoint")
        if isinstance(resumed, str) and Path(resumed).resolve() == checkpoint:
            raise ValueError(f"A follow-up run already exists; inspect it: {metadata_path.parent}")


def verify_checkpoint_payloads(checkpoint):
    """Read M1 state on CPU without changing weights, RNG or checkpoint files."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name[11:].isdigit():
        raise ValueError("Expected a numbered M1 checkpoint")
    step = int(checkpoint.name[11:])
    metadata_path = checkpoint.parent / "run_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    dtype = metadata.get("compute_dtype")
    if dtype not in {"torch.float16", "torch.bfloat16"}:
        raise ValueError("Checkpoint has no supported M1 compute dtype")
    names = ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
             "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]
    if dtype == "torch.float16":
        names.append("scaler.pt")

    def fingerprints():
        hashes = {}
        for name in names:
            file = checkpoint / name
            if file.is_symlink() or not file.is_file() or not file.stat().st_size:
                raise ValueError(f"Missing or unsafe checkpoint file: {name}")
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
        return hashes

    def finite(value):
        if torch.is_tensor(value):
            if not value.numel() or not bool(torch.isfinite(value).all()):
                raise ValueError("Checkpoint contains empty or non-finite tensors")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Checkpoint contains a non-finite number")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)

    before = fingerprints()
    state = json.loads((checkpoint / "trainer_state.json").read_text("utf-8"))
    finite(state)
    if type(state.get("global_step")) is not int or state["global_step"] != step or step <= 0:
        raise ValueError("Trainer step differs from checkpoint name")
    config = json.loads((checkpoint / "adapter_config.json").read_text("utf-8"))
    rank = config.get("r")
    if (config.get("peft_type") != "LORA" or config.get("bias") != "none"
            or type(rank) is not int or rank <= 0 or config.get("rank_pattern")):
        raise ValueError("Expected the pinned bias-free LoRA adapter")
    weights = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    finite(weights)
    if not weights or any(not name.endswith((".lora_A.weight", ".lora_B.weight"))
                          or value.ndim != 2 for name, value in weights.items()):
        raise ValueError("Unexpected M1 adapter tensors")
    for name, value in weights.items():
        peer = name.replace(".lora_A.weight", ".lora_B.weight") if ".lora_A." in name else (
            name.replace(".lora_B.weight", ".lora_A.weight"))
        tensor_rank = value.shape[0] if ".lora_A." in name else value.shape[1]
        if tensor_rank != rank or peer not in weights or tensor_rank != (
            weights[peer].shape[1] if ".lora_A." in name else weights[peer].shape[0]
        ):
            raise ValueError("LoRA A/B rank mismatch")

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    finite(optimizer)
    finite(scheduler)
    groups, moments = optimizer.get("param_groups"), optimizer.get("state")
    if not isinstance(groups, list) or not groups or not isinstance(moments, dict) or not moments:
        raise ValueError("Missing AdamW optimizer state")
    ids = [index for group in groups for index in group["params"]]
    if (any(type(index) is not int for index in ids) or len(ids) != len(set(ids))
            or set(ids) != set(moments) or len(ids) != len(weights)):
        raise ValueError("Optimizer parameters differ from the adapter")
    shapes = []
    for moment in moments.values():
        moment_step = moment.get("step")
        if torch.is_tensor(moment_step):
            moment_step = moment_step.item()
        if isinstance(moment_step, bool) or moment_step != step:
            raise ValueError("Optimizer step differs from trainer state")
        first, second = moment.get("exp_avg"), moment.get("exp_avg_sq")
        if not torch.is_tensor(first) or not torch.is_tensor(second) or first.shape != second.shape:
            raise ValueError("Invalid AdamW moment tensors")
        if bool((second < 0).any()):
            raise ValueError("AdamW second moments must be nonnegative")
        shapes.append(tuple(first.shape))
    if Counter(shapes) != Counter(tuple(weight.shape) for weight in weights.values()):
        raise ValueError("Optimizer moment shapes differ from the adapter")
    rates = scheduler.get("_last_lr")
    if (scheduler.get("last_epoch") != step or not isinstance(rates, list)
            or len(rates) != len(groups)):
        raise ValueError("Scheduler step or groups differ from optimizer state")
    if any(rate < 0 or not math.isclose(rate, group["lr"], rel_tol=1e-9, abs_tol=1e-12)
           for rate, group in zip(rates, groups, strict=True)):
        raise ValueError("Scheduler learning rates differ from optimizer state")

    # Permit only NumPy's MT19937 payload, never unrestricted pickle loading.
    numpy_core = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"
    reconstruct = __import__(numpy_core + ".multiarray", fromlist=["_reconstruct"])._reconstruct
    allowed = [(reconstruct, "numpy.core.multiarray._reconstruct"),
               (reconstruct, "numpy._core.multiarray._reconstruct"),
               np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):
        rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=True)
    finite(rng)
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator(device="cpu").set_state(rng["cpu"])
    cuda = rng.get("cuda")
    # Non-distributed Trainer saves a tensor; distributed Trainer saves a list.
    cuda_states = [cuda] if torch.is_tensor(cuda) else cuda
    if (not isinstance(cuda_states, list) or len(cuda_states) != 1
            or not torch.is_tensor(cuda_states[0]) or cuda_states[0].dtype != torch.uint8
            or cuda_states[0].ndim != 1 or not cuda_states[0].numel()):
        raise ValueError("Expected the saved single-GPU RNG state")
    if dtype == "torch.float16":
        scaler = torch.load(checkpoint / "scaler.pt", map_location="cpu", weights_only=True)
        finite(scaler)
        if not isinstance(scaler, dict) or scaler.get("scale", 0) <= 0:
            raise ValueError("Invalid FP16 scaler state")
    if before != fingerprints() or metadata_bytes != metadata_path.read_bytes():
        raise ValueError("Checkpoint changed while being inspected")
    return {"global_step": step, "compute_dtype": dtype, "adapter_tensors": len(weights),
            "adapter_parameters": sum(weight.numel() for weight in weights.values()),
            "optimizer_parameters": len(ids), "scheduler_step": scheduler["last_epoch"],
            "file_sha256": before,
            "run_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "cuda_rng_format": "tensor" if torch.is_tensor(cuda) else "single_tensor_list",
            "training_args_policy": "hashed_without_unpickling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--budget-protocol", choices=tuple(BUDGET_CEILINGS), default=BASE_BUDGET)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true")
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    previous = json.loads(
        (checkpoint.parent.parent / "execution.json").read_text(encoding="utf-8")
    )

    # Check the release before importing any of its code.
    verify_source(root, previous["provenance"]["source_files"])
    sys.path.insert(0, str(root / "src"))
    spec = importlib.util.spec_from_file_location(
        "pinned_pipeline", root / "scripts/run_pipeline.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    environment = args.environment.resolve(strict=True)
    paths = pipeline.load_environment(environment)
    training = root / "configs/training/sft.yaml"
    model = root / "configs/model/qwen25_vl_3b.yaml"
    provenance = pipeline.collect_provenance(
        root, environment, paths["annotations"], training=training,
        model=model, image_checksums=paths["image_checksums"],
    )
    checkpoint_root = resume_run_root(previous, checkpoint, paths["run_root"])
    checkpoint = pipeline.verify_resume(
        checkpoint, checkpoint_root, provenance, recover_interrupted=args.recover_interrupted,
    )
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    stop = args.stop_after_step
    session_stop = validate_stop(state, stop, budget_protocol=args.budget_protocol)
    output_root = args.output_root.resolve()
    if output_root == paths["run_root"].resolve():
        raise ValueError("Use the normal pipeline when the original output root is writable")
    verify_no_followup(checkpoint, output_root)
    integrity = verify_checkpoint_payloads(checkpoint)
    if args.check_only:
        print(f"Resume checks passed: step {state['global_step']} -> {stop}")
        print(f"Budget: {args.budget_protocol}; ceiling: {BUDGET_CEILINGS[args.budget_protocol]}")
        print(json.dumps(integrity, sort_keys=True))
        return
    run = output_root / ("sft-full_" + pipeline.datetime.now(pipeline.timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    command = training_command(root, paths, run, checkpoint, session_stop)
    metadata = {
        "stage": "sft-full", "started_at_utc": pipeline.utc_now(),
        "paths": {k: str(v) for k, v in paths.items()}, "command": command,
        "provenance": provenance, "resume_from_checkpoint": str(checkpoint),
        "output_relocation": {"original_run_root": str(paths["run_root"]),
                              "output_root": str(output_root),
                              "checkpoint_run_root": str(checkpoint_root),
                              "reason": "Keep new outputs separate from the recorded run root"},
        "budget": {"protocol_id": args.budget_protocol,
                   "maximum_step": BUDGET_CEILINGS[args.budget_protocol],
                   "start_step": state["global_step"], "stop_after_step": stop,
                   "planned_updates": stop - state["global_step"],
                   "completion_mode": "full_epoch" if session_stop is None else "session_pause"},
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recover_interrupted": args.recover_interrupted,
        "resume_checkpoint_integrity": integrity,
        "runner": {"id": uuid.uuid4().hex, "host": socket.gethostname(), "pid": os.getpid()},
        "status": "running",
    }
    metadata_path = run / "execution.json"
    pipeline.write_metadata(metadata_path, metadata)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=pipeline.heartbeat_loop,
        args=(run / "runner_heartbeat.json", metadata["runner"]["id"], heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
    print(f"Verified resume: {checkpoint}\nOutput: {run}\nStop: {stop}", flush=True)
    result = 1
    try:
        with (run / "console.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                metadata["runner"]["child_pid"] = process.pid
                pipeline.write_metadata(metadata_path, metadata)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                result = process.wait()
        metadata["status"] = (
            pipeline.verify_training_artifacts(run, session_stop) if result == 0 else "failed"
        )
        if result == 0:
            metadata["output_checkpoint_integrity"] = verify_checkpoint_payloads(
                run / "training" / f"checkpoint-{stop}",
            )
            if session_stop is None:
                metadata["epoch_completion"] = verify_epoch_completion(run, stop)
        metadata["returncode"] = result
    except BaseException as exc:
        metadata.update(status="interrupted_or_failed", error_type=type(exc).__name__)
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join()
        metadata["finished_at_utc"] = pipeline.utc_now()
        pipeline.write_metadata(metadata_path, metadata)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
