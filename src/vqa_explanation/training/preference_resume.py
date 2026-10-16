"""Verified checkpoints for interrupted preference stages."""

import json
from pathlib import Path

from vqa_explanation.data.io import write_json
from vqa_explanation.utils.run_provenance import sha256_file


def checkpoint_files(path, *, fp16):
    required = {
        "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
        "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
    }
    if fp16:
        required.add("scaler.pt")
    if any(not (path / name).is_file() or not (path / name).stat().st_size for name in required):
        raise ValueError("Checkpoint needs adapter, optimizer, scheduler, RNG and scaler state")
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("Checkpoint must not contain symlinks")
        if file.is_file() and file.name != "resume_manifest.json":
            files[file.relative_to(path).as_posix()] = sha256_file(file)
    return files


def resume_identity(plan, runtime, dtype, gpu, budget):
    return {
        "schema": "preference_resume_v1", "plan_sha256": plan["sha256"],
        "runtime": runtime, "dtype": dtype, "gpu": gpu, "budget_updates": budget,
    }


def validate_resume(path, identity):
    path = Path(path).resolve()
    manifest = json.loads((path / "resume_manifest.json").read_text("utf-8"))
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("Resume source/data/M1/config/runtime/GPU/budget differs")
    files = checkpoint_files(path, fp16=identity["dtype"] == "float16")
    if files != manifest.get("files"):
        raise ValueError("Resume checkpoint payload changed or is incomplete")
    state = json.loads((path / "trainer_state.json").read_text("utf-8"))
    step = state.get("global_step")
    if (type(step) is not int or not 0 < step < identity["budget_updates"]
            or step != manifest.get("global_step")
            or state.get("max_steps") != identity["budget_updates"]):
        raise ValueError("Resume checkpoint is outside the original training budget")
    return {"checkpoint": str(path), "global_step": step,
            "manifest_sha256": sha256_file(path / "resume_manifest.json")}


def checkpoint_callback(identity, stop_after_step=None):
    from transformers import TrainerCallback

    class VerifiedCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= state.max_steps:
                control.should_save = True
            if stop_after_step and state.global_step >= stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            files = checkpoint_files(path, fp16=identity["dtype"] == "float16")
            manifest = {**identity, "global_step": state.global_step, "files": files}
            target = path / "resume_manifest.json"
            if target.exists():
                raise FileExistsError(f"Refusing to replace checkpoint manifest: {target}")
            write_json(target, manifest)
            return control

    return VerifiedCheckpoint()
