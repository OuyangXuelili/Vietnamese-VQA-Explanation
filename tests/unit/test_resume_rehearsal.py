import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rehearse_m1_resume


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _safetensors_lora_b(value: float) -> bytes:
    data = struct.pack("<f", value)
    header = json.dumps(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(data)],
            }
        }
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + data


def _checkpoint(run: Path, step: int, *, complete: bool = True, lora_b_value: float = 0.5) -> Path:
    checkpoint = run / "training" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "trainer_state.json", {"global_step": step})
    names = [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    for name in names:
        if complete or name != "optimizer.pt":
            value = (
                _safetensors_lora_b(lora_b_value)
                if name == "adapter_model.safetensors"
                else b"saved"
            )
            (checkpoint / name).write_bytes(value)
    return checkpoint


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    run_root = tmp_path / "runs"
    run_root.mkdir()
    environment = root / "environment.yaml"
    environment.write_text(
        f"annotations: {tmp_path / 'annotations.jsonl'}\n"
        f"image_root: {tmp_path / 'images'}\n"
        f"run_root: {run_root}\n",
        encoding="utf-8",
    )
    return root, run_root, environment


def test_rehearsal_checks_two_new_runs_and_writes_report(tmp_path, monkeypatch):
    root, run_root, environment = _setup(tmp_path)
    old_run = run_root / "sft-optimizer-pilot_old"
    old_run.mkdir()
    (old_run / "keep.txt").write_text("historical", encoding="utf-8")
    provenance = {"source_sha256": "fixed-source", "input_sha256": {"environment": "fixed"}}
    first = run_root / "sft-optimizer-pilot_first"
    second = run_root / "sft-optimizer-pilot_second"
    commands = []

    def fake_run(command, *, cwd, check):
        assert cwd == root
        assert check is False
        commands.append(command)
        if len(commands) == 1:
            checkpoint = _checkpoint(first, 10)
            _write_json(
                first / "execution.json",
                {
                    "stage": "sft-optimizer-pilot",
                    "status": "paused",
                    "returncode": 0,
                    "provenance": provenance,
                },
            )
            _write_json(
                first / "training/session_result.json",
                {
                    "status": "paused",
                    "global_step": 10,
                    "target_global_step": 20,
                    "optimizer_steps_this_run": 10,
                    "checkpoint": str(checkpoint),
                    "max_lora_weight_change": 0.01,
                    "train_metrics": {"train_loss": 1.5},
                },
            )
        else:
            _checkpoint(second, 20, lora_b_value=0.75)
            adapter = second / "training/adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_bytes(b"config")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            _write_json(
                second / "execution.json",
                {
                    "stage": "sft-optimizer-pilot",
                    "status": "passed",
                    "returncode": 0,
                    "resume_from_checkpoint": str(first / "training/checkpoint-10"),
                    "provenance": provenance,
                },
            )
            _write_json(
                second / "training/training_result.json",
                {
                    "status": "passed",
                    "experiment_id": "M1_SFT_PILOT",
                    "global_step": 20,
                    "target_global_step": 20,
                    "resumed_from_global_step": 10,
                    "optimizer_steps_this_run": 10,
                    "adapter": str(adapter),
                    "train_metrics": {"train_loss": 1.4},
                    "final_validation": {"eval_loss": 1.3},
                },
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rehearse_m1_resume.subprocess, "run", fake_run)
    report_path = rehearse_m1_resume.rehearse(environment, root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == second / "resume_rehearsal.json"
    assert report["status"] == "passed"
    assert report["paused_run"] == str(first)
    assert report["completed_run"] == str(second)
    assert report["resumed_optimizer_steps"] == 10
    assert report["changed_lora_b_tensors"] == 1
    assert report["lora_b_tensor_count"] == 1
    assert report["max_lora_b_weight_change"] == pytest.approx(0.25)
    assert commands[0][-2:] == ["--stop-after-step", "10"]
    assert commands[1][-2:] == ["--resume-from-checkpoint", str(first / "training/checkpoint-10")]
    assert (old_run / "keep.txt").read_text(encoding="utf-8") == "historical"


def test_rehearsal_stops_before_resume_if_checkpoint_incomplete(tmp_path, monkeypatch):
    root, run_root, environment = _setup(tmp_path)
    first = run_root / "sft-optimizer-pilot_first"
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        checkpoint = _checkpoint(first, 10, complete=False)
        _write_json(
            first / "execution.json",
            {"stage": "sft-optimizer-pilot", "status": "paused", "returncode": 0},
        )
        _write_json(
            first / "training/session_result.json",
            {
                "status": "paused",
                "global_step": 10,
                "target_global_step": 20,
                "optimizer_steps_this_run": 10,
                "checkpoint": str(checkpoint),
                "max_lora_weight_change": 0.01,
                "train_metrics": {"train_loss": 1.5},
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rehearse_m1_resume.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="optimizer.pt"):
        rehearse_m1_resume.rehearse(environment, root)
    assert len(calls) == 1
    assert not (first / "resume_rehearsal.json").exists()


def test_rehearsal_requires_saved_lora_b_to_change_after_resume(tmp_path):
    first = _checkpoint(tmp_path / "first", 10)
    unchanged = _checkpoint(tmp_path / "unchanged", 20)
    with pytest.raises(RuntimeError, match="No saved LoRA-B tensor changed"):
        rehearse_m1_resume._changed_lora_b_tensors(first, unchanged)
    updated = _checkpoint(tmp_path / "updated", 20, lora_b_value=0.75)
    assert rehearse_m1_resume._changed_lora_b_tensors(first, updated) == (1, 1, 0.25)
    nonfinite = _checkpoint(tmp_path / "nonfinite", 20, lora_b_value=float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite"):
        rehearse_m1_resume._changed_lora_b_tensors(first, nonfinite)


def test_rehearsal_rejects_overflowing_weight_delta(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    values = {
        first: {"lora_B.weight": ("F64", (1,), struct.pack("<d", -1e308))},
        second: {"lora_B.weight": ("F64", (1,), struct.pack("<d", 1e308))},
    }
    monkeypatch.setattr(rehearse_m1_resume, "_lora_b_payloads", values.__getitem__)
    with pytest.raises(RuntimeError, match="Non-finite LoRA-B weight change"):
        rehearse_m1_resume._changed_lora_b_tensors(first, second)
