import csv
import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
import yaml
from PIL import Image

from scripts import run_pipeline
from scripts.run_pipeline import verify_resume, verify_training_artifacts
from vqa_explanation.training.sft_full import (
    plan_data,
    validate_checkpoint,
    validate_config,
    validate_resume_dtype,
    validate_resume_plan,
    validate_resume_runtime,
    validate_session_stop_target,
    validate_training_completion,
)
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def test_resume_refuses_silent_precision_change(tmp_path):
    checkpoint = tmp_path / "checkpoint-400"
    checkpoint.mkdir()
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"compute_dtype": "torch.bfloat16"}), encoding="utf-8",
    )
    validate_resume_dtype(None, "torch.float16")
    validate_resume_dtype(checkpoint, "torch.bfloat16")
    with pytest.raises(ValueError, match="compute dtype differs"):
        validate_resume_dtype(checkpoint, "torch.float16")


def small_config():
    cfg = yaml.safe_load((project_root() / "configs/training/sft.yaml").read_text("utf-8"))
    cfg.update(
        expected_train_count=2,
        expected_validation_count=2,
        expected_test_count=1,
        eval_sample_size=1,
    )
    return cfg


def sample(qid, image_id, split):
    return {
        "question_id": qid,
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "question": "Có gì?",
        "answer": "mèo",
        "explanations": ["Có mèo trong ảnh."],
        "split": split,
    }


def prepared_data(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records = [
        sample("t1", "i1", "train"),
        sample("t2", "i2", "train"),
        sample("v1", "i3", "validation"),
        sample("v2", "i4", "validation"),
        sample("e1", "i5", "test"),
    ]
    for row in records:
        Image.new("RGB", (4, 4)).save(tmp_path / row["image_path"])
    return records


def test_data_plan_is_full_split_disjoint_and_deterministic(tmp_path):
    records = prepared_data(tmp_path)
    cfg = small_config()
    train, validation, plan = plan_data(records, tmp_path, cfg)
    assert {row["question_id"] for row in train} == {"t1", "t2"}
    assert len(validation) == 1 and validation[0]["split"] == "validation"
    assert plan["train_count"] == 2
    assert plan["test_count"] == 1
    assert plan["split_overlap"] == {
        "test__train": 0,
        "test__validation": 0,
        "train__validation": 0,
    }
    assert plan == plan_data(records, tmp_path, cfg)[2]


def test_data_plan_rejects_leakage_and_missing_images(tmp_path):
    records = prepared_data(tmp_path)
    cfg = small_config()
    leaked = deepcopy(records)
    leaked[2]["image_id"] = "i1"
    with pytest.raises(ValueError, match="overlap"):
        plan_data(leaked, tmp_path, cfg)
    (tmp_path / "images" / "i2.jpg").unlink()
    with pytest.raises(ValueError, match="Missing"):
        plan_data(records, tmp_path, cfg)


def test_data_plan_rejects_same_image_path_with_different_ids(tmp_path):
    records = prepared_data(tmp_path)
    records[2]["image_path"] = records[0]["image_path"]
    with pytest.raises(ValueError, match="Image path overlaps train and validation"):
        plan_data(records, tmp_path, small_config())


def test_data_plan_rejects_unsafe_test_image_path(tmp_path):
    records = prepared_data(tmp_path)
    records[-1]["image_path"] = "../outside.jpg"
    with pytest.raises(ValueError, match="Unsafe image path"):
        plan_data(records, tmp_path, small_config())


def test_data_plan_rejects_split_alias_not_used_by_training_selector(tmp_path):
    records = prepared_data(tmp_path)
    records[2]["split"] = "val"
    with pytest.raises(ValueError, match="Noncanonical split label"):
        plan_data(records, tmp_path, small_config())


def test_config_rejects_wrong_checkpoint_schedule_or_split():
    cfg = small_config()
    validate_config(cfg)
    cfg["save_steps"] = 501
    with pytest.raises(ValueError, match="multiple"):
        validate_config(cfg)
    cfg["save_steps"] = 500
    cfg["expected_test_count"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(cfg)


def test_pilot_is_explicitly_not_a_full_m1_run():
    cfg = yaml.safe_load(
        (project_root() / "configs/training/sft_optimizer_pilot.yaml").read_text("utf-8")
    )
    validate_config(cfg)
    cfg["experiment_id"] = "M1_SFT"
    with pytest.raises(ValueError, match="max_steps"):
        validate_config(cfg)


def test_training_completion_requires_target_step_and_real_resume_progress():
    validate_training_completion(20, 20, 0, 1.7, 0.001)
    validate_training_completion(40, 40, 20, 1.6, None)
    with pytest.raises(RuntimeError, match="before target"):
        validate_training_completion(19, 20, 0, 1.7, 0.001)
    with pytest.raises(RuntimeError, match="No new optimizer step"):
        validate_training_completion(20, 40, 20, 1.7, None)
    with pytest.raises(RuntimeError, match="LoRA weights did not update"):
        validate_training_completion(20, 20, 0, 1.7, 0.0)


def test_resume_requires_incomplete_matching_run_and_optimizer_state(tmp_path):
    run_root = tmp_path / "runs"
    checkpoint = run_root / "sft-full_old" / "training" / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 500}), "utf-8")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
    previous = {
        "stage": "sft-full",
        "status": "failed",
        "provenance": {
            "source_sha256": "source",
            "input_sha256": {
                "environment": "environment",
                "annotations": "annotations",
                "training_config": "config",
                "model_config": "model",
                "image_checksums": "images",
            },
        },
    }
    (checkpoint.parent.parent / "execution.json").write_text(json.dumps(previous), "utf-8")
    assert validate_checkpoint(checkpoint) == checkpoint.resolve()
    assert verify_resume(checkpoint, run_root, previous["provenance"]) == checkpoint.resolve()
    changed = deepcopy(previous["provenance"])
    changed["input_sha256"]["annotations"] = "other"
    with pytest.raises(ValueError, match="annotations"):
        verify_resume(checkpoint, run_root, changed)
    changed = deepcopy(previous["provenance"])
    changed["input_sha256"]["environment"] = "other"
    with pytest.raises(ValueError, match="environment"):
        verify_resume(checkpoint, run_root, changed)
    for status in ("paused", "failed", "interrupted_or_failed"):
        previous["status"] = status
        (checkpoint.parent.parent / "execution.json").write_text(json.dumps(previous), "utf-8")
        assert verify_resume(checkpoint, run_root, previous["provenance"]) == checkpoint.resolve()
    for status in ("running", "unknown", "passed", None):
        previous["status"] = status
        (checkpoint.parent.parent / "execution.json").write_text(json.dumps(previous), "utf-8")
        with pytest.raises(ValueError, match="stopped, incomplete"):
            verify_resume(checkpoint, run_root, previous["provenance"])


def test_stale_running_run_can_be_recovered_explicitly_without_rewriting_it(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "runs"
    previous_run = run_root / "sft-full_old"
    checkpoint = previous_run / "training" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 10}), "utf-8")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
    provenance = {
        "source_sha256": "source",
        "input_sha256": {
            "environment": "env",
            "annotations": "data",
            "training_config": "train",
            "model_config": "model",
            "image_checksums": "images",
        },
    }
    previous = {
        "stage": "sft-full",
        "status": "running",
        "provenance": provenance,
        "runner": {
            "id": "runner-1",
            "host": socket.gethostname(),
            "pid": 101,
            "child_pid": 102,
        },
    }
    metadata_path = previous_run / "execution.json"
    metadata_path.write_text(json.dumps(previous), encoding="utf-8")
    heartbeat_path = previous_run / "runner_heartbeat.json"
    stale_time = datetime.now(timezone.utc) - run_pipeline.INTERRUPTED_RUN_GRACE
    heartbeat_path.write_text(
        json.dumps(
            {
                "runner_id": "runner-1",
                "updated_at_utc": (stale_time - timedelta(minutes=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_pipeline, "process_alive", lambda _pid: False)
    original_metadata = metadata_path.read_bytes()
    with pytest.raises(ValueError, match="stopped, incomplete"):
        verify_resume(checkpoint, run_root, provenance)
    assert verify_resume(
        checkpoint, run_root, provenance, recover_interrupted=True
    ) == checkpoint.resolve()
    assert metadata_path.read_bytes() == original_metadata
    changed = deepcopy(provenance)
    changed["input_sha256"]["annotations"] = "changed"
    with pytest.raises(ValueError, match="annotations"):
        verify_resume(checkpoint, run_root, changed, recover_interrupted=True)
    (checkpoint / "optimizer.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="optimizer"):
        verify_resume(checkpoint, run_root, provenance, recover_interrupted=True)
    (checkpoint / "optimizer.pt").write_bytes(b"state")

    monkeypatch.setattr(run_pipeline, "process_alive", lambda pid: pid == 102)
    with pytest.raises(ValueError, match="child_pid is live"):
        verify_resume(checkpoint, run_root, provenance, recover_interrupted=True)
    monkeypatch.setattr(run_pipeline, "process_alive", lambda _pid: None)
    with pytest.raises(ValueError, match="live or unverified"):
        verify_resume(checkpoint, run_root, provenance, recover_interrupted=True)

    monkeypatch.setattr(run_pipeline, "process_alive", lambda _pid: False)
    recent_heartbeat = {
        "runner_id": "runner-1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    heartbeat_path.write_text(
        json.dumps(recent_heartbeat),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="heartbeat is recent"):
        verify_resume(checkpoint, run_root, provenance, recover_interrupted=True)

    heartbeat_path.write_text(
        json.dumps({"runner_id": "runner-2", "updated_at_utc": "2020-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        verify_resume(checkpoint, run_root, provenance, recover_interrupted=True)


def test_resume_rejects_malformed_previous_metadata(tmp_path):
    run_root = tmp_path / "runs"
    previous_run = run_root / "sft-full_old"
    checkpoint = previous_run / "training" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    metadata = previous_run / "execution.json"
    metadata.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        verify_resume(checkpoint, run_root, {"source_sha256": "source"})
    metadata.write_text(
        json.dumps({"stage": "sft-full", "status": "failed", "provenance": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="valid provenance"):
        verify_resume(checkpoint, run_root, {"source_sha256": "source"})


def test_training_artifact_gate_refuses_silent_success_and_wrong_outputs(tmp_path):
    run = tmp_path / "sft-full_run"
    training = run / "training"
    training.mkdir(parents=True)
    with pytest.raises(ValueError, match="training_result.json"):
        verify_training_artifacts(run, None)
    with pytest.raises(ValueError, match="session_result.json"):
        verify_training_artifacts(run, 10)

    checkpoint = training / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 10}), encoding="utf-8"
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    session = {"status": "paused", "global_step": 10, "checkpoint": str(checkpoint)}
    session_path = training / "session_result.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    assert verify_training_artifacts(run, 10) == "paused"
    session_path.write_text(json.dumps({**session, "global_step": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer step"):
        verify_training_artifacts(run, 10)
    session_path.write_text(
        json.dumps({**session, "checkpoint": str(training / "checkpoint-20")}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different checkpoint"):
        verify_training_artifacts(run, 10)

    adapter = training / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"state")
    report = {
        "status": "passed",
        "global_step": 20,
        "target_global_step": 20,
        "adapter": str(adapter),
    }
    result_path = training / "training_result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")
    assert verify_training_artifacts(run, None) == "passed"
    result_path.write_text(
        json.dumps({**report, "target_global_step": 30}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete optimizer step"):
        verify_training_artifacts(run, None)
    result_path.write_text(json.dumps(report), encoding="utf-8")
    (adapter / "adapter_config.json").write_bytes(b"")
    with pytest.raises(ValueError, match="adapter weights"):
        verify_training_artifacts(run, None)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="adapter weights"):
        verify_training_artifacts(run, None)


def test_process_liveness_recognizes_current_process():
    assert run_pipeline.process_alive(os.getpid()) is True


def test_resume_rejects_bad_checkpoint_state_or_missing_adapter(tmp_path):
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
    state = checkpoint / "trainer_state.json"
    state.write_text("not-json", "utf-8")
    with pytest.raises(ValueError, match="Invalid trainer_state"):
        validate_checkpoint(checkpoint)
    state.write_text(json.dumps({"global_step": 499}), "utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint(checkpoint)
    state.write_text(json.dumps({"global_step": 500}), "utf-8")
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="adapter"):
        validate_checkpoint(checkpoint)
    (checkpoint / "adapter_model.bin").write_bytes(b"state")
    assert validate_checkpoint(checkpoint) == checkpoint.resolve()
    (checkpoint / "optimizer.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="optimizer"):
        validate_checkpoint(checkpoint)


def test_resume_requires_rng_and_same_runtime_versions(tmp_path):
    checkpoint = tmp_path / "training" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 10}), "utf-8")
    (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
    for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    with pytest.raises(ValueError, match="rng_state"):
        validate_checkpoint(checkpoint)
    (checkpoint / "rng_state.pth").write_bytes(b"state")
    assert validate_checkpoint(checkpoint) == checkpoint.resolve()

    versions = {"transformers": "4.57.6", "torch": "2.11.0"}
    (checkpoint.parent / "run_metadata.json").write_text(
        json.dumps({"versions": versions}), "utf-8"
    )
    validate_resume_runtime(checkpoint, versions)
    with pytest.raises(ValueError, match="differ"):
        validate_resume_runtime(checkpoint, {**versions, "torch": "2.12.0"})
    with pytest.raises(ValueError, match="tested Transformers"):
        validate_resume_runtime(None, {**versions, "transformers": "4.49.0"})


def test_resume_requires_same_data_plan_and_checkpoint_aligned_stop(tmp_path):
    checkpoint = tmp_path / "training" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    cfg = yaml.safe_load(
        (project_root() / "configs/training/sft_optimizer_pilot.yaml").read_text("utf-8")
    )
    plan = {"annotations_sha256": "a", "train_question_ids_sha256": "ids"}
    (checkpoint.parent / "data_plan.json").write_text(json.dumps(plan), "utf-8")
    validate_resume_plan(checkpoint, plan)
    with pytest.raises(ValueError, match="data plan differs"):
        validate_resume_plan(checkpoint, {**plan, "annotations_sha256": "b"})
    validate_session_stop_target(10, 0, cfg)
    with pytest.raises(ValueError, match="after the resume"):
        validate_session_stop_target(10, 10, cfg)
    with pytest.raises(ValueError, match="align"):
        validate_session_stop_target(11, 0, cfg)
    with pytest.raises(ValueError, match="precede"):
        validate_session_stop_target(20, 0, cfg)


def test_full_stage_dry_run_does_not_allocate_gpu_or_create_run(tmp_path):
    run_root = tmp_path / "runs"
    env = tmp_path / "environment.yaml"
    env.write_text(
        f"annotations: {(tmp_path / 'annotations.jsonl').as_posix()}\n"
        f"image_root: {(tmp_path / 'images').as_posix()}\n"
        f"run_root: {run_root.as_posix()}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "sft-full",
            "--environment",
            str(env),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "train_sft_full.py" in result.stdout
    assert "sft.yaml" in result.stdout
    assert not run_root.exists()


def test_optimizer_pilot_dry_run_passes_stop_and_resume_flags(tmp_path):
    checkpoint = tmp_path / "runs" / "old" / "training" / "checkpoint-10"
    env = tmp_path / "environment.yaml"
    env.write_text(
        f"annotations: {(tmp_path / 'annotations.jsonl').as_posix()}\n"
        f"image_root: {(tmp_path / 'images').as_posix()}\n"
        f"run_root: {(tmp_path / 'runs').as_posix()}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "sft-optimizer-pilot",
            "--environment",
            str(env),
            "--stop-after-step",
            "10",
            "--resume-from-checkpoint",
            str(checkpoint),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"--stop-after-step"' in result.stdout
    assert '"--resume-from-checkpoint"' in result.stdout
    assert not checkpoint.parent.parent.parent.exists()


def test_full_preflight_cli_rejects_replaced_valid_image_before_run(tmp_path):
    records = prepared_data(tmp_path)
    annotations = tmp_path / "all.jsonl"
    annotations.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), "utf-8"
    )
    cfg = tmp_path / "sft.yaml"
    cfg.write_text(yaml.safe_dump(small_config()), "utf-8")
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("model_id: fake\n", "utf-8")
    checksums = tmp_path / "full_image_checksums.csv"
    with checksums.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["image_name", "size_bytes", "sha256"])
        for row in records:
            image = tmp_path / row["image_path"]
            writer.writerow([image.name, image.stat().st_size, sha256_file(image)])
    Image.new("RGB", (4, 4), "blue").save(tmp_path / records[0]["image_path"])
    output_dir = tmp_path / "training"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/train_sft_full.py"),
            "--annotations",
            str(annotations),
            "--image-root",
            str(tmp_path),
            "--image-checksums",
            str(checksums),
            "--config",
            str(cfg),
            "--model-config",
            str(model_cfg),
            "--output-dir",
            str(output_dir),
            "--preflight-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Image checksum mismatch" in result.stderr
    assert not output_dir.exists()
