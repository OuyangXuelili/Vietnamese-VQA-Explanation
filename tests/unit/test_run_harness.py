import hashlib
import json
import subprocess
import sys

import pytest
from PIL import Image

from scripts import run_pipeline
from scripts.check_project import commands
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file, source_fingerprints


def test_file_hash_is_streamed_and_source_ignores_data(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_bytes(b"source" * 200_000)
    (tmp_path / "data" / "private.jsonl").write_text("private", encoding="utf-8")
    assert sha256_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_fingerprints(tmp_path) == {"src/model.py": sha256_file(source)}


def test_check_gate_has_no_gpu_stage():
    assert len(commands(False, False)) == 2
    full = commands(False, True)
    assert "--all-train-records" in full[-1]
    assert all("sft-smoke" not in part for command in full for part in command)


def test_check_gate_dry_run_creates_nothing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/check_project.py"),
            "--full-data",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--all-train-records" in result.stdout
    assert not list(tmp_path.iterdir())


def test_audit_run_records_provenance_and_does_not_overwrite(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / "sample.jpg")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "image_id": "i1",
                "image_path": "sample.jpg",
                "question": "Ảnh có gì?",
                "answer": "Một ảnh",
                "explanations": ["Có một ảnh."],
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    environment = tmp_path / "environment.yaml"
    checksums = tmp_path / "checksums.csv"
    checksums.write_text("image_name,sha256\nsample.jpg,test\n", encoding="utf-8")
    environment.write_text(
        f"annotations: {annotations.as_posix()}\n"
        f"image_root: {images.as_posix()}\n"
        f"run_root: {(tmp_path / 'runs').as_posix()}\n"
        f"image_checksums: {checksums.as_posix()}\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(project_root() / "scripts/run_pipeline.py"),
        "audit",
        "--environment",
        str(environment),
    ]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 2
    for run in runs:
        metadata = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "passed"
        assert metadata["returncode"] == 0
        assert metadata["finished_at_utc"]
        assert metadata["runner"]["pid"] > 0
        assert metadata["runner"]["child_pid"] > 0
        heartbeat = json.loads((run / "runner_heartbeat.json").read_text(encoding="utf-8"))
        assert heartbeat["runner_id"] == metadata["runner"]["id"]
        assert metadata["provenance"]["input_sha256"]["annotations"] == sha256_file(annotations)
        assert metadata["provenance"]["source_sha256"]
        assert metadata["provenance"]["input_sha256"]["image_checksums"] == sha256_file(checksums)
        assert (run / "audit.json").is_file()
        assert (run / "console.log").is_file()


def test_failed_audit_is_recorded_as_failed(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "image_id": "i1",
                "image_path": "missing.jpg",
                "question": "Ảnh có gì?",
                "answer": "Một ảnh",
                "explanations": ["Có một ảnh."],
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        f"annotations: {annotations.as_posix()}\n"
        f"image_root: {images.as_posix()}\n"
        f"run_root: {(tmp_path / 'runs').as_posix()}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "audit",
            "--environment",
            str(environment),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    run = next((tmp_path / "runs").iterdir())
    metadata = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["returncode"] != 0
    assert metadata["finished_at_utc"]


@pytest.mark.parametrize(
    ("stage", "missing"),
    [("sft-optimizer-pilot", "training_result.json"), ("sft-smoke", "smoke_result.json")],
)
def test_training_runner_refuses_zero_exit_without_result_artifacts(
    tmp_path, monkeypatch, stage, missing
):
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("{}\n", encoding="utf-8")
    checksums = tmp_path / "checksums.csv"
    checksums.write_text("image_name,sha256\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        f"annotations: {annotations.as_posix()}\n"
        f"image_root: {images.as_posix()}\n"
        f"run_root: {(tmp_path / 'runs').as_posix()}\n"
        f"image_checksums: {checksums.as_posix()}\n",
        encoding="utf-8",
    )

    class ChildWithoutResult:
        pid = 12345
        stdout = ()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", lambda *_a, **_k: ChildWithoutResult())
    monkeypatch.setattr(
        run_pipeline,
        "collect_provenance",
        lambda *_a, **_k: {"source_sha256": "source", "input_sha256": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", stage, "--environment", str(environment)],
    )
    with pytest.raises(SystemExit) as exit_info:
        run_pipeline.main()
    assert exit_info.value.code == 1
    run = next((tmp_path / "runs").iterdir())
    metadata = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == 1
    assert missing in metadata["artifact_validation_error"]


def test_smoke_artifact_gate_rejects_missing_or_incomplete_evidence(tmp_path):
    run = tmp_path / "sft-smoke_run"
    training = run / "training"
    training.mkdir(parents=True)
    with pytest.raises(ValueError, match="smoke_result.json"):
        run_pipeline.verify_smoke_artifacts(run)

    adapter = training / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    report = {
        "status": "passed",
        "global_step": 2,
        "target_global_step": 2,
        "max_lora_weight_change": 0.1,
        "adapter": str(adapter),
    }
    report_path = training / "smoke_result.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    run_pipeline.verify_smoke_artifacts(run)

    report_path.write_text(json.dumps({**report, "target_global_step": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete optimizer step"):
        run_pipeline.verify_smoke_artifacts(run)
    report_path.write_text(
        json.dumps({**report, "max_lora_weight_change": float("nan")}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no LoRA weight update"):
        run_pipeline.verify_smoke_artifacts(run)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="adapter weights"):
        run_pipeline.verify_smoke_artifacts(run)
