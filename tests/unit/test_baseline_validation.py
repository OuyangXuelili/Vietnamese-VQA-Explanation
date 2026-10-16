import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_baseline_validation as baseline


@pytest.mark.historical
def test_validation_runner_imports_on_the_frozen_inference_release(tmp_path):
    root = Path(__file__).resolve().parents[2]
    release = 'source_20261002_062529_455947'
    archive_path = root / f'artifacts/releases/{release}/{release}.zip'
    if not archive_path.is_file():
        pytest.skip('Historical inference release is not included in source-only checkouts')
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        '6b00d2d0297d06f4365248fd2bf63fd31c69808700888219ffd6e7c90283d9fb')
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_path)
    frozen = tmp_path / 'Vietnamese-VQA-Explanation'
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(frozen / 'src'), str(frozen)])
    checked = subprocess.run(
        [sys.executable, '-X', 'utf8', str(Path(baseline.__file__)), '--help'],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding='utf-8', timeout=60, check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert '--m0-predictions' in checked.stdout


def test_cache_helper_imports_use_the_stable_run_m0_interface():
    tree = ast.parse(Path(baseline.__file__).read_text('utf-8'))
    imports = {node.module: {alias.name for alias in node.names}
               for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert {'read_jsonl', 'select_records', 'validate_selected_records'} <= (
        imports['scripts.run_m0'])
    assert imports['vqa_explanation.inference.export'] == {
        'export_inputs', 'validate_saved_predictions'}


def test_commands_use_full_validation_and_same_protocol(tmp_path):
    environment = {"annotations": tmp_path / "metadata/all.jsonl", "image_root": tmp_path}
    commands = baseline.build_commands(environment, tmp_path / "checkpoint-600", tmp_path / "run")
    for command in commands[:2]:
        for flag, value in (("--limit", "0"), ("--split", "validation"), ("--seed", "42"),
                            ("--max-pixels", "200704"), ("--max-new-tokens", "128")):
            assert command[command.index(flag) + 1] == value
        assert "--resume" in command and "--load-in-4bit" in command
        assert command[command.index("--annotations") + 1].endswith("validation.jsonl")
    assert "--adapter" not in commands[0]
    assert commands[1][commands[1].index("--adapter") + 1].endswith("checkpoint-600")
    assert commands[2][-1] == baseline.VALIDATION_SHA256
    assert all("train" not in Path(command[2]).stem for command in commands)


def test_first_export_does_not_resume_but_saved_plan_stays_stable(tmp_path):
    output = tmp_path / "m0.jsonl"
    command = ["python", "run_m0.py", "--output", str(output), "--resume"]
    actual = baseline.export_command_for_state(command)
    assert "--resume" not in actual
    assert "--resume" in command


@pytest.mark.parametrize("artifact", ["predictions", "manifest", "both"])
def test_existing_export_keeps_strict_resume_checks(tmp_path, artifact):
    output = tmp_path / "m0.jsonl"
    if artifact in ("predictions", "both"):
        output.write_text("original", encoding="utf-8")
    if artifact in ("manifest", "both"):
        output.with_suffix(".jsonl.manifest.json").write_text("{}", encoding="utf-8")
    command = ["python", "run_m0.py", "--output", str(output), "--resume"]
    assert baseline.export_command_for_state(command) == command


def test_execute_starts_both_new_exports_then_resumes_them(tmp_path, monkeypatch):
    environment = {"annotations": tmp_path / "metadata/all.jsonl", "image_root": tmp_path}
    commands = baseline.build_commands(environment, tmp_path / "checkpoint-600", tmp_path)
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            launched.append(command)
            self.stdout = []
            if "--experiment-id" in command:
                output = Path(command[command.index("--output") + 1])
                output.touch(exist_ok=True)
                output.with_suffix(".jsonl.manifest.json").touch(exist_ok=True)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(baseline.subprocess, "Popen", Process)
    baseline.execute(commands, tmp_path)
    assert all("--resume" not in command for command in launched[:2])
    baseline.execute(commands, tmp_path)
    assert all("--resume" in command for command in launched[3:5])
    assert all("--resume" in command for command in commands[:2])
    assert launched[2] == launched[5] == commands[2]


@pytest.fixture
def job(tmp_path, monkeypatch):
    environment = {"annotations": tmp_path / "all.jsonl", "image_root": tmp_path}
    monkeypatch.setattr(baseline, "load_environment", lambda path: environment)
    monkeypatch.setattr(baseline, "validate_inputs", lambda *args: None)
    executed = []
    monkeypatch.setattr(baseline, "execute", lambda commands, output: executed.append(commands))
    args = SimpleNamespace(
        environment=tmp_path / "environment.yaml", checkpoint=tmp_path / "checkpoint-600",
        output_dir=tmp_path / "job", resume=False, dry_run=False, m0_predictions=None,
    )
    return args, executed


def test_dry_run_does_not_create_output(job):
    args, executed = job
    args.dry_run = True
    baseline.run(args)
    assert not args.output_dir.exists() and not executed


def test_resume_checks_original_commands(job):
    args, executed = job
    baseline.run(args)
    with pytest.raises(ValueError, match="directory exists"):
        baseline.run(args)
    args.resume = True
    baseline.run(args)
    assert len(executed) == 2
    args.checkpoint = args.checkpoint.with_name("checkpoint-700")
    with pytest.raises(ValueError, match="commands changed"):
        baseline.run(args)
    assert len(executed) == 2


def test_completed_report_not_overwritten(job):
    args, executed = job
    baseline.run(args)
    report = args.output_dir / "full_validation.json"
    report.write_text('{"comparable":true}', encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="report already exists"):
        baseline.run(args)
    assert len(executed) == 1 and json.loads(report.read_text()) == {"comparable": True}


def test_bad_inputs_stop_before_execute(job, monkeypatch):
    args, executed = job

    def invalid(*args):
        raise ValueError("Annotation hash mismatch")

    monkeypatch.setattr(baseline, "validate_inputs", invalid)
    with pytest.raises(ValueError, match="hash mismatch"):
        baseline.run(args)
    assert not executed and not args.output_dir.exists()


def test_export_without_job_plan_is_preserved(job):
    args, executed = job
    args.output_dir.mkdir()
    export = args.output_dir / "m0.jsonl"
    export.write_text("original", encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="no job plan"):
        baseline.run(args)
    assert export.read_text() == "original" and not executed


@pytest.mark.parametrize("adapter", [{"base_model_name_or_path": "another-model"}, []])
def test_wrong_adapter_is_rejected(tmp_path, monkeypatch, adapter):
    monkeypatch.setattr(
        baseline, "read_expected_validation_annotations", lambda *a, **kw: ({}, "hash")
    )
    monkeypatch.setattr(baseline, "validate_selected_records", lambda *a: None)
    monkeypatch.setattr(baseline, "validate_checkpoint", lambda *a: None)
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
    environment = {"annotations": tmp_path / "all.jsonl", "image_root": tmp_path}
    with pytest.raises(ValueError, match="pinned Qwen"):
        baseline.validate_inputs(environment, tmp_path)


def test_cached_commands_only_generate_m1(tmp_path):
    environment = {"annotations": tmp_path / "metadata/all.jsonl", "image_root": tmp_path}
    cached = tmp_path / "previous/m0.jsonl"
    commands = baseline.build_commands(environment, tmp_path / "checkpoint-1200", tmp_path, cached)
    assert len(commands) == 2
    assert commands[0][commands[0].index("--experiment-id") + 1] == "M1_SFT"
    assert commands[1][3] == str(cached)
    assert commands[1][4] == str(tmp_path / "m1.jsonl")


@pytest.fixture
def m0_cache(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    records = [{
        "question_id": str(index), "image_id": str(index), "image_path": f"{index}.jpg",
        "split": "validation", "question": "Con gì?", "answer": "mèo",
        "explanations": ["Có một con mèo."],
    } for index in range(2)]
    annotations = metadata / "validation.jsonl"
    annotations.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    for row in records:
        (tmp_path / row["image_path"]).write_bytes(b"image fixture")
    environment = {"annotations": metadata / "all.jsonl", "image_root": tmp_path}
    selected = baseline.select_records(records, "validation", 0, 42)
    args = SimpleNamespace(
        annotations=annotations, model_id=baseline.DEFAULT_MODEL,
        revision=baseline.DEFAULT_REVISION, experiment_id="M0_PRETRAINED", adapter=None,
        split="validation", limit=0, seed=42, min_pixels=200704, max_pixels=200704,
        max_new_tokens=128, load_in_4bit=True,
    )
    hashes = {row["question_id"]: baseline.sha256_file(tmp_path / row["image_path"])
              for row in selected}
    inputs = baseline.export_inputs(args, selected, hashes)
    protocol = {"max_new_tokens": 128, "max_pixels": 200704}
    manifest = {"schema_version": 1, "inputs": inputs, "protocol": protocol}
    predictions = tmp_path / "m0.jsonl"
    saved = [{
        **protocol, **row, "experiment_id": "M0_PRETRAINED", "adapter": None,
        "model_id": baseline.DEFAULT_MODEL, "model_revision": baseline.DEFAULT_REVISION,
        "seed": 42, "image_sha256": hashes[row["question_id"]],
        "reference_answer": row["answer"], "reference_explanations": row["explanations"],
        "raw_output": "Đáp án: mèo\nGiải thích: Có một con mèo.",
        "prediction_answer": "mèo", "prediction_explanation": "Có một con mèo.",
        "format_valid": True, "latency_seconds": 1.0,
    } for row in selected]
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in saved), encoding="utf-8"
    )
    manifest_path = predictions.with_suffix(".jsonl.manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return environment, predictions, manifest_path


def test_complete_m0_cache_is_read_only(m0_cache):
    environment, predictions, _ = m0_cache
    before = baseline.cache_identity(predictions)
    assert baseline.validate_m0_cache(environment, predictions) == before
    assert baseline.cache_identity(predictions) == before


@pytest.mark.parametrize("fault", ["partial", "duplicate", "answer", "image", "newline",
                                  "source", "version", "protocol", "schema"])
def test_invalid_m0_cache_is_rejected(m0_cache, fault):
    environment, predictions, manifest_path = m0_cache
    rows = baseline.read_jsonl(predictions)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if fault == "partial":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "answer":
        rows[0]["reference_answer"] = "chó"
    elif fault == "image":
        (environment["image_root"] / rows[0]["image_path"]).write_bytes(b"changed image")
    elif fault == "source":
        manifest["inputs"]["source_files"]["scripts/run_m0.py"] = "0" * 64
    elif fault == "version":
        manifest["inputs"]["versions"]["transformers"] = "different"
    elif fault == "protocol":
        manifest["protocol"] = {}
    elif fault == "schema":
        manifest["schema_version"] = 2
    predictions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("" if fault == "newline" else "\n"), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    before = baseline.cache_identity(predictions)
    with pytest.raises(ValueError):
        baseline.validate_m0_cache(environment, predictions)
    assert baseline.cache_identity(predictions) == before


def test_cached_job_records_provenance_and_does_not_copy_m0(job, m0_cache, monkeypatch):
    args, executed = job
    environment, predictions, _ = m0_cache
    monkeypatch.setattr(baseline, "load_environment", lambda path: environment)
    args.m0_predictions = predictions
    before = baseline.cache_identity(predictions)
    baseline.run(args)
    assert len(executed[0]) == 2
    assert not (args.output_dir / "m0.jsonl").exists()
    assert json.loads((args.output_dir / "m0_cache.json").read_text()) == before
    args.resume = True
    baseline.run(args)
    assert len(executed) == 2 and baseline.cache_identity(predictions) == before
    with predictions.open("a", encoding="utf-8") as destination:
        destination.write("\n")
    with pytest.raises(ValueError, match="cache changed"):
        baseline.run(args)
    assert len(executed) == 2


def test_bad_cache_does_not_launch_gpu(job, m0_cache, monkeypatch):
    args, executed = job
    environment, predictions, _ = m0_cache
    monkeypatch.setattr(baseline, "load_environment", lambda path: environment)
    args.m0_predictions = predictions
    predictions.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        baseline.run(args)
    assert not executed and not (args.output_dir / "commands.json").exists()


def test_cache_cannot_be_inside_output(job):
    args, executed = job
    args.m0_predictions = args.output_dir / "m0.jsonl"
    with pytest.raises(ValueError, match="outside"):
        baseline.run(args)
    assert not executed and not args.output_dir.exists()
