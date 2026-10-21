"""Active notebooks separate train and validation without rewriting run history."""

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vqa_explanation.training.preference_launch import launch_command, stage_output

ROOT = Path(__file__).resolve().parents[2]
TRAIN = "04_train_preferences.ipynb"
EVAL = "05_evaluate_preferences.ipynb"


def notebook(name):
    return json.loads((ROOT / "notebooks" / name).read_text("utf-8"))


def code(name, cell_id):
    return next("".join(cell["source"]) for cell in notebook(name)["cells"]
                if cell["id"] == cell_id)


def execute(name, cell_id, namespace):
    exec(compile(code(name, cell_id), cell_id, "exec"), namespace)


@pytest.mark.parametrize("name, cells, code_cells", [(TRAIN, 16, 10), (EVAL, 12, 7)])
def test_notebook_structure_and_safe_defaults(name, cells, code_cells):
    book = notebook(name)
    assert len(book["cells"]) == cells
    assert len({cell["id"] for cell in book["cells"]}) == cells
    sources = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(sources) == code_cells
    for cell in sources:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    combined = "\n".join("".join(cell["source"]) for cell in sources)
    assert combined.count("drive.mount(") == 1
    assert "timeout_ms=600000" in combined
    assert "preference_training_launch_v1.json" not in combined
    assert "train_sft_full" not in combined
    assert "generate_preference_candidates" not in combined
    assert "expand_preferences.py" not in combined
    config_id = "preference-config" if name == TRAIN else "validation-config"
    namespace = {}
    execute(name, config_id, namespace)
    assert namespace["INSPECTED"] is False
    if name == TRAIN:
        assert namespace["RUN_GATE"] is False and namespace["RUN_TRAINING"] is False
    else:
        assert namespace["RUN_EXPORT"] is False
        assert "launch_preference_training.py" not in combined
        assert "train_preference.py" not in combined
        assert "train_dpo.py" not in combined


def test_shared_setup_cells_are_consistent():
    assert code(TRAIN, "preference-storage") == code(EVAL, "validation-storage")
    common = (ROOT / "scripts/bootstrap_cloud_source.py").read_bytes()
    for name, cell_id in ((TRAIN, "preference-source"), (EVAL, "validation-source")):
        source = code(name, cell_id)
        assert hashlib.sha256(common).hexdigest() in source
        assert "setup_preference_environment.py" in source
        assert "--sha256" in source
        assert "src/vqa_explanation/utils/process.py" in source
        assert "run_command([str(part) for part in command], cwd=REPO)" in source
    eval_data = textwrap.dedent(code(EVAL, "validation-data").split("else:\n", 1)[1])
    assert eval_data.split("ENVIRONMENT =", 1)[1] == (
        code(TRAIN, "preference-data").split("ENVIRONMENT =", 1)[1])


@pytest.mark.parametrize("available, explicit, expected", [
    ((True, False), False, 0),
    ((False, True), False, 1),
    ((True, True), True, 1),
    ((True, True), False, None),
    ((False, False), False, None),
])
def test_mount_finds_unique_project_without_a_fixed_owner_id(
    tmp_path, monkeypatch, available, explicit, expected,
):
    drive_root = tmp_path / "drive"
    mydrive = drive_root / "MyDrive/Vietnamese_VQA_NLE"
    shared = drive_root / ".shortcut-targets-by-id/owner/Vietnamese_VQA_NLE"
    for path, exists in zip((mydrive, shared), available, strict=True):
        if exists:
            marker = path / "reports/m1_selection/frozen_checkpoint3683_v2/selection.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}")
    mount_calls = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda *a, **kw: mount_calls.append((a, kw)))
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def path(value):
        return Path(value.replace("/content/drive", str(drive_root)))

    namespace = {"Path": path, "PROJECT_DIR": str(shared) if explicit else None}
    if expected is None:
        with pytest.raises(ValueError, match="unique match"):
            execute(TRAIN, "preference-storage", namespace)
    else:
        execute(TRAIN, "preference-storage", namespace)
        assert namespace["ROOT"] == (mydrive, shared)[expected]
    # Mount is unnecessary whenever MyDrive already exists.
    if not (drive_root / "MyDrive").is_dir():
        assert mount_calls == [(("/content/drive",), {"timeout_ms": 600000})]


def setup_record(tmp_path, name):
    storage = tmp_path / "storage"
    source = storage / "source"
    source.mkdir(parents=True)
    (source / "bootstrap_cloud_source.py").write_bytes(
        (ROOT / "scripts/bootstrap_cloud_source.py").read_bytes())
    repo = tmp_path / "source_repo"
    repo.mkdir()
    training = {
        "schema": "preference_training_launch_v1", "scope": "main_preference_experiment",
        "data_locked": True, "expected_pairs": 1000, "training_seed": 43,
        "source_release": "source_test", "source_sha256": "a" * 64,
        "run_group": "runs/seed43", "environment": "env.yaml", "study_config": "study.yaml",
        "m1_adapter": "runs/m1",
    }
    training_path = source / "training.json"
    training_path.write_text(json.dumps(training))
    training_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()
    record = {
        "schema": "preference_validation_launch_v1", "scope": "validation_only",
        "training_seed": 43, "questions_per_arm": 1407,
        "source_release": "source_eval", "source_sha256": "b" * 64,
        "training_launch": "source/training.json", "training_launch_sha256": training_hash,
        "run_group": training["run_group"], "output_dir": "validation/seed43",
        "m2_stage": "m2_train", "m3_stage": "m3_train_resume150",
    }
    record_path = source / "validation.json"
    record_path.write_text(json.dumps(record))
    key = "LAUNCH" if name == TRAIN else "VALIDATION"
    selected = training_path if name == TRAIN else record_path
    namespace = {
        "ROOT": storage, "Path": Path, "hashlib": hashlib, "json": json,
        "TASK": "validation",
        "re": __import__("re"), "sys": sys, "EXPECTED_SEED": 43,
        "INSPECTED": True,
        f"{key}_RECORD": selected.relative_to(storage).as_posix(),
        f"{key}_SHA256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }
    return namespace, repo, selected, training_path


@pytest.mark.parametrize("name, cell_id", [(TRAIN, "preference-source"),
                                          (EVAL, "validation-source")])
@pytest.mark.parametrize("fault", [None, "record_hash", "bootstrap", "unlocked", "seed",
                                  "release", "training_hash", "run"])
def test_source_setup_stops_changed_inputs_before_environment_install(
    tmp_path, name, cell_id, fault,
):
    namespace, repo, selected, training_path = setup_record(tmp_path, name)
    data = json.loads(selected.read_text())
    if fault == "record_hash":
        namespace["LAUNCH_SHA256" if name == TRAIN else "VALIDATION_SHA256"] = "0" * 64
    elif fault == "bootstrap":
        (namespace["ROOT"] / "source/bootstrap_cloud_source.py").write_text("changed")
    elif fault == "unlocked":
        changed = json.loads(training_path.read_text())
        changed["data_locked"] = False
        training_path.write_text(json.dumps(changed))
        if name == EVAL:
            data["training_launch_sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    elif fault == "seed":
        data["training_seed"] = 44
    elif fault == "release":
        data["source_release"] = "../other"
    elif fault == "training_hash" and name == EVAL:
        data["training_launch_sha256"] = "0" * 64
    elif fault == "run" and name == EVAL:
        data["run_group"] = "runs/wrong"
    if fault != "record_hash":
        if fault != "unlocked" or name != TRAIN:
            selected.write_text(json.dumps(data))
        key = "LAUNCH_SHA256" if name == TRAIN else "VALIDATION_SHA256"
        namespace[key] = hashlib.sha256(selected.read_bytes()).hexdigest()
    calls = []
    namespace["subprocess"] = SimpleNamespace(
        check_output=lambda command, **kw: (calls.append(command) or str(repo)),
        run=lambda command, **kw: calls.append(command),
    )
    namespace["runpy"] = SimpleNamespace(run_path=lambda path: {
        "run_command": lambda command, **kw: calls.append(command),
    })
    invalid = fault not in (None, "training_hash", "run") or (
        name == EVAL and fault in ("training_hash", "run"))
    if invalid:
        with pytest.raises(ValueError):
            execute(name, cell_id, namespace)
        assert calls == []
        assert namespace["INSPECTED"] is False
    else:
        execute(name, cell_id, namespace)
        assert namespace["REPO"] == repo
        assert len(calls) == 2
        assert calls[0][2] == "--archive"
        assert calls[1][1].endswith("setup_preference_environment.py")
        assert namespace["PYTHON"] == Path("/content/vqa_envs") / data["source_release"] / (
            "bin/python")
        assert all("train_preference.py" not in str(call) for call in calls)


@pytest.mark.parametrize("name, cell_id", [(TRAIN, "preference-data"), (EVAL, "validation-data")])
@pytest.mark.parametrize("ready, changed", [(True, False), (False, False), (True, True)])
def test_dataset_restore_uses_profile_and_checks_hash(tmp_path, name, cell_id, ready, changed):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}")
    annotations, checksums = tmp_path / "images/all.jsonl", tmp_path / "images/checksums.csv"
    if ready:
        annotations.parent.mkdir()
        annotations.write_text("{}")
        checksums.write_text("fixture")
    calls = []
    child_environments = []

    def read_paths(*args, **kwargs):
        child_environments.append(kwargs["env"])
        return json.dumps({"annotations": str(annotations), "image_checksums": str(checksums),
                           "image_root": str(tmp_path / "images")})

    namespace = {
        "ROOT": tmp_path, "REPO": tmp_path, "PYTHON": tmp_path / "env/bin/python",
        "TASK": "validation",
        "Path": Path, "json": json, "hashlib": hashlib, "os": os,
        "run": lambda cmd: calls.append(cmd),
        "launch": {
            "environment": "env.yaml", "dataset_bundle": "bundle",
            "dataset_bundle_sha256": "0" * 64 if changed else hashlib.sha256(b"{}").hexdigest(),
            "preference_archive": "preference.zip", "preference_archive_sha256": "a" * 64,
            "preference_root": "preferences",
        },
        "subprocess": SimpleNamespace(check_output=read_paths), "INSPECTED": True,
    }
    if changed:
        with pytest.raises(ValueError, match="Dataset bundle changed"):
            execute(name, cell_id, namespace)
        assert not any("verify_coco_bundle.py" in str(c) for c in calls)
    else:
        execute(name, cell_id, namespace)
        extracts = [c for c in calls if "verify_coco_bundle.py" in str(c)]
        assert len(extracts) == int(not ready)
        if extracts:
            assert extracts[0][-1] == str(tmp_path / "images")
    assert namespace["INSPECTED"] is False
    assert "PYTHONPATH" not in child_environments[0]
    assert "PYTHONHOME" not in child_environments[0]
    assert child_environments[0]["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize("arm", ["m2", "m3"])
@pytest.mark.parametrize("mode", ["gate", "train"])
@pytest.mark.parametrize("inspected, requested", [(False, True), (True, False), (True, True)])
def test_training_cells_call_existing_launcher_only_after_inspection(
    tmp_path, arm, mode, inspected, requested,
):
    calls = []
    namespace = {
        "INSPECTED": inspected, "RUN_GATE": requested, "RUN_TRAINING": requested,
        "run": lambda cmd: calls.append(cmd), "launch_command": launch_command,
        "REPO": ROOT, "ROOT": tmp_path, "LAUNCH_PATH": tmp_path / "launch.json",
        "RUN_FOLDER": tmp_path / "runs/seed44", "PYTHON": tmp_path / "env/bin/python",
        "RESUME": {"m2": None, "m3": None}, "ATTEMPT": {"m2": "initial", "m3": "initial"},
    }
    if not inspected:
        with pytest.raises(RuntimeError, match="inspection"):
            execute(TRAIN, f"preference-{arm}-{mode}", namespace)
    else:
        execute(TRAIN, f"preference-{arm}-{mode}", namespace)
        assert len(calls) == int(requested)
        if calls:
            command = calls[0]
            assert command[0] == str(namespace["PYTHON"])
            assert command[command.index("--arm") + 1] == arm
            assert command[command.index("--mode") + 1] == mode
            assert command[command.index("--output-dir") + 1].endswith(f"{arm}_{mode}")
            assert ("--gpu-gate" in command) == (mode == "train")


def test_resume_preserves_original_output_and_requires_new_attempt(tmp_path):
    calls = []
    original = tmp_path / "runs/m3_train"
    original.mkdir(parents=True)
    evidence = original / "result.json"
    evidence.write_text('{"status": "interrupted"}')
    namespace = {
        "INSPECTED": True, "RUN_TRAINING": True,
        "run": lambda cmd: calls.append(cmd), "launch_command": launch_command,
        "REPO": ROOT, "ROOT": tmp_path, "LAUNCH_PATH": tmp_path / "launch.json",
        "RUN_FOLDER": tmp_path / "runs", "PYTHON": tmp_path / "env/bin/python",
        "RESUME": {"m3": original / "training/checkpoint-150"},
        "ATTEMPT": {"m3": "initial"},
    }
    with pytest.raises(ValueError, match="new attempt"):
        execute(TRAIN, "preference-m3-train", namespace)
    namespace["ATTEMPT"]["m3"] = "resume150"
    execute(TRAIN, "preference-m3-train", namespace)
    assert calls[0][calls[0].index("--output-dir") + 1].endswith("m3_train_resume150")
    assert evidence.read_text() == '{"status": "interrupted"}'


def test_failed_inspection_revokes_previous_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    def failed(command):
        raise subprocess.CalledProcessError(1, command)
    namespace = {
        "REPO": ROOT, "LAUNCH_PATH": tmp_path / "launch.json", "ROOT": tmp_path,
        "RUN_FOLDER": tmp_path / "run", "PYTHON": tmp_path / "env/bin/python",
        "sys": sys, "INSPECTED": True, "run": failed,
    }
    with pytest.raises(subprocess.CalledProcessError):
        execute(TRAIN, "preference-preflight", namespace)
    assert namespace["INSPECTED"] is False


@pytest.mark.parametrize("use_baseline", [False, True])
def test_validation_is_independent_and_selects_declared_resume(tmp_path, use_baseline):
    calls = []
    namespace = {
        "PYTHON": tmp_path / "eval/bin/python", "REPO": ROOT, "ROOT": tmp_path,
        "TASK": "validation",
        "json": json,
        "ENVIRONMENT": tmp_path / "env.yaml", "OUTPUT": tmp_path / "validation",
        "launch": {"study_config": "study43.yaml", "m1_adapter": "m1"},
        "record": {"run_group": "runs/seed43", "m2_stage": "m2_train",
                   "m3_stage": "m3_train_resume150"},
        "run": lambda cmd: calls.append(cmd),
    }
    if use_baseline:
        namespace["record"].update({"baseline_source": "evaluation/verified_baseline",
                                     "baseline_sha256": {"m0.jsonl": "a" * 64}})
    execute(EVAL, "validation-inspect", namespace)
    command = calls[0]
    assert command[-1] == "--inspect"
    assert command[command.index("--m3-stage") + 1] == "m3_train_resume150"
    assert command[command.index("--study-config") + 1].endswith("study43.yaml")
    assert ("--baseline-source" in command) is use_baseline
    if use_baseline:
        assert command[command.index("--baseline-source") + 1] == str(
            tmp_path / "evaluation/verified_baseline")
        assert json.loads(command[command.index("--baseline-sha256") + 1]) == {
            "m0.jsonl": "a" * 64}
    namespace["RUN_EXPORT"] = False
    execute(EVAL, "validation-export", namespace)
    assert len(calls) == 1
    namespace["RUN_EXPORT"] = True
    execute(EVAL, "validation-export", namespace)
    assert calls[1] == command[:-1]


def test_validation_template_requires_explicit_record_before_source_setup(tmp_path):
    namespace = {"ROOT": tmp_path}
    execute(EVAL, "validation-config", namespace)
    assert namespace["VALIDATION_RECORD"] == namespace["VALIDATION_SHA256"] == ""
    with pytest.raises(ValueError, match="Select a validation launch"):
        execute(EVAL, "validation-source", namespace)
    assert namespace["INSPECTED"] is False


def test_validation_failure_does_not_enable_export():
    def failed(command):
        raise subprocess.CalledProcessError(1, command)
    namespace = {
        "PYTHON": Path("python"), "REPO": ROOT, "ROOT": ROOT,
        "TASK": "validation",
        "ENVIRONMENT": Path("env.yaml"), "OUTPUT": Path("validation"),
        "launch": {"study_config": "study.yaml", "m1_adapter": "m1"},
        "record": {"run_group": "run", "m2_stage": "m2_train", "m3_stage": "m3_train"},
        "run": failed, "INSPECTED": True,
    }
    with pytest.raises(subprocess.CalledProcessError):
        execute(EVAL, "validation-inspect", namespace)
    assert namespace["INSPECTED"] is False
    namespace["RUN_EXPORT"] = True
    with pytest.raises(RuntimeError, match="inspection"):
        execute(EVAL, "validation-export", namespace)


def test_semantic_task_does_not_restore_images_or_invoke_inference(tmp_path):
    annotations = tmp_path / "reports/m1_selection/frozen_checkpoint3683_v2/validation.jsonl"
    annotations.parent.mkdir(parents=True)
    annotations.write_text("frozen")
    calls = []
    namespace = {
        "TASK": "semantic", "ROOT": tmp_path, "REPO": ROOT,
        "SEMANTIC_OUTPUT": "evaluation/semantic44", "PYTHON": tmp_path / "env/bin/python",
        "OUTPUT": tmp_path / "validation44", "run": lambda cmd: calls.append(cmd),
    }
    execute(EVAL, "validation-data", namespace)
    assert calls == [] and namespace["ANNOTATIONS"] == annotations
    execute(EVAL, "validation-inspect", namespace)
    assert namespace["INSPECTED"] is True
    assert "run_study_semantic.py" in calls[0][2]
    assert calls[0][-1] == "--inspect"
    assert not any("run_study_validation.py" in str(c) for c in calls)


def test_semantic_missing_annotations_stops_without_using_training_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="validation annotations"):
        execute(EVAL, "validation-data", {
            "TASK": "semantic", "ROOT": tmp_path, "SEMANTIC_OUTPUT": "semantic"})


def test_results_cell_reads_without_modifying_outputs(tmp_path):
    report = tmp_path / "m3_train_resume150/training_result.json"
    report.parent.mkdir()
    report.write_text('{"status":"passed","global_step":375,"epoch":3}')
    before = report.read_bytes()
    execute(TRAIN, "preference-results", {
        "RUN_FOLDER": tmp_path, "ATTEMPT": {"m2": "initial", "m3": "resume150"},
        "stage_output": stage_output, "json": json,
    })
    assert report.read_bytes() == before


def test_pinned_training_record_default_matches_local_record_if_present():
    cases = (
        (TRAIN, "preference-config", "LAUNCH", "seed44_20261017",
         "preference_training_seed44_20261017.json"),
    )
    for name, config, key, folder, filename in cases:
        namespace = {}
        execute(name, config, namespace)
        record = ROOT / "artifacts/preference_v2" / folder / filename
        if record.is_file():
            assert hashlib.sha256(record.read_bytes()).hexdigest() == namespace[f"{key}_SHA256"]


@pytest.mark.historical
def test_archived_notebook_preserves_original_bytes():
    path = ROOT / "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "83972dbdae24e32b7edfda204b156b1aa988c2f52387e53f4d5adacc8124863d")


@pytest.mark.historical
def test_local_notebook_history_remains_unchanged_when_available():
    manifest = ROOT / "artifacts/notebook_cleanup_20261017/before.json"
    if manifest.is_file():
        for entry in json.loads(manifest.read_text("utf-8")):
            path = Path(entry["path"])
            if path == ROOT / "notebooks/04_train_preferences.ipynb":
                path = ROOT / "artifacts/notebook_cleanup_20261017/04_before.ipynb"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
