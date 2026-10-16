"""The active notebook is a small launcher, with training disabled by default."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_has_one_setup_and_two_short_arm_cells():
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    assert [cell["id"] for cell in notebook["cells"]] == [
        "preference-intro", "preference-storage", "preference-source", "preference-preflight",
        "preference-m2", "preference-m3", "preference-results",
        "preference-validation-intro", "preference-validation-source",
        "preference-validation-inspect", "preference-validation-export",
        "preference-expansion-intro", "preference-expansion-storage", "preference-expansion-setup",
        "preference-expansion-inspect", "preference-expansion-run",
        "preference-v2-intro", "preference-v2-setup", "preference-v2-preflight",
        "preference-v2-m2-gate", "preference-v2-m2-train",
        "preference-v2-m3-gate", "preference-v2-m3-train", "preference-v2-results",
        "preference-v2-validation-intro", "preference-v2-validation-setup",
        "preference-v2-validation-inspect", "preference-v2-validation-export",
        "preference-image-check-intro", "preference-image-check-setup",
        "preference-image-check-run",
        "preference-seed43-intro", "preference-seed43-setup",
        "preference-seed43-m2", "preference-seed43-m3",
    ]
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"}
    for source in code.values():
        ast.parse(source)
    assert sum(source.count("drive.mount(") for source in code.values()) == 1
    assert "MODE = 'inspect'" in code["preference-preflight"]
    assert "launch['run_group']" in code["preference-preflight"]
    assert "datetime" not in code["preference-preflight"]
    assert "ATTEMPT = {'m2': 'initial', 'm3': 'initial'}" in code["preference-preflight"]
    assert "stage_output(RUN_FOLDER, arm, MODE, ATTEMPT[arm])" in code["preference-results"]
    assert "setup_preference_environment.py" in code["preference-source"]
    assert "pip', 'install'" not in code["preference-source"]
    assert "python_executable=PYTHON" in code["preference-preflight"]
    assert len(code["preference-m2"].splitlines()) == 3
    assert len(code["preference-m3"].splitlines()) == 3
    assert "train_sft_full" not in "\n".join(code.values())
    assert "generate_preference_candidates" not in "\n".join(code.values())
    assert "run_study_validation.py" in code["preference-validation-source"]
    assert "['--inspect']" in code["preference-validation-inspect"]
    assert len(code["preference-validation-export"].splitlines()) == 1
    expansion = code["preference-expansion-setup"]
    assert "preference_expansion_v2.json" in expansion
    assert "--sha256" in expansion and "EXPANSION_LAUNCH['metadata_sha256']" in expansion
    assert "'--target', '1000'" in expansion
    assert "train_preference" not in expansion
    assert "ITEM_LIMIT = 20" in code["preference-expansion-run"]
    assert "['--inspect']" in code["preference-expansion-inspect"]
    assert "expected_pairs') == 1000" in code["preference-v2-setup"]
    assert "bootstrap_sha256" in code["preference-v2-setup"]
    assert "preference_decision_record" in code["preference-v2-preflight"] or (
        "launch_command" in code["preference-v2-preflight"]
    )
    assert "'m2', 'gate'" in code["preference-v2-m2-gate"]
    assert "'m2', 'train'" in code["preference-v2-m2-train"]
    assert "'m3', 'gate'" in code["preference-v2-m3-gate"]
    assert "'m3', 'train'" in code["preference-v2-m3-train"]
    validation = code["preference-v2-validation-setup"]
    assert "preference_validation_v2_resume01.json" in validation
    assert "training_launch_sha256" in validation and "'--sha256'" in validation
    assert "'--study-config'" in validation and "run_study_validation.py" in validation
    assert "setup_preference_environment.py" in validation
    assert "EVAL_ENV_V2" in validation and "PYTHON_V2 =" not in validation
    assert "['--inspect']" in code["preference-v2-validation-inspect"]
    assert len(code["preference-v2-validation-export"].splitlines()) == 1
    image_setup = code["preference-image-check-setup"]
    assert "729a89c1d3f61dc8df67376c40f5fda0717b7a96b7bcd26e806a5066526c8f63" in image_setup
    assert "prepare_validation(*sys.argv[1:5],study_config=sys.argv[5])" in image_setup
    assert "setup_preference_environment.py" in image_setup
    assert "torch.cuda.is_available()" in image_setup
    image_run = code["preference-image-check-run"]
    assert "('m1', 'M1_SFT'), ('m2', 'M2_SFT_CHOSEN'), ('m3', 'M3_DPO')" in image_run
    assert "'run'" in image_run and "train_preference" not in image_run
    assert "image_evidence['adapters'][arm]" in image_run
    setup43 = code["preference-seed43-setup"]
    assert "replication['training_seed'] == 43" in setup43
    assert "REPLICATION_RUN.exists()" in setup43
    assert "parent_record.read_bytes()" in setup43
    assert "project_root()) == sys.argv[1]" in setup43
    assert "'--mode', 'inspect'" in setup43
    for arm in ('m2', 'm3'):
        replica = code[f"preference-seed43-{arm}"]
        assert "('gate', 'train')" in replica
        assert "'--gpu-gate'" in replica
        assert "resume" not in replica
    digest = hashlib.sha256((ROOT / "scripts/bootstrap_cloud_source.py").read_bytes()).hexdigest()
    assert digest in code["preference-source"]
    assert all(not cell["outputs"] and cell["execution_count"] is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_image_check_manifest_hash_is_portable_without_reselecting_images():
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["id"] == "preference-image-check-setup")
    expected = next(ast.literal_eval(node.value) for node in ast.parse(source).body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "IMAGE_BUNDLE_SHA"
                            for target in node.targets))
    assert ".replace(b'\\r\\n', b'\\n')" in source
    # The frozen artifact is optional in a source-only checkout.
    manifest = ROOT / "artifacts/image_dependence/validation_200_seed42/manifest.json"
    if manifest.is_file():
        payload = manifest.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
        assert hashlib.sha256(payload.replace(b"\n", b"\r\n").replace(
            b"\r\n", b"\n")).hexdigest() == expected


def test_archived_notebook_preserves_its_exact_original_bytes():
    path = ROOT / "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "83972dbdae24e32b7edfda204b156b1aa988c2f52387e53f4d5adacc8124863d")


def test_storage_setup_allows_time_for_auth_and_checks_launch_record():
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "mount"]
    assert len(calls) == 1
    assert {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords} == {
        "timeout_ms": 600000,
    }
    assert "if not LAUNCH_PATH.is_file():" in source
    assert "print('Project storage ready:', ROOT)" in source


@pytest.mark.parametrize("available, expected", [
    ((True, True), 0), ((True, False), 0), ((False, True), 1), ((False, False), None),
])
def test_main_setup_selects_storage_with_m1_evidence(tmp_path, available, expected):
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["id"] == "preference-v2-setup")
    block = source[source.index("PROJECT_ROOTS ="):source.index("LAUNCH_PATH_V2 =")]
    folders = (tmp_path / "shared", tmp_path / "mydrive")
    for folder, present in zip(folders, available, strict=True):
        folder.mkdir()
        if present:
            selection = folder / "reports/m1_selection/frozen_checkpoint3683_v2/selection.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
    namespace = {"Path": lambda value: folders[0 if ".shortcut-targets-by-id/" in value else 1]}
    if expected is None:
        with pytest.raises(FileNotFoundError, match="frozen M1 selection"):
            exec(compile(block, "project-storage", "exec"), namespace)
    else:
        exec(compile(block, "project-storage", "exec"), namespace)
        assert namespace["ROOT"] == folders[expected]


@pytest.mark.parametrize("locked", [False, True])
def test_data_setup_is_conditional_and_precedes_inspection(tmp_path, locked):
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({
        "annotations": str(tmp_path / "dataset/metadata/all.jsonl"),
        "image_root": str(tmp_path / "dataset"), "run_root": str(tmp_path / "runs"),
    }), encoding="utf-8")
    calls = []

    class Commands:
        @staticmethod
        def run(command, **kwargs):
            calls.append(command)

    namespace = {
        "ROOT": tmp_path, "REPO": tmp_path, "LAUNCH_PATH": tmp_path / "launch.json",
        "hashlib": hashlib, "run_command": Commands.run, "PYTHON": tmp_path / "env/python",
        "launch": {"data_locked": locked, "environment": str(environment),
                   "run_group": "runs/main", "dataset_bundle": "bundle",
                   "dataset_bundle_sha256": hashlib.sha256(b"{}").hexdigest()},
    }
    exec(compile(source, "preference-preflight", "exec"), namespace)
    assert len(calls) == (3 if locked else 2)
    if locked:
        assert calls[0][1].endswith("verify_coco_bundle.py") and "--extract-to" in calls[0]
    assert all("inspect" in command for command in calls[-2:])


def test_wrong_bundle_hash_blocks_extraction_and_inspection(tmp_path):
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({
        "annotations": str(tmp_path / "dataset/metadata/all.jsonl"),
        "image_root": str(tmp_path / "dataset"), "run_root": str(tmp_path / "runs"),
    }), encoding="utf-8")
    namespace = {
        "ROOT": tmp_path, "REPO": tmp_path, "hashlib": hashlib,
        "launch": {"data_locked": True, "environment": str(environment),
                   "run_group": "runs/main", "dataset_bundle": "bundle",
                   "dataset_bundle_sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="Dataset bundle changed"):
        exec(compile(source, "preference-preflight", "exec"), namespace)


@pytest.mark.parametrize("images_ready, changed", [
    (True, None), (False, None), (False, "bundle"), (True, "run"),
])
def test_main_validation_recovers_without_training_setup(
    tmp_path, monkeypatch, images_ready, changed,
):
    from vqa_explanation.utils import process

    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["id"] == "preference-v2-validation-setup")
    storage, repo = tmp_path / "shared", tmp_path / "source"
    source = source.replace(
        "/content/drive/.shortcut-targets-by-id/17bcTXATT45J5LEmtruMRs1ddwRY5kBwc/Vietnamese_VQA_NLE",
        storage.as_posix(),
    ).replace("/content/drive/MyDrive/Vietnamese_VQA_NLE", (tmp_path / "unused").as_posix())
    selection = storage / "reports/m1_selection/frozen_checkpoint3683_v2/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}")
    (storage / "source").mkdir()
    (storage / "source/bootstrap_cloud_source.py").write_bytes(
        (ROOT / "scripts/bootstrap_cloud_source.py").read_bytes())
    dataset = storage / "dataset"
    dataset.mkdir()
    (dataset / "bundle.json").write_text("{}")
    bundle_hash = hashlib.sha256(b"{}").hexdigest()
    launch = {
        "data_locked": True, "expected_pairs": 1000, "run_group": "runs/main",
        "environment": "environment.yaml", "study_config": "study.yaml",
        "m1_adapter": "runs/m1", "dataset_bundle": "dataset",
        "dataset_bundle_sha256": "changed" if changed == "bundle" else bundle_hash,
    }
    launch_bytes = json.dumps(launch).encode()
    (storage / "source/preference_training_launch_v2_20261015.json").write_bytes(launch_bytes)
    record = {
        "schema": "preference_validation_launch_v1",
        "training_launch_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "run_group": "runs/other" if changed == "run" else launch["run_group"],
        "source_release": "source_fixed", "source_sha256": "fixed",
        "output_dir": "evaluation/validation",
    }
    (storage / "source/preference_validation_v2_resume01.json").write_text(json.dumps(record))
    repo.mkdir()
    annotations = tmp_path / "images/metadata/all.jsonl"
    if images_ready:
        annotations.parent.mkdir(parents=True)
        annotations.write_text("{}")
    (repo / "environment.yaml").write_text(yaml.safe_dump({
        "annotations": str(annotations), "image_root": str(tmp_path / "images"),
        "run_root": str(storage / "runs"),
    }))
    calls = []
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(repo))
    monkeypatch.setattr(process, "run_command", lambda command, **kwargs: calls.append(command))
    namespace = {}
    if changed:
        with pytest.raises(ValueError, match="Dataset bundle changed" if changed == "bundle"
                           else "different training run"):
            exec(compile(source, "validation-recovery", "exec"), namespace)
        return
    exec(compile(source, "validation-recovery", "exec"), namespace)
    assert len(calls) == (1 if images_ready else 2)
    assert calls[0][1].endswith("setup_preference_environment.py")
    assert "source_fixed" in calls[0][-1]
    if not images_ready:
        assert calls[1][1].endswith("verify_coco_bundle.py")
    assert namespace["RUN_FOLDER_V2"] == storage / "runs/main"
    assert "--study-config" in namespace["EVAL_COMMAND_V2"]
    assert not {"REPO_V2", "PYTHON_V2", "launch_command"} & namespace.keys()
