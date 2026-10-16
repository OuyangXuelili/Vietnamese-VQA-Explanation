import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vqa_explanation.utils.environment import load_environment, project_root


@pytest.mark.parametrize(
    ("filename", "field", "expected"),
    [
        ("vivqax_smoke.yaml", "image_archive", "data/cache/coco_smoke_500.zip"),
        ("vivqax.yaml", "preference_root", "data/preferences"),
    ],
)
def test_local_data_storage_paths(filename, field, expected):
    config_path = project_root() / "configs/data" / filename
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dataset"][field] == expected


def test_local_paths_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = load_environment(Path("configs/environments/local.yaml"))
    assert paths["image_root"] == project_root() / "data/raw/coco"
    assert paths["run_root"] == project_root() / "runs"


def test_environment_supports_external_data(tmp_path):
    config = tmp_path / "env.yaml"
    config.write_text(
        f"annotations: {tmp_path.as_posix()}/a.jsonl\nimage_root: images\nrun_root: runs\n"
    )
    assert load_environment(config, tmp_path)["annotations"] == tmp_path / "a.jsonl"


def test_environment_supports_image_checksum_manifest(tmp_path):
    config = tmp_path / "env.yaml"
    config.write_text(
        "annotations: a.jsonl\nimage_root: images\nrun_root: runs\n"
        "image_checksums: checksums.csv\n",
        encoding="utf-8",
    )
    assert load_environment(config, tmp_path)["image_checksums"] == tmp_path / "checksums.csv"


def test_environment_rejects_unexpanded_variable(tmp_path):
    config = tmp_path / "env.yaml"
    config.write_text("annotations: ${UNSET_VQA_TEST_VAR}/a\nimage_root: images\nrun_root: runs\n")
    with pytest.raises(ValueError, match="Unresolved"):
        load_environment(config, tmp_path)


def test_kaggle_paths_are_explicit_and_have_no_drive_dependency(tmp_path, monkeypatch):
    monkeypatch.setenv("VQA_ANNOTATIONS", str(tmp_path / "metadata/all.jsonl"))
    monkeypatch.setenv("VQA_IMAGE_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("VQA_IMAGE_CHECKSUMS", str(tmp_path / "metadata/checksums.csv"))
    paths = load_environment(Path("configs/environments/kaggle.example.yaml"))
    assert paths["annotations"] == tmp_path / "metadata/all.jsonl"
    assert paths["image_root"] == tmp_path / "data"
    assert "vqa_runs" in str(paths["run_root"])
    assert all("drive/MyDrive" not in str(path) for path in paths.values())


def test_dry_run_works_outside_repo_and_creates_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(project_root() / "scripts/run_pipeline.py"), "preflight", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-only" in result.stdout
    assert "/content/" not in result.stdout
    assert not list(tmp_path.iterdir())


def test_full_preflight_dry_run_checks_all_training_records(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "preflight",
            "--environment",
            "configs/environments/local_full.yaml",
            "--all-train-records",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-only" in result.stdout
    assert "--all-train-records" in result.stdout
    assert "image_disjoint_v1" in result.stdout
    assert not list(tmp_path.iterdir())


def test_all_training_records_cannot_be_used_for_smoke_training(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "sft-smoke",
            "--all-train-records",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only valid for preflight" in result.stderr
