import json
import shutil
from copy import deepcopy

import pytest
import yaml

from vqa_explanation.training import sensitivity
from vqa_explanation.training.preference_data import validate_study_arm
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    source = sensitivity.project_root()
    for relative in (
        "configs/experiments/preference_study_v2.yaml",
        "configs/experiments/preference_study_lr5e6.yaml",
        "configs/training/sft_chosen_full.yaml", "configs/training/dpo_full.yaml",
        "configs/training/sft_chosen_lr5e6.yaml", "configs/training/dpo_lr5e6.yaml",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    parent = {key: key for key in (
        "preferences", "preference_root", "preference_archive", "preference_archive_sha256",
        "m1_adapter", "m1_selection", "environment", "dataset_bundle", "dataset_bundle_sha256",
        "preference_decision_record", "preference_decision_record_sha256",
    )}
    parent.update(schema="preference_training_launch_v1", scope="main_preference_experiment",
                  data_locked=True, expected_pairs=1000, run_group="runs/main_seed42",
                  study_config="configs/experiments/preference_study_v2.yaml")
    parent_path = tmp_path / "parent_launch.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    study = tmp_path / "configs/experiments/preference_study_lr5e6.yaml"
    config = yaml.safe_load(study.read_text("utf-8"))
    record = {**parent, "study_config": study.relative_to(tmp_path).as_posix(),
              "study_role": "learning_rate_sensitivity", "training_seed": 42,
              "parent_training_launch": parent_path.name,
              "parent_training_launch_sha256": sha256_file(parent_path),
              "run_group": "runs/comparison_seed42_lr5e6",
              **{f"{a}_config": config["training"][f"{a}_config"] for a in ("m2", "m3")}}
    monkeypatch.setattr(sensitivity, "project_root", lambda: tmp_path)
    return record, tmp_path, study


def test_lr_comparison_identifies_the_declared_parameter(comparison):
    result = sensitivity.validate_sensitivity_launch(*comparison)
    assert result["learning_rates"] == [1e-5, 5e-6]
    assert result["role"] == "learning_rate_sensitivity"


@pytest.mark.parametrize("fault", ["role", "seed", "m1", "data", "old_run", "parent_hash",
                                  "epochs", "quality", "beta", "length", "lr", "arm"])
def test_changed_factor_stops_the_comparison(comparison, fault):
    record, root, study = comparison
    if fault in {"epochs", "quality"}:
        payload = yaml.safe_load(study.read_text("utf-8"))
        if fault == "epochs":
            payload["training"]["epochs_per_branch"] = 4
        else:
            payload["preferences"]["quality_review_sha256"] = "wrong"
        study.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif fault in {"beta", "length", "lr"}:
        path = root / "configs/training/dpo_lr5e6.yaml"
        cfg = yaml.safe_load(path.read_text("utf-8"))
        cfg[{"beta": "beta", "length": "preflight_max_length", "lr": "learning_rate"}[
            fault]] = {"beta": .2, "length": 2048, "lr": 1e-5}[fault]
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    else:
        key, value = {
            "role": ("study_role", "main"), "seed": ("training_seed", 43),
            "m1": ("m1_adapter", "another_adapter"), "data": ("preferences", "other_pairs"),
            "old_run": ("run_group", "runs/main_seed42"),
            "parent_hash": ("parent_training_launch_sha256", "wrong"),
            "arm": ("m2_config", "configs/training/sft_chosen_full.yaml"),
        }[fault]
        record[key] = value
    with pytest.raises(ValueError):
        sensitivity.validate_sensitivity_launch(record, root, study)


def test_main_config_is_not_rewritten_by_sensitivity_validation(comparison):
    _, root, _ = comparison
    path = root / "configs/experiments/preference_study_v2.yaml"
    original = path.read_bytes()
    assert sensitivity.validate_sensitivity_launch({}, root, path) is None
    assert path.read_bytes() == original


def test_declared_arms_preserve_all_non_lr_training_values():
    root = sensitivity.project_root()
    study = root / "configs/experiments/preference_study_lr5e6.yaml"
    baseline_path = root / "configs/experiments/preference_study_v2.yaml"
    baseline_study = yaml.safe_load(baseline_path.read_text("utf-8"))
    comparison_study = yaml.safe_load(study.read_text("utf-8"))
    for arm in ("m2", "m3"):
        baseline = sensitivity.load_config(root / baseline_study["training"][f"{arm}_config"])
        path = root / comparison_study["training"][f"{arm}_config"]
        changed = deepcopy(baseline)
        changed["learning_rate"] = 5e-6
        assert sensitivity.load_config(path) == changed
        protocol = validate_study_arm(path, study)
        assert protocol["effective_batch"] == 8
        assert (changed["expected_pairs"] // protocol["effective_batch"]
                * changed["num_train_epochs"]) == 375


def test_launcher_checks_sensitivity_before_reading_data(comparison, monkeypatch):
    from vqa_explanation.training import preference_launch

    record, root, study = comparison
    record["study_config"] = str(study)
    record["training_seed"] = 43
    path = root / "comparison_launch.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(preference_launch, "prepare_stage",
                        lambda *args: pytest.fail("Must stop before reading training data"))
    with pytest.raises(ValueError, match="data or M1"):
        preference_launch.prepare_launch(path, root, "m2", "inspect")
