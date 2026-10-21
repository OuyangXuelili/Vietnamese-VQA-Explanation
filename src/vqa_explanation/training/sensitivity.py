"""Check the one-variable LR comparison without changing the trainer."""

import json
from copy import deepcopy
from pathlib import Path

import yaml

from vqa_explanation.training.preference_data import load_config
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def validate_sensitivity_launch(record, project_dir, study_path):
    study = yaml.safe_load(Path(study_path).read_text("utf-8"))
    if "sensitivity" not in study and "study_role" not in record:
        return None
    if record.get("study_role") != "learning_rate_sensitivity":
        raise ValueError("Sensitivity study requires its own declared launch role")
    spec = study.get("sensitivity", {})
    if (spec.get("parameter") != "learning_rate" or spec.get("baseline_value") != 1e-5
            or spec.get("value") != 5e-6 or study.get("seed") != 42):
        raise ValueError("Require the declared seed-42 LR 1e-5 versus 5e-6 comparison")
    root = project_root()
    parent_path = root / "configs/experiments/preference_study_v2.yaml"
    if (spec.get("parent_study") != parent_path.relative_to(root).as_posix()
            or sha256_file(parent_path) != spec.get("parent_study_sha256")):
        raise ValueError("Sensitivity parent study changed")
    parent = yaml.safe_load(parent_path.read_text("utf-8"))
    expected = deepcopy(parent)
    expected.update(declared_on=study["declared_on"], planned_seeds=[42], sensitivity=spec)
    expected["training"].update(
        learning_rate=5e-6, m2_config="configs/training/sft_chosen_lr5e6.yaml",
        m3_config="configs/training/dpo_lr5e6.yaml")
    if study != expected:
        raise ValueError("Sensitivity study changed a factor other than learning rate")
    for arm in ("m2", "m3"):
        baseline = load_config(root / parent["training"][f"{arm}_config"])
        current = load_config(root / study["training"][f"{arm}_config"])
        if current != {**baseline, "learning_rate": 5e-6}:
            raise ValueError(f"Sensitivity {arm} changed a factor other than learning rate")
        if record.get(f"{arm}_config") != study["training"][f"{arm}_config"]:
            raise ValueError("Sensitivity launch selects different arm configs")
    relative = Path(record.get("parent_training_launch", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Parent launch must stay within the project")
    parent_launch = Path(project_dir) / relative
    if not parent_launch.resolve().is_relative_to(Path(project_dir).resolve()):
        raise ValueError("Parent launch must stay within the project")
    if sha256_file(parent_launch) != record["parent_training_launch_sha256"]:
        raise ValueError("Sensitivity parent training launch changed")
    original = json.loads(parent_launch.read_text("utf-8"))
    fields = (
        "schema", "scope", "data_locked", "expected_pairs", "preferences", "preference_root",
        "preference_archive", "preference_archive_sha256", "m1_adapter", "m1_selection",
        "environment", "dataset_bundle", "dataset_bundle_sha256",
        "preference_decision_record", "preference_decision_record_sha256",
    )
    if (original.get("study_config") != spec["parent_study"]
            or original.get("training_seed", 42) != 42
            or record.get("training_seed") != 42
            or any(key not in record or key not in original
                   or record[key] != original[key] for key in fields)):
        raise ValueError("Sensitivity launch changed the baseline data or M1 inputs")
    if (record["run_group"] == original["run_group"]
            or not record["run_group"].endswith("_seed42_lr5e6")):
        raise ValueError("Sensitivity needs its own seed42_lr5e6 run group")
    return {"role": record["study_role"], "parent_study_sha256": sha256_file(parent_path),
            "parent_training_launch_sha256": sha256_file(parent_launch),
            "learning_rates": [1e-5, 5e-6]}
