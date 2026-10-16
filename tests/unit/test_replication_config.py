import copy
from pathlib import Path

import pytest
import yaml

from vqa_explanation.training.preference_data import load_config, validate_study_arm

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["sft_chosen", "dpo"])
def test_replication_changes_only_training_seed(name):
    baseline = load_config(ROOT / f"configs/training/{name}_full.yaml")
    path = ROOT / f"configs/training/{name}_seed43.yaml"
    cfg = load_config(path)
    expected = {**baseline, "seed": 43}
    assert cfg == expected
    study = ROOT / "configs/experiments/preference_study_seed43.yaml"
    comparison = validate_study_arm(path, study)
    assert comparison["effective_batch"] == 8


def test_replication_preserves_data_budget_and_evaluation_protocol():
    base = yaml.safe_load((ROOT / "configs/experiments/preference_study_v2.yaml").read_text())
    path = ROOT / "configs/experiments/preference_study_seed43.yaml"
    replica = yaml.safe_load(path.read_text())
    expected = copy.deepcopy(base)
    expected.update(seed=43, declared_on="2026-10-16")
    expected["training"].update(m2_config="configs/training/sft_chosen_seed43.yaml",
                                m3_config="configs/training/dpo_seed43.yaml")
    assert replica == expected
