import copy
from pathlib import Path

import pytest
import yaml

from vqa_explanation.training.preference_data import load_config, validate_study_arm

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["sft_chosen", "dpo"])
@pytest.mark.parametrize("seed", [43, 44])
def test_replication_changes_only_training_seed(name, seed):
    baseline = load_config(ROOT / f"configs/training/{name}_full.yaml")
    path = ROOT / f"configs/training/{name}_seed{seed}.yaml"
    cfg = load_config(path)
    expected = {**baseline, "seed": seed}
    assert cfg == expected
    study = ROOT / f"configs/experiments/preference_study_seed{seed}.yaml"
    comparison = validate_study_arm(path, study)
    assert comparison["effective_batch"] == 8


@pytest.mark.parametrize("seed,date", [(43, "2026-10-16"), (44, "2026-10-17")])
def test_replication_preserves_data_budget_and_evaluation_protocol(seed, date):
    base = yaml.safe_load((ROOT / "configs/experiments/preference_study_v2.yaml").read_text())
    path = ROOT / f"configs/experiments/preference_study_seed{seed}.yaml"
    replica = yaml.safe_load(path.read_text())
    expected = copy.deepcopy(base)
    expected.update(seed=seed, declared_on=date)
    expected["training"].update(m2_config=f"configs/training/sft_chosen_seed{seed}.yaml",
                                m3_config=f"configs/training/dpo_seed{seed}.yaml")
    assert replica == expected
