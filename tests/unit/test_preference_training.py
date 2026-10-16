import copy
import json
import math
from types import SimpleNamespace

import pytest
import yaml

from vqa_explanation.training.preference_data import (
    PairCollator,
    adapter_fingerprint,
    load_config,
)


@pytest.mark.parametrize("name", ["sft_chosen", "dpo"])
def test_stage_configs_match_protocol(name):
    cfg = load_config(f"configs/training/{name}.yaml")
    study = yaml.safe_load(open("configs/experiments/study_budget_v2.yaml", encoding="utf-8"))
    assert cfg["expected_pairs"] == study["preferences"]["locked_pairs"] == 186
    assert cfg["gradient_accumulation_steps"] == 8
    assert cfg["dtype"] == "auto"


def test_main_plan_changes_only_lr_and_locked_pair_count():
    earlier = load_config("configs/archive/sft_chosen_lr2e5.yaml")
    matched = load_config("configs/training/sft_chosen.yaml")
    dpo = load_config("configs/training/dpo.yaml")
    assert earlier.keys() == matched.keys()
    assert {key for key in earlier if earlier[key] != matched[key]} == {
        "learning_rate", "expected_pairs"
    }
    assert matched["learning_rate"] == dpo["learning_rate"] == 0.000005


def test_main_configs_match_the_locked_reviewed_dataset():
    from pathlib import Path

    from vqa_explanation.training.preference_data import validate_study_arm

    path = Path("configs/experiments/preference_study_v2.yaml")
    study = yaml.safe_load(path.read_text(encoding="utf-8"))
    comparison = validate_study_arm("configs/training/dpo_full.yaml", path)
    assert comparison["matched_fields"]["num_train_epochs"] == 3
    assert comparison["matched_fields"]["expected_pairs"] == 1000
    assert study["preferences"]["data_locked"] is True
    assert study["preferences"]["locked_pairs"] == 1000
    assert study["preferences"]["frozen_bundle"].endswith("frozen_1000_reviewed_20261015")
    assert study["preferences"]["quality_review_sha256"] == (
        "5f8ec683d70d8d802515c5fb518555b848be165157ddc718380a6bb447dc4fdd"
    )
    assert study["preferences"]["required_review_version"] == 2
    assert study["preferences"]["require_visual_quality"] is True


@pytest.mark.parametrize("count,expected", [(186, 72), (500, 189), (1000, 375)])
def test_epoch_update_budget_matches_pinned_transformers(count, expected):
    from transformers import Trainer

    args = SimpleNamespace(max_steps=-1, gradient_accumulation_steps=8, num_train_epochs=3)
    trainer = SimpleNamespace(num_examples=lambda loader: len(loader))
    values = Trainer.set_initial_training_values(trainer, args, list(range(count)), 8)
    assert values[-1] == expected


@pytest.mark.parametrize("locked_pairs", [None, True, 0, 185])
def test_locked_study_rejects_missing_or_different_pair_count(tmp_path, locked_pairs):
    from pathlib import Path

    from vqa_explanation.training.preference_data import validate_study_arm

    study = yaml.safe_load(
        Path("configs/experiments/study_budget_v2.yaml").read_text(encoding="utf-8")
    )
    study["preferences"]["locked_pairs"] = locked_pairs
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study), encoding="utf-8")
    with pytest.raises(ValueError, match="locked_pairs|expected_pairs"):
        validate_study_arm("configs/training/sft_chosen.yaml", path)


def test_declared_main_comparison_and_experiment_point_to_matched_sft():
    from pathlib import Path

    from vqa_explanation.training.preference_data import validate_study_arm

    study_path = Path("configs/experiments/study_budget_v2.yaml")
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    arm = yaml.safe_load(Path("configs/experiments/m2_sft_chosen.yaml").read_text(encoding="utf-8"))
    assert arm["training_config"] == study["training"]["m2_config"]
    m2 = validate_study_arm(study["training"]["m2_config"], study_path)
    m3 = validate_study_arm(study["training"]["m3_config"], study_path)
    assert m2["protocol_id"] == m3["protocol_id"] == "vqa_nle_matched_lr_v2"
    assert m2["config_sha256"] == m3["config_sha256"]
    assert m2["matched_fields"]["learning_rate"] == 0.000005
    with pytest.raises(ValueError, match="not the declared m2"):
        validate_study_arm("configs/archive/sft_chosen_lr2e5.yaml", study_path)


@pytest.mark.parametrize("field,value", [
    ("learning_rate", 0.00002), ("seed", 43), ("expected_pairs", 999),
    ("max_pixels", 201488), ("preflight_max_length", 2048), ("dtype", "float16"),
])
def test_comparison_rejects_unmatched_arms_before_data_or_model_load(tmp_path, field, value):
    from pathlib import Path

    from vqa_explanation.training.preference_data import validate_study_arm

    sft = tmp_path / "sft.yaml"
    dpo = tmp_path / "dpo.yaml"
    sft.write_text(Path("configs/training/sft_chosen.yaml").read_text(encoding="utf-8"))
    changed = load_config("configs/training/dpo.yaml")
    changed[field] = value
    dpo.write_text(yaml.safe_dump(changed), encoding="utf-8")
    study = yaml.safe_load(
        Path("configs/experiments/study_budget_v2.yaml").read_text(encoding="utf-8")
    )
    study["training"].update(m2_config=str(sft), m3_config=str(dpo))
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study), encoding="utf-8")
    with pytest.raises(ValueError, match="Comparison arms differ"):
        validate_study_arm(sft, path)


@pytest.mark.parametrize("field,value", [
    ("seed", 43), ("learning_rate", 0.00002), ("epochs_per_branch", 2),
    ("effective_batch", 16), ("checkpoint_selection", "best_validation"),
])
def test_comparison_rejects_study_declarations_different_from_configs(tmp_path, field, value):
    from pathlib import Path

    from vqa_explanation.training.preference_data import validate_study_arm

    study = yaml.safe_load(
        Path("configs/experiments/study_budget_v2.yaml").read_text(encoding="utf-8")
    )
    if field == "seed":
        study[field] = value
    else:
        study["training"][field] = value
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study), encoding="utf-8")
    with pytest.raises(ValueError, match="Config differs from study"):
        validate_study_arm("configs/training/sft_chosen.yaml", path)


def test_comparison_cli_needs_no_dataset_or_gpu():
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "scripts/train_preference.py",
        "--config", "configs/training/sft_chosen.yaml", "--check-comparison",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["arm"] == "m2"
    result = subprocess.run([
        sys.executable, "scripts/train_preference.py",
        "--config", "configs/archive/sft_chosen_lr2e5.yaml", "--preflight-only",
        "--preferences", "missing", "--m1-adapter", "missing", "--m1-selection", "missing",
    ], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "not the declared m2 arm" in result.stderr
    assert "FileNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_length", 512),
        ("precompute_ref_log_probs", True),
        ("beta", float("nan")),
        ("disable_dropout", False),
        ("num_train_epochs", 0),
        ("num_train_epochs", True),
        ("num_train_epochs", 1.5),
        ("expected_pairs", 0),
        ("unknown", True),
        ("dtype", "float32"),
    ],
)
def test_bad_dpo_config_fails_before_model_load(tmp_path, field, value):
    cfg = yaml.safe_load(open("configs/training/dpo.yaml", encoding="utf-8"))
    cfg[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_adapter_fingerprint_covers_bytes_not_directory_name(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        adapter_fingerprint(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"old")
    first = adapter_fingerprint(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"new")
    assert adapter_fingerprint(tmp_path)["sha256"] != first["sha256"]


def test_pair_collator_retains_image_and_same_question_for_both_sides():
    calls = []

    def capture(rows):
        calls.append(rows[0])
        return {"image_grid_thw": "retained", "labels": rows[0]["explanations"]}

    pair = dict(
        source_question_id="q1",
        image_id="1",
        image_path="images/one.jpg",
        question="Animal?",
        split="train",
        chosen="Đáp án: cat\nGiải thích: Fur.",
        rejected="Đáp án: cat\nGiải thích: Wings.",
    )
    outputs = PairCollator(capture)([pair])
    assert all(row["image_path"] == "images/one.jpg" for row in calls)
    assert calls[0]["question_id"] == calls[1]["question_id"] == "q1"
    assert calls[0]["answer"] == calls[1]["answer"]
    assert all(row["image_grid_thw"] == "retained" for row in outputs.values())
    with pytest.raises(ValueError, match="batch size"):
        PairCollator(capture)([pair, pair])


def test_completion_logp_is_shifted_and_excludes_prompt_and_image():
    torch = pytest.importorskip("torch")
    from vqa_explanation.training.preference import completion_logp

    logits = torch.randn(1, 5, 13, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
        "pixel_values": torch.ones(2, 3),
        "image_grid_thw": torch.ones(1, 3),
    }

    def model(**kwargs):
        assert "pixel_values" in kwargs and "image_grid_thw" in kwargs
        assert "labels" not in kwargs and kwargs["use_cache"] is False
        return SimpleNamespace(logits=logits)

    actual = completion_logp(model, batch)
    expected = torch.log_softmax(logits.float(), -1)[0, 2, 4]
    expected += torch.log_softmax(logits.float(), -1)[0, 3, 5]
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[:, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, 2:4]) > 0
    assert torch.count_nonzero(logits.grad[:, 4:]) == 0


def test_dpo_loss_matches_standard_ratio_and_gradient_direction():
    torch = pytest.importorskip("torch")
    from vqa_explanation.training.preference import dpo_loss

    chosen, rejected = (
        torch.tensor(-3.0, requires_grad=True),
        torch.tensor(-5.0, requires_grad=True),
    )
    ref_chosen, ref_rejected = chosen.detach().clone(), rejected.detach().clone()
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, 0.1)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    improved = dpo_loss(torch.tensor(-2.0), torch.tensor(-6.0), ref_chosen, ref_rejected, 0.1)
    assert improved < loss


def _tiny_policy(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=16,
        n_positions=12,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = get_peft_model(
        GPT2LMHeadModel(config),
        LoraConfig(
            r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM", lora_dropout=0.0
        ),
    )
    parent = tmp_path / "parent"
    model.save_pretrained(parent)
    model.load_adapter(parent, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    return model


def test_reference_restored_after_error(tmp_path):
    from vqa_explanation.training.preference import reference_adapter, trainable_policy_only

    model = _tiny_policy(tmp_path)
    with pytest.raises(RuntimeError, match="forward failed"), reference_adapter(model):
        assert model.active_adapter == "reference"
        raise RuntimeError("forward failed")
    assert model.active_adapter == "default"
    trainable_policy_only(model)


@pytest.mark.parametrize("stage", ["dpo", "sft_chosen"])
def test_real_cpu_trainer_updates_policy_not_reference_and_saves(tmp_path, stage):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainingArguments

    from vqa_explanation.training.preference import (
        make_trainer_class,
        reference_state,
        verify_reference,
    )

    model = _tiny_policy(tmp_path)
    frozen = reference_state(model)
    initial = copy.deepcopy(get_peft_model_state_dict(model, adapter_name="default"))
    chosen = {
        "input_ids": torch.tensor([[1, 2, 9, 10]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 9, 10]]),
    }
    rejected = {
        "input_ids": torch.tensor([[1, 2, 8, 11]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 8, 11]]),
    }
    batch = {"chosen": chosen, "rejected": rejected} if stage == "dpo" else chosen
    args = TrainingArguments(
        output_dir=str(tmp_path / "training"),
        use_cpu=True,
        max_steps=2,
        learning_rate=0.01,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    trainer = make_trainer_class({"stage": stage, "beta": 0.1})(
        model=model,
        args=args,
        train_dataset=[{"row": i} for i in range(4)],
        data_collator=lambda _: batch,
    )
    assert trainer.model_accepts_loss_kwargs is False
    loss = trainer.compute_loss(model, batch)
    if stage == "dpo":
        assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    result = trainer.train()
    assert trainer.state.global_step == 2 and math.isfinite(result.training_loss)
    verify_reference(model, frozen)
    after = get_peft_model_state_dict(model, adapter_name="default")
    assert any(not torch.equal(value, after[name]) for name, value in initial.items())
    target = tmp_path / "saved"
    model.save_pretrained(target, selected_adapters=["default"])
    stored = load_file(target / "adapter_model.safetensors")
    assert set(stored) == set(after)
    assert all(torch.equal(value.cpu(), stored[name]) for name, value in after.items())
    assert not (target / "reference").exists()


def test_new_stage_uses_fp16_on_t4_and_rejects_bf16_emulation():
    from vqa_explanation.training.preference import use_bf16

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _: (7, 5),
        is_bf16_supported=lambda **_: True,
    )
    assert use_bf16({"dtype": "auto"}, cuda) is False
    assert use_bf16({"dtype": "float16"}, cuda) is False
    with pytest.raises(ValueError, match="native BF16"):
        use_bf16({"dtype": "bfloat16"}, cuda)


def test_mutated_reference_is_detected(tmp_path):
    torch = pytest.importorskip("torch")
    from vqa_explanation.training.preference import reference_state, verify_reference

    model = _tiny_policy(tmp_path)
    before = reference_state(model)
    with torch.no_grad():
        model.get_parameter(next(iter(before))).add_(1.0)
    with pytest.raises(ValueError, match="reference weights changed"):
        verify_reference(model, before)


def test_selection_cannot_claim_test_as_validation(tmp_path):
    from vqa_explanation.training.preference_data import validate_selection

    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"status": "selected", "selection_split": "test"}))
    with pytest.raises(ValueError, match="validation-only"):
        validate_selection(path, {}, {"id": "model", "revision": "rev"}, "a" * 64, [])


@pytest.mark.parametrize("stage", ["sft_chosen", "dpo"])
def test_interrupted_training_matches_continuous_weights_and_scheduler(tmp_path, stage):
    torch = pytest.importorskip("torch")
    from peft import get_peft_model_state_dict
    from transformers import TrainingArguments, set_seed

    from vqa_explanation.training.preference import (
        make_trainer_class,
        reference_state,
        verify_reference,
    )
    from vqa_explanation.training.preference_resume import checkpoint_callback, validate_resume

    identity = {"schema": "preference_resume_v1", "plan_sha256": "a" * 64,
                "runtime": {"fixture": "1"}, "dtype": "float32", "gpu": None,
                "budget_updates": 4}
    rows = [{"index": i} for i in range(8)]

    def collate(items):
        token = 7 + items[0]["index"] % 4
        chosen = {"input_ids": torch.tensor([[1, 2, token, 12]]),
                  "attention_mask": torch.ones(1, 4, dtype=torch.long),
                  "labels": torch.tensor([[-100, -100, token, 12]])}
        rejected = {**chosen, "input_ids": torch.tensor([[1, 2, 5, 6]]),
                    "labels": torch.tensor([[-100, -100, 5, 6]])}
        return {"chosen": chosen, "rejected": rejected} if stage == "dpo" else chosen

    def trainer(name, stop=None):
        folder = tmp_path / name
        folder.mkdir()
        model = _tiny_policy(folder)
        set_seed(42)
        args = TrainingArguments(
            output_dir=str(folder / "training"), use_cpu=True, max_steps=4,
            learning_rate=0.01, gradient_accumulation_steps=2,
            per_device_train_batch_size=1, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="steps", save_steps=2, logging_steps=1,
            remove_unused_columns=False, report_to="none", disable_tqdm=True,
            dataloader_pin_memory=False,
        )
        return make_trainer_class({"stage": stage, "beta": 0.1})(
            model=model, args=args, train_dataset=rows, data_collator=collate,
            callbacks=[checkpoint_callback(identity, stop)],
        )

    full = trainer("full")
    full.train()
    partial = trainer("partial", stop=2)
    partial.train()
    checkpoint = tmp_path / "partial/training/checkpoint-2"
    assert validate_resume(checkpoint, identity)["global_step"] == 2
    resumed = trainer("resumed")
    frozen = reference_state(resumed.model)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    verify_reference(resumed.model, frozen)
    expected = get_peft_model_state_dict(full.model, adapter_name="default")
    actual = get_peft_model_state_dict(resumed.model, adapter_name="default")
    for name, value in expected.items():
        torch.testing.assert_close(value, actual[name], rtol=0, atol=0)
    assert resumed.state.global_step == full.state.global_step == 4
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
    with pytest.raises(ValueError, match="budget differs"):
        validate_resume(checkpoint, {**identity, "budget_updates": 5})
    (checkpoint / "optimizer.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="payload changed"):
        validate_resume(checkpoint, identity)
