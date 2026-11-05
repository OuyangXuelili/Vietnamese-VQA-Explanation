from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from vqa_explanation.training.preference_data import load_config, validate_study_arm
from vqa_explanation.training.sft_data import VQACollator, answer_token_mask


def test_answer_mask_uses_offsets_and_excludes_prompt_labels_and_explanation():
    offsets = [(0, 0), (0, 6), (6, 9), (9, 12), (12, 20), (0, 0)]
    assert answer_token_mask(offsets, [-100, 1, 2, 3, 4, 5], 8, 12) == [
        False, False, True, True, False, False]
    with pytest.raises(ValueError, match="no supervised"):
        answer_token_mask(offsets, [-100] * 6, 8, 12)


@pytest.mark.parametrize("offsets,start,end", [
    ([(0, 1)], 2, 2), ([(0, 1)], True, 2), ([(2, 1)], 0, 2),
    ([(0, "1")], 0, 2), ([], 0, 2),
])
def test_answer_mask_rejects_bad_spans(offsets, start, end):
    with pytest.raises(ValueError):
        answer_token_mask(offsets, [1], start, end)


class CharacterProcessor:
    image_token = "<image>"
    image_processor = SimpleNamespace(merge_size=1)
    tokenizer = SimpleNamespace(get_vocab=lambda: {})

    def __init__(self, *, changed_offsets=False):
        self.image_calls = 0
        self.changed_offsets = changed_offsets

    def apply_chat_template(self, messages, **kwargs):
        prompt = "USER<image>QUESTION\nASSISTANT\n"
        return prompt if kwargs["add_generation_prompt"] else (
            prompt + messages[-1]["content"][0]["text"] + "<eos>")

    def __call__(self, text, images=None, return_tensors=None, return_offsets_mapping=False):
        text = text[0]
        if images is not None:
            self.image_calls += 1
            text = text.replace(self.image_token, self.image_token * 4)
        ids = list(map(ord, text))
        result = {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)))}
        if images is not None:
            result.update(image_grid_thw=np.array([[1, 2, 2]]), pixel_values=np.ones((4, 3)))
        if return_offsets_mapping:
            result["offset_mapping"] = np.array([[(i, i + 1) for i in range(len(text))]])
            if self.changed_offsets:
                result["input_ids"][0, -1] += 1
        return result


def test_collator_masks_only_answer_without_reprocessing_image(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    row = {"question_id": "q", "image_id": "i", "image_path": "image.jpg", "split": "train",
           "question": "Con gì?", "answer": "mèo", "explanations": ["Có một con mèo."]}
    processor = CharacterProcessor()
    collator = VQACollator(processor, tmp_path, include_answer_mask=True, return_tensors="np")
    batch = collator([row])
    assert ''.join(chr(i) for i in batch["input_ids"][batch["answer_mask"]]) == "mèo"
    assert not np.any(batch["answer_mask"] & (batch["labels"] == -100))
    assert processor.image_calls == 1
    changed = VQACollator(CharacterProcessor(changed_offsets=True), tmp_path,
                          include_answer_mask=True, return_tensors="np")
    with pytest.raises(ValueError, match="offsets differ"):
        changed([row])


def test_answer_nll_gradient_has_no_explanation_or_prompt_targets():
    import torch

    from vqa_explanation.training.preference import completion_logp

    logits = torch.nn.Parameter(torch.zeros(1, 5, 16))

    def model(**kwargs):
        assert "answer_mask" not in kwargs and "labels" not in kwargs
        return SimpleNamespace(logits=logits)

    batch = {"input_ids": torch.tensor([[1, 2, 9, 10, 3]]),
             "labels": torch.tensor([[-100, -100, 9, 10, 3]]),
             "answer_mask": torch.tensor([[False, False, True, False, False]])}
    full, nll = completion_logp(model, batch, with_answer_nll=True)
    assert nll.item() == pytest.approx(np.log(16))
    torch.testing.assert_close(full, completion_logp(model, batch))
    nll.backward()
    assert logits.grad[0, 1].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[0, [0, 2, 3, 4]]) == 0
    batch["answer_mask"][0, 0] = True
    with pytest.raises(ValueError, match="assistant supervision"):
        completion_logp(model, batch, with_answer_nll=True)


def test_new_arm_is_bound_to_matching_control_configs():
    study = Path("configs/experiments/answer_preservation.yaml")
    report = validate_study_arm("configs/training/dpo_answer.yaml", study)
    assert report["arm"] == "m4"
    assert set(report["config_sha256"]) == {"m2", "m3", "m4"}
    assert report["matched_fields"]["expected_pairs"] == 1000
    with pytest.raises(ValueError, match="not the declared m4"):
        validate_study_arm("configs/training/dpo_answer.yaml",
                           "configs/experiments/preference_study_v2.yaml")


@pytest.mark.parametrize("weight", [-0.1, float("nan"), True, "0.1", 0])
def test_new_arm_rejects_invalid_or_missing_answer_weight(tmp_path, weight):
    cfg = yaml.safe_load(Path("configs/training/dpo_answer.yaml").read_text("utf-8"))
    cfg["answer_nll_weight"] = weight
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_regularization_weight_is_part_of_declared_comparison(tmp_path):
    cfg = yaml.safe_load(Path("configs/experiments/answer_preservation.yaml").read_text("utf-8"))
    cfg["training"]["answer_nll_weight"] = 0.5
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")
    with pytest.raises(ValueError, match="answer_nll_weight"):
        validate_study_arm("configs/training/dpo_answer.yaml", path)


def test_zero_weight_is_exactly_standard_dpo(tmp_path):
    import torch

    from tests.unit.test_preference_training import _tiny_policy
    from vqa_explanation.training.preference import make_trainer_class

    model = _tiny_policy(tmp_path)
    chosen = {"input_ids": torch.tensor([[1, 2, 9, 10]]),
              "labels": torch.tensor([[-100, -100, 9, 10]])}
    rejected = {"input_ids": torch.tensor([[1, 2, 8, 11]]),
                "labels": torch.tensor([[-100, -100, 8, 11]])}
    batch = {"chosen": chosen, "rejected": rejected}
    standard = make_trainer_class({"stage": "dpo", "beta": 0.1})
    zero = make_trainer_class({"stage": "dpo", "beta": 0.1, "answer_nll_weight": 0.0})
    torch.testing.assert_close(standard.compute_loss(SimpleNamespace(), model, batch),
                               zero.compute_loss(SimpleNamespace(), model, batch), rtol=0, atol=0)
