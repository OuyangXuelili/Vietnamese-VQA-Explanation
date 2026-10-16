import pytest

from vqa_explanation.training.sft import validate_smoke_completion
from vqa_explanation.training.sft_data import assistant_labels, select_training_records, target_for


def row(qid, image, split="train"):
    return dict(
        question_id=qid,
        image_id=image,
        split=split,
        image_path=f"images/{image}.jpg",
        question="Có gì?",
        answer="mèo",
        explanations=["Có một con mèo."],
    )


def test_masks_prompt_and_vision_but_keeps_assistant_eos():
    assert assistant_labels([1, 99, 2, 3, 4], [1, 99, 2], {99}) == [-100] * 3 + [3, 4]


def test_refuses_template_mismatch():
    with pytest.raises(ValueError, match="prefix"):
        assistant_labels([1, 2, 3], [1, 4], set())


def test_refuses_empty_target():
    with pytest.raises(ValueError, match="No assistant"):
        assistant_labels([1, 2], [1, 2], set())


def test_string_explanation_keeps_whole_reference():
    record = row("a", "1")
    record["explanations"] = "Có một con mèo."
    assert target_for(record) == "Đáp án: mèo\nGiải thích: Có một con mèo."


def test_never_trains_heldout_and_selection_is_reproducible():
    records = [row(str(i), str(i)) for i in range(20)] + [row("v", "v", "validation")]
    chosen = select_training_records(records, 5, 42)
    assert len(chosen) == 5
    assert all(r["split"] == "train" for r in chosen)
    assert chosen == select_training_records(records, 5, 42)


def test_rejects_image_leakage():
    with pytest.raises(ValueError, match="overlap"):
        select_training_records([row("a", "1"), row("b", "1", "test")], 1, 42)


def test_rejects_shared_path_even_when_image_ids_differ():
    heldout = row("b", "2", "validation")
    heldout["image_path"] = "images/1.jpg"
    with pytest.raises(ValueError, match="overlap"):
        select_training_records([row("a", "1"), heldout], 1, 42)


def test_rejects_noncanonical_split_label_in_selection():
    with pytest.raises(ValueError, match="Noncanonical split label"):
        select_training_records([row("a", "1"), row("b", "2", "val")], 1, 42)


def test_rejects_duplicate_question():
    with pytest.raises(ValueError, match="Duplicate"):
        select_training_records([row("a", "1"), row("a", "2")], 1, 42)


def test_smoke_requires_target_step_finite_loss_and_updated_weights():
    validate_smoke_completion(2, 2, 1.0, 0.001)
    with pytest.raises(RuntimeError, match="not target"):
        validate_smoke_completion(1, 2, 1.0, 0.001)
    with pytest.raises(RuntimeError, match="Non-finite"):
        validate_smoke_completion(2, 2, float("nan"), 0.001)
    with pytest.raises(RuntimeError, match="did not update"):
        validate_smoke_completion(2, 2, 1.0, 0.0)


def test_native_bf16_excludes_emulation_and_handles_older_torch():
    from types import SimpleNamespace

    from vqa_explanation.training.sft import supports_native_bf16

    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda device: (7, 5),
        is_bf16_supported=lambda **kw: calls.append(kw) or True,
    )
    assert not supports_native_bf16(cuda)
    assert not calls
    cuda.get_device_capability = lambda device: (8, 0)
    assert supports_native_bf16(cuda)
    assert calls == [{"including_emulation": False}]
    cuda.is_bf16_supported = lambda: True
    assert supports_native_bf16(cuda)
    cuda.is_available = lambda: False
    assert not supports_native_bf16(cuda)


def test_grid_reuse_is_equal_to_double_processing_and_caches_vocab(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from vqa_explanation.training.sft_data import VQACollator

    class Processor:
        image_token = "<|image_pad|>"
        image_processor = SimpleNamespace(merge_size=2)

        def __init__(self):
            self.image_calls = 0
            self.vocab_calls = 0
            self.tokenizer = SimpleNamespace(get_vocab=self.get_vocab)

        def get_vocab(self):
            self.vocab_calls += 1
            return {self.image_token: 999}

        def apply_chat_template(self, messages, **kwargs):
            prompt = "Q" + self.image_token + "A"
            return prompt if kwargs["add_generation_prompt"] else prompt + "cat!"

        def __call__(self, text, images=None, return_tensors=None):
            text = text[0]
            if images is not None:
                self.image_calls += 1
                text = text.replace(self.image_token, self.image_token * 4)
            # Treat the image placeholder as one special token, as Qwen does.
            ids = []
            while text:
                if text.startswith(self.image_token):
                    ids.append(999)
                    text = text[len(self.image_token) :]
                else:
                    ids.append(ord(text[0]))
                    text = text[1:]
            result = {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)))}
            if images is not None:
                result.update(image_grid_thw=np.array([[1, 4, 4]]), pixel_values=np.ones((16, 3)))
            return result

    (tmp_path / "images").mkdir()
    Image.new("RGB", (24, 32)).save(tmp_path / "images/1.jpg")
    old_processor, new_processor = Processor(), Processor()
    old = VQACollator(old_processor, tmp_path, reuse_image_grid=False, return_tensors="np")
    new = VQACollator(new_processor, tmp_path, return_tensors="np")
    for _ in range(2):
        expected, actual = old([row("a", "1")]), new([row("a", "1")])
        assert expected.keys() == actual.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual["labels"].tolist() == [[-100] * 6 + list(map(ord, "cat!"))]
    assert old_processor.image_calls == 4
    assert new_processor.image_calls == 2
    assert new_processor.vocab_calls == 1
    with pytest.raises(ValueError, match="exactly one"):
        new._prompt_ids("no image", None, {"image_grid_thw": np.array([[1, 4, 4]])})
    with pytest.raises(ValueError, match="divisible"):
        new._prompt_ids(new_processor.image_token, None, {"image_grid_thw": np.array([[1, 3, 3]])})
    with pytest.raises(ValueError, match="Invalid image grid"):
        new._prompt_ids(new_processor.image_token, None, {"image_grid_thw": np.array([[1, 0, 4]])})
