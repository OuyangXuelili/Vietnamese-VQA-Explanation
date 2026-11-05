"""Offline CPU behavior tests for the production generation guard."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vqa_explanation.inference.generation_safety import NumericalInferenceError
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner


class Inputs(dict):
    def to(self, device):
        assert device == torch.device("cpu")
        return self


class Processor:
    chat_template = "offline template"

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_args = messages, kwargs
        return "prompt"

    def __call__(self, **kwargs):
        assert kwargs["text"] == ["prompt"]
        assert kwargs["images"][0].mode == "RGB"
        assert kwargs["return_tensors"] == "pt"
        return Inputs(input_ids=torch.tensor([[3, 4]]))

    def batch_decode(self, ids, **kwargs):
        self.decoded = (ids.clone(), kwargs)
        return ["  Đáp án: mèo\nGiải thích: Có mèo.  "]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Current Qwen layout nests visual under model; older layouts do not.
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Identity()
        self.lm_head = torch.nn.Identity()
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(to_dict=lambda: {"use_cache": True})
        self.bad_stage = None
        self.bad_value = float("nan")
        self.failure = False
        self.steps = 0

    def forward(self, input_ids):
        vision = torch.tensor([[self.bad_value if self.bad_stage == "vision" else 1.]])
        self.model.visual(vision)
        if self.failure:
            raise RuntimeError("model failed")
        logits = torch.tensor([[1., self.bad_value if self.bad_stage == "head" else 2.]])
        logits = self.lm_head(logits)
        if self.bad_stage == "logits":
            logits = torch.tensor([[self.bad_value, 1.]])
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.called = kwargs
        ids = kwargs["input_ids"]
        for _ in range(2):
            output = self(ids)
            # A legal processed token mask must not be mistaken for raw corruption.
            scores = output.logits.clone()
            scores[:, 0] = -torch.inf
            ids = torch.cat((ids, scores.argmax(-1, keepdim=True)), dim=1)
            self.steps += 1
        return ids


class Adapter(torch.nn.Module):
    """Like PEFT, generation bypasses this wrapper's forward."""

    def __init__(self, base):
        super().__init__()
        self.base_model = base
        self.device = base.device
        self.generation_config = base.generation_config

    def get_base_model(self):
        return self.base_model

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)


@pytest.fixture
def setup_runner(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(path)
    runner = Qwen25VLRunner.__new__(Qwen25VLRunner)
    runner.model, runner.processor, runner.torch = TinyModel(), Processor(), torch
    runner.min_pixels, runner.max_pixels = 10, 20
    runner.load_in_4bit, runner.compute_dtype = False, "torch.float32"
    return runner, path


def assert_no_hooks(model):
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage", ["vision", "head", "logits"])
@pytest.mark.parametrize("adapter", [False, True])
def test_raw_nonfinite_stops_before_decoding_and_next_request_recovers(
    setup_runner, value, stage, adapter,
):
    runner, path = setup_runner
    base = runner.model
    if adapter:
        runner.model = Adapter(base)
    base.bad_stage, base.bad_value = stage, value
    for _ in range(2):
        with pytest.raises(NumericalInferenceError, match="Non-finite"):
            runner.generate(path, "Con gì?")
        assert not hasattr(runner.processor, "decoded")
        assert base.steps == 0
        assert_no_hooks(runner.model)
    base.bad_stage = None
    assert runner.generate(path, "Con gì?").startswith("Đáp án: mèo")
    assert_no_hooks(runner.model)


def test_finite_generation_preserves_args_output_protocol_and_legal_masks(setup_runner):
    runner, path = setup_runner
    before = runner.protocol_metadata(7)
    for _ in range(2):
        assert runner.generate(path, "Con gì?", max_new_tokens=7) == (
            "Đáp án: mèo\nGiải thích: Có mèo.")
        args = runner.model.called
        assert set(args) == {"input_ids", "do_sample", "max_new_tokens"}
        assert args["do_sample"] is False and args["max_new_tokens"] == 7
        assert torch.equal(args["input_ids"], torch.tensor([[3, 4]]))
        ids, decode_args = runner.processor.decoded
        assert torch.equal(ids, torch.tensor([[1, 1]]))
        assert decode_args == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        assert runner.protocol_metadata(7) == before
        assert_no_hooks(runner.model)


def test_model_failure_removes_hooks_and_allows_retry(setup_runner):
    runner, path = setup_runner
    runner.model.failure = True
    with pytest.raises(RuntimeError, match="model failed"):
        runner.generate(path, "Con gì?")
    assert_no_hooks(runner.model)
    runner.model.failure = False
    assert runner.generate(path, "Con gì?").startswith("Đáp án:")
    assert_no_hooks(runner.model)


def test_partial_hook_registration_failure_removes_already_registered_hooks(
    setup_runner, monkeypatch,
):
    runner, path = setup_runner

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(runner.model.lm_head, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        runner.generate(path, "Con gì?")
    assert_no_hooks(runner.model)
