from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.unit.test_comparison import adapter_sample, annotation, sample
from tests.unit.test_generation_provenance import export_case as export_case  # noqa: F401
from tests.unit.test_prediction_export import metadata_runner
from vqa_explanation.evaluation.comparison import compare_predictions
from vqa_explanation.inference.export import export_predictions
from vqa_explanation.utils.model_loading import (
    quantization_kwargs,
    resolve_dtype,
    validate_loading_profile,
)


def test_explicit_precision_and_quantization_do_not_depend_on_torch_defaults():
    torch = SimpleNamespace(float16="half", bfloat16="brain", float32="single")
    assert resolve_dtype(torch, "float16", native_bf16=True) == "half"
    assert resolve_dtype(torch, "auto", native_bf16=True) == "brain"
    with pytest.raises(ValueError, match="native BF16"):
        resolve_dtype(torch, "bfloat16", native_bf16=False)
    assert quantization_kwargs("half", double_quant=True)["bnb_4bit_use_double_quant"] is True
    assert quantization_kwargs("half", double_quant=False)["bnb_4bit_use_double_quant"] is False


@pytest.mark.parametrize("profile", [
    {}, {"dtype": "float16", "double_quant": "false", "attention": "sdpa"},
    {"dtype": "float64", "double_quant": False, "attention": "sdpa"},
    {"dtype": "float16", "double_quant": False, "attention": "flash_unknown"},
])
def test_loading_profile_rejects_ambiguous_values(profile):
    with pytest.raises(ValueError):
        validate_loading_profile(profile)


def test_protocol_records_effective_loading_profile():
    runner = metadata_runner()
    profile = {"dtype": "float16", "double_quant": True, "attention": "sdpa"}
    runner.loading_profile = profile
    metadata = runner.protocol_metadata(8)
    assert metadata["model_loading"] == profile
    profile["double_quant"] = False
    assert metadata["model_loading"]["double_quant"] is True


def test_pair_comparison_rejects_different_loading_profiles():
    left, right = sample(), adapter_sample()
    profile = {"dtype": "float16", "double_quant": False, "attention": "sdpa"}
    left["model_loading"] = dict(profile)
    right["model_loading"] = dict(profile)
    assert compare_predictions([left], [right])["comparable"]
    right["model_loading"]["double_quant"] = True
    result = compare_predictions([left], [right])
    assert not result["comparable"]
    assert any("model_loading" in issue for issue in result["issues"])
    del right["model_loading"]
    assert not compare_predictions([left], [right])["comparable"]


def test_new_arm_compares_on_validation_without_changing_four_arm_benchmark():
    left, right = adapter_sample(), adapter_sample()
    left["experiment_id"] = "M3_DPO"
    right["experiment_id"] = "M4_DPO_ANSWER"
    expected = {"q1": annotation()}
    assert compare_predictions([left], [right], expected_annotations=expected,
                               expected_experiments=("M3_DPO", "M4_DPO_ANSWER"))["comparable"]
    with pytest.raises(ValueError, match="validation-only"):
        compare_predictions([left], [right], expected_split="test", expected_annotations=expected,
                            expected_experiments=("M3_DPO", "M4_DPO_ANSWER"))


def test_resume_rejects_profile_change_before_loading_model(export_case, tmp_path):
    args, _, _ = export_case
    profile = tmp_path / "loading.yaml"
    profile.write_text("dtype: float16\ndouble_quant: false\nattention: sdpa\n", "utf-8")
    args.loading_profile = profile
    runner = metadata_runner()
    runner.loading_profile = {"dtype": "float16", "double_quant": False, "attention": "sdpa"}
    runner.generate = lambda *a, **kw: "Đáp án: cat\nGiải thích: A cat is visible."
    export_predictions(args, runner_factory=lambda *a, **kw: runner)
    original = args.output.read_bytes()
    profile.write_text("dtype: float16\ndouble_quant: true\nattention: sdpa\n", "utf-8")
    args.resume = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed profile must fail before loading model")

    with pytest.raises(SystemExit, match="changed"):
        export_predictions(args, runner_factory=forbidden)
    assert args.output.read_bytes() == original


def test_profile_dtype_must_match_actual_precision():
    left, right = sample(), deepcopy(adapter_sample())
    profile = {"dtype": "float32", "double_quant": False, "attention": "sdpa"}
    left["model_loading"] = dict(profile)
    right["model_loading"] = dict(profile)
    assert not compare_predictions([left], [right])["comparable"]
