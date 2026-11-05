from pathlib import Path

import pytest

from vqa_explanation.inference.generation_safety import NumericalInferenceError
from vqa_explanation.inference.stability import probe_cases
from vqa_explanation.utils.run_provenance import sha256_file


def test_numerical_probe_continues_with_control_after_known_failure(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"offline fixture; runner does not decode images")
    cases = [{"name": name, "role": role, "question": name, "image_path": image.name,
              "image_sha256": sha256_file(image)}
             for name, role in (("before", "control"), ("nan", "regression"),
                                ("after", "control"))]
    called = []

    class Runner:
        def generate(self, path, question, **kwargs):
            called.append(question)
            assert path == image and kwargs["max_new_tokens"] == 8
            if question == "nan":
                raise NumericalInferenceError("Non-finite model output at vision")
            return "finite output"

    results = probe_cases(Runner(), cases, tmp_path, 8)
    assert called == ["before", "nan", "after"]
    assert [row["status"] for row in results] == ["finite", "nonfinite", "finite"]
    assert results[-1]["raw_output"] == "finite output"


def test_numerical_probe_rejects_changed_input_before_runner(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"original")
    case = {"name": "case", "role": "control", "question": "Q", "image_path": image.name,
            "image_sha256": sha256_file(image)}
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        probe_cases(object(), [case], Path(tmp_path), 8)
