"""Numerical inference failures must produce errors, never success payloads."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vqa_explanation.demo.service import DemoError, LiveService
from vqa_explanation.demo.web import create_app
from vqa_explanation.inference.generation_safety import NumericalInferenceError


class Runner:
    def __init__(self):
        self.error = NumericalInferenceError("Non-finite model output at logits")
        self.paths = []
        self.text = "Đáp án: xanh\nGiải thích: ảnh xanh"

    def generate(self, path, question, **kwargs):
        assert path.is_file()
        self.paths.append(path)
        if self.error:
            raise self.error
        return self.text


@pytest.fixture
def live():
    stream = io.BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="PNG")
    runner = Runner()
    return LiveService(runner, {"experiment_id": "M1_SFT"}), runner, stream.getvalue()


def assert_clean(service, runner):
    assert not service.lock.locked()
    assert all(not path.exists() and not path.parent.exists() for path in runner.paths)


def test_service_maps_only_numerical_error_and_recovers(live):
    service, runner, data = live
    for _ in range(2):
        with pytest.raises(DemoError, match="NaN/Inf") as caught:
            service.predict(data, "Màu gì?")
        assert caught.value.status_code == 503
        assert caught.value.__cause__ is runner.error
        assert "thử ảnh khác" in str(caught.value)
        assert_clean(service, runner)
    runner.error = RuntimeError("ordinary failure")
    with pytest.raises(RuntimeError, match="ordinary failure"):
        service.predict(data, "Màu gì?")
    assert_clean(service, runner)
    runner.error = None
    runner.text = "!" * 128
    malformed = service.predict(data, "Màu gì?")
    assert malformed.raw_output == runner.text and not malformed.format_valid
    assert_clean(service, runner)


def test_api_numerical_error_has_no_success_payload_then_succeeds(live):
    service, runner, data = live
    payload = {"question": "Màu gì?", "image_base64": base64.b64encode(data).decode()}
    with TestClient(create_app(service), base_url="http://127.0.0.1") as api:
        response = api.post("/api/predict", json=payload)
        assert response.status_code == 503
        assert set(response.json()) == {"detail"}
        assert "lỗi số học" in response.json()["detail"]
        assert "thử ảnh khác" in response.json()["detail"]
        assert_clean(service, runner)
        runner.error = None
        response = api.post("/api/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["answer"] == "xanh"
        assert response.json()["format_valid"] is True
        assert_clean(service, runner)
