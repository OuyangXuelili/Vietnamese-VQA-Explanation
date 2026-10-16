import asyncio
import base64
import io
import json
import threading

import pytest
from PIL import Image

from scripts.run_demo import launch_settings
from vqa_explanation.demo.service import (
    DemoError,
    LiveService,
    ReplayService,
    decode_image,
    load_live_service,
    validate_question,
)
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("question", [None, "", "  ", "x" * 501, 42])
def test_demo_rejects_bad_questions(question):
    with pytest.raises(DemoError):
        validate_question(question)


def test_demo_image_bounds_and_normalization(image_bytes):
    assert decode_image(image_bytes).mode == "RGB"
    for data in (b"", b"bad", b"x" * (6 * 1024 * 1024 + 1)):
        with pytest.raises(DemoError):
            decode_image(data)
    stream = io.BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="GIF")
    with pytest.raises(DemoError, match="JPEG/PNG"):
        decode_image(stream.getvalue())


class FakeRunner:
    def generate(self, path, question, **kwargs):
        self.path = path
        assert Image.open(path).size == (16, 16)
        assert question == "Ảnh màu gì?"
        return "Đáp án: xanh\nGiải thích: toàn bộ ảnh có màu xanh"


def test_live_parses_real_runner_output_and_cleans_temporary_image(image_bytes):
    runner = FakeRunner()
    service = LiveService(runner, {"experiment_id": "M1_SFT"})
    result = service.predict(image_bytes, " Ảnh màu gì? ")
    assert result.mode == "live" and result.format_valid and result.answer == "xanh"
    assert not runner.path.exists()
    assert not service.lock.locked()


def test_live_failure_releases_lock_and_never_repairs_bad_format(image_bytes):
    class BrokenRunner:
        def generate(self, *args, **kwargs):
            raise RuntimeError("failure")
    service = LiveService(BrokenRunner(), {"experiment_id": "M1_SFT"})
    with pytest.raises(RuntimeError):
        service.predict(image_bytes, "Ảnh màu gì?")
    assert not service.lock.locked()
    service.runner.generate = lambda *args, **kwargs: "just a sentence"
    result = service.predict(image_bytes, "Ảnh màu gì?")
    assert not result.format_valid and result.raw_output == "just a sentence"


def test_live_rejects_concurrent_request(image_bytes):
    service = LiveService(FakeRunner(), {"experiment_id": "M1_SFT"})
    with service.lock, pytest.raises(DemoError, match="GPU đang"):
        service.predict(image_bytes, "Ảnh màu gì?")


@pytest.fixture
def replay_files(tmp_path, image_bytes):
    image = tmp_path / "image.png"
    image.write_bytes(image_bytes)
    annotations = tmp_path / "validation.jsonl"
    record = {"question_id": "q1", "image_id": "i1", "image_path": "image.png",
              "split": "validation", "question": "Ảnh màu gì?", "answer": "xanh",
              "explanations": ["ảnh xanh"]}
    annotations.write_text(json.dumps(record) + "\n", encoding="utf-8")
    inputs = {"experiment_id": "M1_SFT", "split": "validation", "adapter": "M1",
              "adapter_files": {"adapter": "hash"}, "model_id": "pinned-model",
              "model_revision": "revision", "seed": 42,
              "annotations_sha256": sha256_file(annotations),
              "image_hashes": {"q1": sha256_file(image)}}
    protocol = {"min_pixels": 200704, "max_pixels": 200704, "max_new_tokens": 128}
    predictions = tmp_path / "predictions.jsonl"
    row = {**record, **protocol, **{key: inputs[key] for key in (
        "experiment_id", "adapter", "model_id", "model_revision", "seed")},
        "image_sha256": sha256_file(image), "reference_answer": "xanh",
        "reference_explanations": ["ảnh xanh"], "raw_output": "Đáp án: xanh\nGiải thích: ảnh xanh",
        "prediction_answer": "xanh", "prediction_explanation": "ảnh xanh", "format_valid": True,
        "latency_seconds": 1.2}
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": inputs, "protocol": protocol}), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "selected", "selection_split": "validation",
        "model_id": "pinned-model", "model_revision": "revision",
        "adapter": {"files": inputs["adapter_files"]}, "evidence": {
            key: {"sha256": sha256_file(path)} for key, path in (
                ("validation_predictions", predictions), ("validation_manifest", manifest),
                ("validation_annotations", annotations))}}), encoding="utf-8")
    return predictions, manifest, annotations, tmp_path, selection


def test_replay_exact_saved_outputs_and_only_fixed_examples(replay_files):
    service = ReplayService(*replay_files, count=1)
    assert service.info()["mode"] == "replay"
    assert service.example("q1")["raw_output"] == "Đáp án: xanh\nGiải thích: ảnh xanh"
    assert service.example("q1")["latency_seconds"] == 1.2
    assert service.image("q1")
    with pytest.raises(DemoError):
        service.example("../selection.json")
    (replay_files[3] / "image.png").write_bytes(b"changed")
    with pytest.raises(DemoError, match="đã đổi"):
        service.image("q1")


def test_replay_rejects_changed_export_and_image(replay_files):
    predictions, _, _, root, _ = replay_files
    with pytest.raises(ValueError, match="count"):
        ReplayService(*replay_files, count=0)
    (root / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ReplayService(*replay_files, count=1)
    predictions.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen M1 selection"):
        ReplayService(*replay_files, count=1)


def client(service, **kwargs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vqa_explanation.demo.web import create_app

    return TestClient(create_app(service, **kwargs), base_url="http://127.0.0.1")


def test_demo_http_replay_cannot_claim_live_inference(replay_files):
    api = client(ReplayService(*replay_files, count=1))
    assert api.get("/").status_code == 200
    assert api.get("/api/info").json()["mode"] == "replay"
    assert api.get("/assets/demo.js").status_code == 200
    assert api.get("/assets/credentials.json").status_code == 404
    assert api.get("/api/examples/q1/image").headers["content-type"] == "image/png"
    assert api.post("/api/predict", json={}).status_code == 409
    assert api.get("/api/info", headers={"Origin": "https://foreign.test"}).status_code == 403
    assert api.get("/api/info", headers={"Host": "foreign.test"}).status_code == 403
    assert "default-src 'self'" in api.get("/").headers["content-security-policy"]


def test_demo_http_live_input_errors_and_private_auth(image_bytes):
    service = LiveService(FakeRunner(), {"experiment_id": "M1_SFT"})
    api = client(service, auth=("tester", "local-test-password"))
    assert api.get("/api/info").status_code == 401
    assert api.get("/api/info", auth=("tester", "wrong")).status_code == 401
    api.auth = ("tester", "local-test-password")
    assert api.get("/api/info").status_code == 200
    payload = {"image_base64": base64.b64encode(image_bytes).decode(), "question": "Ảnh màu gì?"}
    assert api.post("/api/predict", json=payload).json()["answer"] == "xanh"
    assert api.post("/api/predict", json={**payload, "question": " "}).status_code == 400
    assert api.post("/api/predict", json={**payload, "image_base64": "bad!"}).status_code == 400
    assert api.post("/api/predict", content=b"bad").status_code == 415
    assert api.get("/api/examples/q1").status_code == 404


def test_demo_timeout_keeps_worker_lock_until_finish(image_bytes):
    started, release = threading.Event(), threading.Event()
    class SlowRunner:
        def generate(self, *args, **kwargs):
            started.set()
            release.wait(2)
            return "Đáp án: xanh\nGiải thích: ảnh xanh"
    service = LiveService(SlowRunner(), {"experiment_id": "M1_SFT"})
    api = client(service, timeout_seconds=.01)
    payload = {"question": "Ảnh màu gì?", "image_base64": base64.b64encode(image_bytes).decode()}
    try:
        with api:
            assert api.post("/api/predict", json=payload).status_code == 504
            assert started.is_set() and service.lock.locked()
            assert api.post("/api/predict", json=payload).status_code == 409
            release.set()
    finally:
        release.set()


def test_demo_handles_the_distinct_python310_asyncio_timeout(monkeypatch, image_bytes):
    class LegacyAsyncioTimeoutError(Exception):
        pass

    async def legacy_wait_for(awaitable, _timeout):
        awaitable.close()
        raise LegacyAsyncioTimeoutError

    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncioTimeoutError)
    monkeypatch.setattr(asyncio, "wait_for", legacy_wait_for)
    api = client(LiveService(FakeRunner(), {"experiment_id": "M1_SFT"}))
    response = api.post("/api/predict", json={
        "question": "Ảnh màu gì?", "image_base64": base64.b64encode(image_bytes).decode(),
    })
    assert response.status_code == 504
    assert "Quá thời gian chờ" in response.json()["detail"]


def test_demo_auth_is_required_before_external_listen(monkeypatch):
    monkeypatch.delenv("VQA_DEMO_USER", raising=False)
    monkeypatch.delenv("VQA_DEMO_PASSWORD", raising=False)
    assert launch_settings("127.0.0.1", 7860) == {"auth": None}
    with pytest.raises(ValueError, match="authentication"):
        launch_settings("0.0.0.0", 7860)
    monkeypatch.setenv("VQA_DEMO_USER", "tester")
    with pytest.raises(ValueError, match="both"):
        launch_settings("0.0.0.0", 7860)
    monkeypatch.setenv("VQA_DEMO_PASSWORD", "local-test-password")
    assert launch_settings("0.0.0.0", 7860)["auth"][0] == "tester"


def test_live_factory_rejects_wrong_adapter_and_cpu_before_model_load(
    tmp_path, monkeypatch,
):
    from vqa_explanation.training.preference_data import adapter_fingerprint

    adapter = tmp_path / "checkpoint-3683"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": {"experiment_id": "M1_SFT", "split": "validation",
        "adapter_files": adapter_fingerprint(adapter)["files"], "model_id": "model",
        "model_revision": "rev"}, "protocol": {}}), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "selected", "selection_split": "validation",
        "adapter": adapter_fingerprint(adapter), "model_id": "model", "model_revision": "rev",
        "evidence": {"validation_manifest": {"sha256": sha256_file(manifest)}}}),
        encoding="utf-8")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        load_live_service(adapter, selection, manifest)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="frozen M1 selection"):
        load_live_service(adapter, selection, manifest)


def test_release_includes_demo_frontend(tmp_path):
    from scripts.package_source import REQUIRED_FILES, source_files

    for name in (*REQUIRED_FILES, "app/static/index.html", "app/static/demo.css",
                 "app/static/demo.js"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    selected = {path.relative_to(tmp_path).as_posix() for path in source_files(tmp_path)}
    assert {"app/static/index.html", "app/static/demo.css", "app/static/demo.js"} <= selected
