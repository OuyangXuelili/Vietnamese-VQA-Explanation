import ast
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scripts import run_demo
from scripts.run_demo import launch_settings
from vqa_explanation.demo import colab
from vqa_explanation.demo.service import LiveService
from vqa_explanation.demo.web import create_app

ORIGIN = "https://session-1234-7860-colab.googleusercontent.com"
CURRENT_ORIGIN = "https://7860-gpu-t4-s-session-token-b.asia-southeast1.prod.r.colab.dev"
INTERNAL_AUTHORITY = (
    "gpu-t4-s-session-token.asia-southeast1-b.c.codatalab-user-runtimes.internal:8007"
)
FRAME = "https://different-cell-0-colab.googleusercontent.com"
IDENTITY = {"experiment_id": "M1_SFT", "model_id": "fixture", "model_revision": "revision",
            "adapter_sha256": "a" * 64}


def test_proxy_uses_exact_private_origin_and_frame_parents():
    policy = colab.proxy_policy(ORIGIN + "/", 7860)
    assert policy["origin"] == ORIGIN
    assert policy["host"] == ORIGIN.removeprefix("https://")
    assert "https://session-1234-0-colab.googleusercontent.com" in policy["frame_sources"]
    assert "*" not in policy["frame_sources"]
    frame = "https://different-cell-0-colab.googleusercontent.com"
    assert frame in colab.proxy_policy(ORIGIN, frame_origin=frame)["frame_sources"]
    for invalid in ("https://evil.test", ORIGIN, frame + "?token=test"):
        with pytest.raises(ValueError, match="frame"):
            colab.proxy_policy(ORIGIN, frame_origin=invalid)


def test_current_colab_proxy_requires_its_exact_frame_and_matching_port():
    policy = colab.proxy_policy(CURRENT_ORIGIN + "/", 7860, FRAME)
    assert policy["origin"] == CURRENT_ORIGIN
    assert policy["host"] == CURRENT_ORIGIN.removeprefix("https://")
    assert policy["internal_authority"] == INTERNAL_AUTHORITY
    assert FRAME in policy["frame_sources"] and "*" not in policy["frame_sources"]
    assert launch_settings("127.0.0.1", 7860, CURRENT_ORIGIN, FRAME)["colab_origin"] == (
        CURRENT_ORIGIN
    )
    with pytest.raises(ValueError, match="frame"):
        colab.proxy_policy(CURRENT_ORIGIN)
    with pytest.raises(ValueError, match="port"):
        colab.proxy_policy(CURRENT_ORIGIN, 8000, FRAME)


@pytest.mark.parametrize("host", [
    CURRENT_ORIGIN.removeprefix("https://") + ":443",
    CURRENT_ORIGIN.removeprefix("https://") + ":7860",
    INTERNAL_AUTHORITY,
])
def test_private_proxy_forwarding_retains_origin_checks(host):
    service = LiveService(None, IDENTITY)
    app = create_app(service, colab_origin=CURRENT_ORIGIN, colab_frame_origin=FRAME)
    api = TestClient(app, client=("127.0.0.1", 12345))
    assert api.get("/api/info", headers={"Host": host, "Origin": CURRENT_ORIGIN}).status_code == 200
    denied = api.get("/api/info", headers={"Host": host, "Origin": "https://evil.test"})
    assert denied.status_code == 403


@pytest.mark.parametrize("host", [
    "foreign.test:7860", "169.254.169.254:7860", "8.8.8.8:7860", "172.28.0.12:8000",
    "172.28.0.12:7860", "10.0.0.2:7860", "192.168.1.2:7860", "0.0.0.0:7860",
    "user:password@172.28.0.12:7860", "172.28.0.12:7860/path",
    CURRENT_ORIGIN.removeprefix("https://") + ":8000",
    INTERNAL_AUTHORITY.replace("session-token", "another-session"),
    INTERNAL_AUTHORITY.replace("southeast1-b", "southeast1-a"),
    INTERNAL_AUTHORITY.replace(":8007", ":8008"),
])
def test_proxy_does_not_trust_arbitrary_authorities(host):
    policy = colab.proxy_policy(CURRENT_ORIGIN, 7860, FRAME)
    assert not colab.proxy_host_allowed(host, policy, "127.0.0.1")


def test_internal_proxy_target_requires_a_real_loopback_peer():
    service = LiveService(None, IDENTITY)
    app = create_app(service, colab_origin=CURRENT_ORIGIN, colab_frame_origin=FRAME)
    api = TestClient(app, client=("203.0.113.10", 12345))
    assert api.get("/api/info", headers={
        "Host": INTERNAL_AUTHORITY, "X-Forwarded-For": "127.0.0.1",
        "X-Forwarded-Host": colab.proxy_policy(CURRENT_ORIGIN, 7860, FRAME)["host"],
    }).status_code == 403


def test_internal_authority_is_not_shared_with_a_different_proxy_session():
    policy = colab.proxy_policy(ORIGIN, 7860)
    assert not colab.proxy_host_allowed(INTERNAL_AUTHORITY, policy, "127.0.0.1")


def test_demo_server_does_not_trust_forwarded_peer_headers(monkeypatch):
    import uvicorn

    class Service:
        mode = "replay"
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--mode", "replay"])
    monkeypatch.setattr(run_demo, "ReplayService", lambda *args, **kwargs: Service())
    monkeypatch.setattr(run_demo, "load_environment", lambda *_: {"image_root": "."})
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    run_demo.main()
    assert calls[0]["proxy_headers"] is False


@pytest.mark.parametrize("origin", [
    "http://session-7860-colab.googleusercontent.com", "https://evil.test",
    "https://session-7860-colab.googleusercontent.com.evil.test",
    "https://user:password@session-7860-colab.googleusercontent.com",
    ORIGIN + ":443", ORIGIN + "/path", ORIGIN + "?token=test", ORIGIN + "#fragment",
    "https://session-0-colab.googleusercontent.com",
    "https://session-65536-colab.googleusercontent.com",
    CURRENT_ORIGIN + ".evil.test", CURRENT_ORIGIN + ":443", CURRENT_ORIGIN + "/path",
    CURRENT_ORIGIN + "?token=test", CURRENT_ORIGIN + "#fragment",
    CURRENT_ORIGIN.replace("https://", "https://user:password@"),
    CURRENT_ORIGIN.replace("https://", "http://"),
    CURRENT_ORIGIN.replace("7860-", "0-"), CURRENT_ORIGIN.replace("7860-", "65536-"),
    CURRENT_ORIGIN.replace(".prod.r.", ".other.r."),
])
def test_proxy_rejects_other_origins(origin):
    with pytest.raises(ValueError, match="Colab"):
        colab.proxy_policy(origin)


def test_colab_launch_cannot_expose_a_network_listener(monkeypatch):
    monkeypatch.delenv("VQA_DEMO_USER", raising=False)
    monkeypatch.delenv("VQA_DEMO_PASSWORD", raising=False)
    assert launch_settings("127.0.0.1", 7860, ORIGIN)["colab_origin"] == ORIGIN
    with pytest.raises(ValueError, match="port"):
        launch_settings("127.0.0.1", 8000, ORIGIN)
    monkeypatch.setenv("VQA_DEMO_USER", "test")
    monkeypatch.setenv("VQA_DEMO_PASSWORD", "fixture-password")
    with pytest.raises(ValueError, match="loopback"):
        launch_settings("0.0.0.0", 7860, ORIGIN)


@pytest.mark.parametrize("origin,frame", [(ORIGIN, None), (CURRENT_ORIGIN, FRAME)])
def test_colab_http_allows_private_proxy_but_not_foreign_hosts_or_origins(origin, frame):
    class Runner:
        def generate(self, *args, **kwargs):
            return "Đáp án: xanh\nGiải thích: ảnh có màu xanh"
    service = LiveService(Runner(), IDENTITY)
    policy = colab.proxy_policy(origin, frame_origin=frame)
    api = TestClient(create_app(service, colab_origin=origin, colab_frame_origin=frame),
                     base_url="http://127.0.0.1")
    response = api.get("/api/info", headers={"Origin": origin})
    assert response.status_code == 200 and response.json()["mode"] == "live"
    assert "x-frame-options" not in response.headers
    assert "https://colab.research.google.com" in response.headers["content-security-policy"]
    assert api.get("/", headers={"Host": policy["host"]}).status_code == 200
    denied_host = api.get("/", headers={"Host": "foreign.test"})
    assert denied_host.status_code == 403
    assert denied_host.json() == {"detail": "Unexpected proxy host"}
    denied_origin = api.get("/", headers={"Origin": "https://foreign.test"})
    assert denied_origin.status_code == 403
    assert denied_origin.json() == {"detail": "Unexpected proxy origin"}
    assert api.get("/", headers={"Origin": "null"}).status_code == 403
    assert api.get("/", headers={"Host": "foreign.test", "X-Forwarded-Host":
                                policy["host"]}).status_code == 403
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(stream, "PNG")
    response = api.post("/api/predict", headers={"Origin": origin}, json={
        "question": "Ảnh màu gì?", "image_base64": base64.b64encode(stream.getvalue()).decode(),
    })
    assert response.status_code == 200 and response.json()["answer"] == "xanh"


@pytest.fixture
def web_startup(tmp_path, monkeypatch):
    class Process:
        pid = 123
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, **kwargs):
            return 0
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps({"mode": "live", **IDENTITY}).encode()
    process = Process()
    monkeypatch.setattr(colab.socket, "create_connection",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("closed")))
    monkeypatch.setattr(colab.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(colab, "urlopen", lambda *args, **kwargs: Response())
    return process, {"command": ["fixture-python", "run_demo.py"], "cwd": tmp_path / "source",
                     "output_dir": tmp_path / "logs", "port": 7860,
                     "expected_identity": IDENTITY, "source_release": "source_fixture"}


def test_web_startup_records_identity_without_uploads_or_credentials(web_startup):
    process, arguments = web_startup
    actual, receipt = colab.start_web(**arguments)
    assert actual is process and receipt["status"] == "ready"
    assert receipt == json.loads((arguments["output_dir"] / "receipt.json").read_text())
    assert sorted(p.name for p in arguments["output_dir"].iterdir()) == [
        "receipt.json", "server.log",
    ]
    colab.stop_web(process)
    assert process.stopped
    with pytest.raises(FileExistsError):
        colab.start_web(**arguments)


def test_wrong_backend_identity_is_stopped_and_failure_is_preserved(web_startup, monkeypatch):
    process, arguments = web_startup
    arguments["expected_identity"] = {**IDENTITY, "model_revision": "other"}
    with pytest.raises(RuntimeError, match="identity"):
        colab.start_web(**arguments)
    assert process.stopped
    assert json.loads((arguments["output_dir"] / "receipt.json").read_text())["status"] == "failed"


def test_web_startup_refuses_an_occupied_port_without_creating_a_run(web_startup, monkeypatch):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass
    _, arguments = web_startup
    monkeypatch.setattr(colab.socket, "create_connection", lambda *args, **kwargs: Connection())
    with pytest.raises(RuntimeError, match="already in use"):
        colab.start_web(**arguments)
    assert not arguments["output_dir"].exists()


def test_web_startup_timeout_stops_only_its_child(web_startup, monkeypatch):
    process, arguments = web_startup
    moments = iter([0, 1, 300])
    monkeypatch.setattr(colab.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(colab.time, "sleep", lambda _: None)
    monkeypatch.setattr(colab, "urlopen",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not ready")))
    with pytest.raises(TimeoutError, match="startup"):
        colab.start_web(**arguments)
    assert process.stopped


def test_stop_escalates_only_after_its_child_timeout(web_startup, monkeypatch):
    process, _ = web_startup
    calls = []

    def wait(**kwargs):
        if not calls:
            calls.append("timeout")
            raise subprocess.TimeoutExpired("fixture", 15)
    monkeypatch.setattr(process, "wait", wait)
    monkeypatch.setattr(process, "kill", lambda: calls.append("kill"), raising=False)
    colab.stop_web(process)
    assert calls == ["timeout", "kill"]


def test_web_notebook_is_separate_and_does_not_train_or_extract_the_dataset():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/07_demo_web.ipynb").read_text("utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(notebook["cells"]) == 7 and len(cells) == 5
    for cell in cells:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    config = {}
    exec("".join(cells[0]["source"]), config)
    assert config["RUN_WEB"] is False
    assert config["SOURCE_RELEASE"] == config["SOURCE_SHA256"] == ""
    combined = "\n".join("".join(cell["source"]) for cell in cells)
    assert "serve_kernel_port_as_iframe" in combined
    assert "serve_kernel_port_as_window" not in combined
    start = next(cell for cell in cells if cell["id"] == "web-start")
    assert "window.location.origin" in "".join(start["source"])
    assert "serve_kernel_port_as_iframe" in "".join(start["source"])
    for unused in ("verify_coco_bundle", "train_preference", "run_study_validation"):
        assert unused not in combined
    from scripts.package_source import source_files

    assert root / "notebooks/07_demo_web.ipynb" in source_files(root)
