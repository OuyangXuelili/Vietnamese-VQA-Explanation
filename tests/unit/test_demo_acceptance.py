"""Acceptance orchestration tests use fixtures, not evidence of a GPU run."""

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from vqa_explanation.demo import acceptance
from vqa_explanation.demo.service import LiveService
from vqa_explanation.training.preference_data import adapter_fingerprint
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def demo_files(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
    records, predictions = [], []
    protocol = {"min_pixels": 200704, "max_pixels": 200704, "max_new_tokens": 128}
    identity = {"experiment_id": "M1_SFT", "adapter": "M1", "model_id": "fixture-model",
                "model_revision": "pinned-revision", "seed": 42}
    hashes = {}
    for index in range(5):
        path = images / f"image{index}.png"
        Image.new("RGB", (16, 16), (index, 128, 0)).save(path)
        record = {"question_id": f"q{index}", "image_id": f"i{index}",
                  "image_path": path.name, "question": "Ảnh màu gì?", "answer": "xanh",
                  "explanations": ["ảnh xanh"], "split": "validation"}
        records.append(record)
        hashes[record["question_id"]] = sha256_file(path)
        predictions.append({**record, **identity, **protocol, "image_sha256": sha256_file(path),
            "reference_answer": "xanh", "reference_explanations": ["ảnh xanh"],
            "raw_output": "Đáp án: xanh\nGiải thích: ảnh xanh", "prediction_answer": "xanh",
            "prediction_explanation": "ảnh xanh", "format_valid": True, "latency_seconds": 1})
    annotations, saved = tmp_path / "validation.jsonl", tmp_path / "predictions.jsonl"
    for path, rows in ((annotations, records), (saved, predictions)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": {**identity, "split": "validation",
        "adapter_files": adapter_fingerprint(adapter)["files"], "image_hashes": hashes,
        "annotations_sha256": sha256_file(annotations)}, "protocol": protocol}), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "selected", "selection_split": "validation",
        "model_id": identity["model_id"], "model_revision": identity["model_revision"],
        "adapter": adapter_fingerprint(adapter), "evidence": {
            key: {"sha256": sha256_file(path)} for key, path in (
                ("validation_predictions", saved), ("validation_manifest", manifest),
                ("validation_annotations", annotations))}}), encoding="utf-8")
    return dict(predictions=saved, manifest=manifest, annotations=annotations,
                image_root=images, selection=selection, adapter=adapter, replay_count=1)


def test_inspect_is_read_only_and_excludes_gallery(demo_files, monkeypatch):
    before = {str(path): sha256_file(path) for path in demo_files["image_root"].parent.rglob("*")
              if path.is_file()}
    monkeypatch.setattr(acceptance, "load_live_service", lambda *args: pytest.fail("No GPU load"))
    plan = acceptance.inspect_demo(**demo_files)
    assert [case["question_id"] for case in plan["cases"]] == ["q1", "q2", "q3"]
    assert plan["scope"] == "validation_only"
    assert before == {str(path): sha256_file(path)
                      for path in demo_files["image_root"].parent.rglob("*") if path.is_file()}
    assert acceptance.inspect_demo(**demo_files) == plan


def test_changed_adapter_or_case_is_rejected(demo_files):
    path = demo_files["image_root"] / "image1.png"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        acceptance.inspect_demo(**demo_files)
    (demo_files["adapter"] / "adapter_config.json").write_text("changed")
    with pytest.raises(ValueError, match="Adapter"):
        acceptance.inspect_demo(**demo_files)


def test_insufficient_distinct_images_is_not_filled_with_replay(demo_files):
    with pytest.raises(ValueError, match="three distinct"):
        acceptance.inspect_demo(**{**demo_files, "replay_count": 3})


def test_output_cannot_overwrite_inputs_or_old_runs(demo_files, tmp_path):
    plan = acceptance.inspect_demo(**demo_files)
    for path in (tmp_path, demo_files["image_root"] / "output", demo_files["adapter"] / "check"):
        with pytest.raises(ValueError, match="protected"):
            acceptance.check_output_path(path, plan)
    output = tmp_path / "old_check"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new demo check"):
        acceptance.check_output_path(output, plan)


def test_cpu_stops_before_model_load_or_output_creation(demo_files, tmp_path, monkeypatch):
    import torch

    plan = acceptance.inspect_demo(**demo_files)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(acceptance, "load_live_service", lambda *args: pytest.fail("No CPU load"))
    output = tmp_path / "cpu_check"
    with pytest.raises(RuntimeError, match="CUDA"):
        acceptance.run_demo_check(plan, output, expected_sha256=plan["sha256"])
    assert not output.exists()


def test_changed_inspection_stops_before_running(demo_files, tmp_path):
    plan = acceptance.inspect_demo(**demo_files)
    with pytest.raises(ValueError, match="inspect again"):
        acceptance.run_demo_check(plan, tmp_path / "bad", expected_sha256="0" * 64)
    plan["cases"][0]["question"] = "changed"
    with pytest.raises(ValueError, match="inspect again"):
        acceptance.run_demo_check(plan, tmp_path / "bad", expected_sha256=plan["sha256"])
    assert not (tmp_path / "bad").exists()


@pytest.fixture
def fake_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fixture GPU")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)


def test_real_http_handlers_and_reload_with_fixture_service(
    demo_files, tmp_path, monkeypatch, fake_gpu,
):
    calls = []

    class Runner:
        def generate(self, path, question, **kwargs):
            assert path.is_file() and question == "Ảnh màu gì?"
            return "Đáp án: xanh\nGiải thích: ảnh xanh"

    def load(*args):
        calls.append(args)
        return LiveService(Runner(), {"experiment_id": "M1_SFT"})

    monkeypatch.setattr(acceptance, "load_live_service", load)
    plan = acceptance.inspect_demo(**demo_files)
    output = tmp_path / "check"
    report = acceptance.run_demo_check(plan, output, expected_sha256=plan["sha256"])
    assert len(calls) == 2
    assert report["status"] == "technical_passed"
    assert len(report["cases"]) == 3 and len(report["reload"]) == 1
    assert report["model_quality_reviewed"] is report["browser_ui_reviewed"] is False
    assert report == json.loads((output / "report.json").read_text("utf-8"))
    assert len((output / "results.jsonl").read_text("utf-8").splitlines()) == 4


def test_failed_inference_keeps_error_and_partial_outputs(
    demo_files, tmp_path, monkeypatch, fake_gpu,
):
    class Runner:
        count = 0

        def generate(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("fixture failure")
            return "Đáp án: xanh\nGiải thích: ảnh xanh"

    monkeypatch.setattr(acceptance, "load_live_service",
                        lambda *args: LiveService(Runner(), {"experiment_id": "M1_SFT"}))
    plan = acceptance.inspect_demo(**demo_files)
    output = tmp_path / "failed_check"
    with pytest.raises(RuntimeError, match="HTTP 503"):
        acceptance.run_demo_check(plan, output, expected_sha256=plan["sha256"])
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert report["status"] == "failed" and report["error"]["type"] == "RuntimeError"
    assert len((output / "results.jsonl").read_text("utf-8").splitlines()) == 1


def test_config_rejects_unresolved_environment(tmp_path, monkeypatch):
    config = tmp_path / "demo.yaml"
    config.write_text("examples: 12\npredictions: ${VQA_DEMO_MISSING}/file\n")
    monkeypatch.delenv("VQA_DEMO_MISSING", raising=False)
    with pytest.raises(ValueError, match="Unresolved"):
        acceptance.load_demo_config(config)


def test_demo_notebook_is_thin_and_does_not_run_by_default():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/06_demo_live.ipynb").read_text("utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(notebook["cells"]) == 11 and len(cells) == 7
    for cell in cells:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    config = next(cell for cell in cells if cell["id"] == "demo-config")
    namespace = {}
    exec("".join(config["source"]), namespace)
    assert namespace["RUN_DEMO_CHECK"] is False and namespace["PLAN"] is None
    assert namespace["SOURCE_RELEASE"] == namespace["SOURCE_SHA256"] == ""
    combined = "\n".join("".join(cell["source"]) for cell in cells)
    assert combined.count("drive.mount(") == 1
    assert "train_preference.py" not in combined and "run_study_validation.py" not in combined
    environment = root / "configs/environments/demo_colab.example.yaml"
    assert environment.is_file()
    assert "configs/environments/demo_colab.example.yaml" in combined
    from scripts.package_source import source_files

    assert root / "notebooks/06_demo_live.ipynb" in source_files(root)
