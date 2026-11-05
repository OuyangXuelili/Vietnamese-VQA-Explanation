"""Keep numerical generation guard changes in export resume provenance."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vqa_explanation.inference import export

GUARD_SOURCE = "src/vqa_explanation/inference/generation_safety.py"


@pytest.fixture
def export_case(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    records = [{
        "question_id": "q1", "image_id": "image1", "image_path": "image.jpg",
        "question": "What is visible?", "answer": "A cat",
        "explanations": ["A cat is in the image."], "split": "validation",
    }]
    annotations.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"offline image fixture; no model or decoder is used")
    args = SimpleNamespace(
        adapter=None, model_id="offline-model", revision="offline-revision",
        experiment_id="M0_PRETRAINED", split="validation", limit=0, seed=42,
        min_pixels=256, max_pixels=1024, max_new_tokens=32, load_in_4bit=False,
        annotations=annotations, image_root=tmp_path,
        output=tmp_path / "predictions.jsonl", resume=False,
    )
    hashes = {"q1": hashlib.sha256(image.read_bytes()).hexdigest()}
    return args, records, hashes


def test_guard_hash_matches_file_contents(export_case):
    args, records, hashes = export_case
    inputs = export.export_inputs(args, records, hashes)
    guard = Path(__file__).resolve().parents[2] / GUARD_SOURCE

    assert inputs["source_files"][GUARD_SOURCE] == hashlib.sha256(guard.read_bytes()).hexdigest()


@pytest.fixture
def copied_sources(export_case, tmp_path, monkeypatch):
    args, records, hashes = export_case
    inputs = export.export_inputs(args, records, hashes)
    workspace = Path(__file__).resolve().parents[2]
    root = tmp_path / "source"
    for name in inputs["source_files"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((workspace / name).read_bytes())
    monkeypatch.setattr(export, "project_root", lambda: root)
    return root


def test_helper_only_change_changes_export_inputs(export_case, copied_sources):
    args, records, hashes = export_case
    before = export.export_inputs(args, records, hashes)
    guard = copied_sources / GUARD_SOURCE
    guard.write_bytes(guard.read_bytes() + b"\n# provenance regression edit\n")

    after = export.export_inputs(args, records, hashes)

    assert before != after
    assert after["source_files"][GUARD_SOURCE] == hashlib.sha256(guard.read_bytes()).hexdigest()
    after["source_files"][GUARD_SOURCE] = before["source_files"][GUARD_SOURCE]
    assert after == before


def test_resume_rejects_helper_change_before_runner_construction(export_case, copied_sources):
    args, _, _ = export_case
    runner = Mock()
    runner.protocol_metadata.return_value = {"prompt_version": "offline-fixture"}
    runner.generate.return_value = "Answer: A cat\nExplanation: A cat is in the image."
    export.export_predictions(args, runner_factory=Mock(return_value=runner))
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    original_output = args.output.read_bytes()
    original_manifest = manifest_path.read_bytes()
    guard = copied_sources / GUARD_SOURCE
    guard.write_bytes(guard.read_bytes() + b"\n# helper-only change after export\n")
    args.resume = True
    runner_factory = Mock(side_effect=AssertionError("Runner must not be constructed"))

    with pytest.raises(SystemExit, match="Resume inputs/source/adapter/environment changed"):
        export.export_predictions(args, runner_factory=runner_factory)

    runner_factory.assert_not_called()
    assert args.output.read_bytes() == original_output
    assert manifest_path.read_bytes() == original_manifest
