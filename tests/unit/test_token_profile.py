import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vqa_explanation.training import token_profile
from vqa_explanation.utils.environment import project_root


def test_measure_tokens_checks_image_and_label_alignment():
    assert token_profile.measure_tokens("q1", [1, 99, 2, 3], [-100, -100, 2, 3], 99) == {
        "question_id": "q1",
        "total_tokens": 4,
        "supervised_tokens": 2,
        "image_tokens": 1,
    }
    with pytest.raises(ValueError, match="alignment"):
        token_profile.measure_tokens("q1", [1, 99], [-100], 99)
    with pytest.raises(ValueError, match="Missing image"):
        token_profile.measure_tokens("q1", [1, 2], [-100, 2], 99)


def test_summary_uses_nearest_rank_and_reports_overflow():
    measured = [
        {
            "question_id": f"q{value}",
            "total_tokens": value,
            "supervised_tokens": 2,
            "image_tokens": 1,
        }
        for value in range(1, 101)
    ]
    result = token_profile.summarize(measured, 95)
    assert result["total_tokens"] == {
        "min": 1,
        "p50": 50,
        "p90": 90,
        "p95": 95,
        "p99": 99,
        "max": 100,
    }
    assert result["over_max_length"] == 5
    assert result["over_max_length_rate"] == 0.05
    assert result["longest"][0]["question_id"] == "q100"


def test_profile_with_fake_processor_is_deterministic_and_never_overwrites(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.jsonl"
    records = [
        {
            "question_id": f"q{i}",
            "image_id": f"i{i}",
            "image_path": f"images/{i}.jpg",
            "question": "Ảnh có gì?",
            "answer": "Con mèo",
            "explanations": ["Ảnh có con mèo."],
            "split": "train" if i < 4 else ("validation" if i == 4 else "test"),
        }
        for i in range(6)
    ]
    annotations.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    training_config = tmp_path / "readiness.yaml"
    training_config.write_text(
        "seed: 42\nsample_size: 2\nmin_pixels: 200704\nmax_pixels: 200704\nmax_length: 6\n",
        encoding="utf-8",
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model:\n  id: fake/model\n  revision: fixed\n", encoding="utf-8")

    class FakeProcessor:
        tokenizer = SimpleNamespace(get_vocab=lambda: {"<|image_pad|>": 99})

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeCollator:
        def __init__(self, processor, image_root, max_length):
            assert max_length == 10**9

        def __call__(self, selected):
            value = int(selected[0]["question_id"][1:])
            tokens = [1, 99, 2, 3] + [4] * value
            return {"input_ids": [tokens], "labels": [[-100, -100, 2, 3] + [4] * value]}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=FakeProcessor))
    monkeypatch.setattr(token_profile, "VQACollator", FakeCollator)
    monkeypatch.setattr(token_profile.importlib.metadata, "version", lambda _: "test-version")
    output = tmp_path / "profile.json"
    first = token_profile.profile(annotations, images, output, training_config, model_config)
    assert first["status"] == "passed"
    assert first["train_records"] == 4
    assert first["sample_size"] == 2
    assert first["summary"]["samples"] == 2
    assert first["selected_question_ids_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == first
    with pytest.raises(FileExistsError):
        token_profile.profile(annotations, images, output, training_config, model_config)

    assert "Train sample" in first["note"]
    training_config.write_text(
        "seed: 42\nsample_size: 4\nmin_pixels: 200704\nmax_pixels: 200704\nmax_length: 6\n",
        encoding="utf-8",
    )
    full = token_profile.profile(
        annotations, images, tmp_path / "full.json", training_config, model_config
    )
    assert full["sample_size"] == full["train_records"] == 4
    assert "Full train coverage" in full["note"]
    assert "test excluded" in full["note"]
    assert full["validation_sample_size"] == 0
    combined = token_profile.profile(
        annotations, images, tmp_path / "combined.json", training_config, model_config,
        validation_count=1,
    )
    assert combined["sample_size"] == combined["train_records"] == 4
    assert combined["validation_question_ids"] == ["q4"]
    assert combined["summary"]["samples"] == 5
    assert combined["train_summary"]["samples"] == 4
    assert combined["validation_summary"]["samples"] == 1
    assert combined["validation_summary"]["over_max_length"] == 1
    for invalid in (-1, True, 2):
        with pytest.raises(ValueError, match="validation_count"):
            token_profile.profile(
                annotations, images, tmp_path / f"invalid_{invalid}.json",
                training_config, model_config, validation_count=invalid,
            )


def test_token_profile_stage_dry_run_from_outside_repo(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "token-profile",
            "--environment",
            "configs/environments/local_full.yaml",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "profile_sft_tokens.py" in result.stdout
    assert "sft_readiness.yaml" in result.stdout
    assert not list(tmp_path.iterdir())
