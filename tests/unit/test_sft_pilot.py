import subprocess
import sys

import pytest

from vqa_explanation.training.sft_pilot import load_config, pilot_record
from vqa_explanation.utils.environment import project_root


def _record(question_id: str, image_id: str, split: str) -> dict:
    return {
        "question_id": question_id,
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "question": "Ảnh có gì?",
        "answer": "một vật",
        "explanations": ["Vật xuất hiện trong ảnh."],
        "split": split,
    }


def test_pilot_record_selects_only_explicit_train_id():
    rows = [_record("q1", "i1", "train"), _record("q2", "i2", "validation")]
    assert pilot_record(rows, "q1", 42)["question_id"] == "q1"
    with pytest.raises(ValueError, match="not a unique train record"):
        pilot_record(rows, "q2", 42)


def test_pilot_record_rejects_image_overlap():
    rows = [_record("q1", "i1", "train"), _record("q2", "i1", "validation")]
    with pytest.raises(ValueError, match="overlap"):
        pilot_record(rows, "q1", 42)


def test_pilot_config_rejects_invalid_values(tmp_path):
    cfg = tmp_path / "pilot.yaml"
    cfg.write_text(
        "seed: 42\nquestion_id: q1\nmin_pixels: 200704\n"
        "max_pixels: 100000\nmax_length: 1536\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_pixels exceeds"):
        load_config(cfg)


def test_pilot_dry_run_uses_explicit_config_without_creating_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/run_pipeline.py"),
            "sft-batch-pilot",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pilot_sft_batch.py" in result.stdout
    assert "sft_batch_pilot.yaml" in result.stdout
    assert "batch_pilot.json" in result.stdout
    assert not list(tmp_path.iterdir())
