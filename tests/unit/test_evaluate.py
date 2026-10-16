import hashlib
import json
import os

import pytest

from scripts import evaluate as evaluate_script
from scripts.evaluate import evaluate_predictions, read_predictions


def test_cli_uses_headless_backend_instead_of_inheriting_colab(tmp_path, monkeypatch):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"prediction_answer": "cat", "reference_answer": "cat"}), "utf-8")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.setattr("sys.argv", ["evaluate.py", str(path)])
    evaluate_script.main()
    assert os.environ["MPLBACKEND"] == "Agg"


def test_evaluate_predictions_reports_answer_format_and_latency_metrics() -> None:
    records = [
        {
            "prediction_answer": "Con mèo!",
            "reference_answer": "con mèo",
            "format_valid": True,
            "latency_seconds": 2.0,
        },
        {
            "prediction_answer": "một con mèo đen",
            "reference_answer": "con mèo",
            "format_valid": False,
            "latency_seconds": 4.0,
        },
    ]

    metrics = evaluate_predictions(records)

    assert metrics["n"] == 2
    assert metrics["exact_match"] == 0.5
    assert metrics["token_f1"] == pytest.approx(5 / 6)
    assert metrics["format_rate"] == 0.5
    assert metrics["format_valid_n"] == 2
    assert metrics["mean_latency_seconds"] == 3.0
    assert metrics["median_latency_seconds"] == 3.0
    assert metrics["explanation_status"] == "unavailable"
    assert metrics["explanation_rouge_l_f1"] is None


def test_read_predictions_reports_invalid_json_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prediction_answer": "cat"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_predictions(path)


def test_read_predictions_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    record = {"prediction_answer": "cat", "reference_answer": "cat"}
    path.write_text(f"\n{json.dumps(record)}\n\n", encoding="utf-8")

    assert read_predictions(path) == [record]


def test_evaluate_explanations_requires_complete_records():
    rows = [
        {
            "prediction_answer": "mèo",
            "reference_answer": "mèo",
            "prediction_explanation": "Có mèo trên bàn.",
            "reference_explanations": ["Mèo ở trên bàn."],
        },
        {
            "prediction_answer": "chó",
            "reference_answer": "chó",
            "prediction_explanation": "",
            "reference_explanations": ["Có chó trong ảnh."],
        },
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["explanation_status"] == "scored"
    assert metrics["explanation_n"] == 2
    assert metrics["explanation_rouge_l_f1"] == pytest.approx(0.375)

    del rows[1]["prediction_explanation"]
    with pytest.raises(ValueError, match="Record 2"):
        evaluate_predictions(rows)
    del rows[1]["reference_explanations"]
    with pytest.raises(ValueError, match="every prediction record"):
        evaluate_predictions(rows)


def test_format_rate_rejects_non_boolean_or_partial_values():
    rows = [
        {"prediction_answer": "mèo", "reference_answer": "mèo", "format_valid": "false"},
        {"prediction_answer": "chó", "reference_answer": "chó", "format_valid": False},
    ]
    with pytest.raises(ValueError, match="boolean"):
        evaluate_predictions(rows)
    del rows[0]["format_valid"]
    with pytest.raises(ValueError, match="every prediction record"):
        evaluate_predictions(rows)


def test_evaluate_rejects_duplicate_or_partial_question_ids():
    rows = [
        {"question_id": "q1", "prediction_answer": "mèo", "reference_answer": "mèo"},
        {"question_id": "q1", "prediction_answer": "chó", "reference_answer": "chó"},
    ]
    with pytest.raises(ValueError, match="Duplicate question_id"):
        evaluate_predictions(rows)
    del rows[1]["question_id"]
    with pytest.raises(ValueError, match="every record"):
        evaluate_predictions(rows)
    rows[1]["question_id"] = "   "
    with pytest.raises(ValueError, match="every record"):
        evaluate_predictions(rows)


@pytest.mark.parametrize("reference", ["", "   ", "!!!"])
def test_evaluate_rejects_empty_reference_after_normalization(reference):
    with pytest.raises(ValueError, match="empty reference_answer"):
        evaluate_predictions([{"prediction_answer": reference, "reference_answer": reference}])


def test_evaluate_cli_never_overwrites_existing_report(tmp_path, monkeypatch, capsys):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"prediction_answer": "mèo", "reference_answer": "mèo"}) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "metrics.json"
    report.write_text("historical report", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["evaluate.py", str(predictions), "--output", str(report)]
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate_script.main()
    assert exit_info.value.code == 2
    assert "Output exists" in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == "historical report"


def test_evaluate_cli_runs_pinned_semantic_metric_on_the_selected_split(
    tmp_path, monkeypatch
):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "question_id": "q1",
            "image_id": "i1",
            "split": "validation",
            "prediction_answer": "mèo",
            "reference_answer": "mèo",
            "prediction_explanation": "Có mèo.",
            "reference_explanations": ["Trong ảnh có mèo."],
        },
        {
            "question_id": "q2",
            "image_id": "i2",
            "split": "test",
            "prediction_answer": "chó",
            "reference_answer": "chó",
            "prediction_explanation": "Có chó.",
            "reference_explanations": ["Trong ảnh có chó."],
        },
    ]
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = tmp_path / "metrics.json"
    received = []

    def fake_bertscore(records, **kwargs):
        received.append((records, kwargs))
        return {"status": "scored", "n": len(records), "mean_f1": 0.75}

    monkeypatch.setattr(evaluate_script, "evaluate_bertscore", fake_bertscore)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            str(predictions),
            "--split",
            "validation",
            "--bertscore",
            "--bertscore-device",
            "cpu",
            "--bertscore-batch-size",
            "8",
            "--output",
            str(report_path),
        ],
    )

    evaluate_script.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"] == "validation"
    assert len(report["predictions_sha256"]) == 64
    assert report["explanation_bertscore"]["mean_f1"] == 0.75
    assert [row["question_id"] for row in received[0][0]] == ["q1"]
    assert received[0][1]["device"] == "cpu"
    assert received[0][1]["batch_size"] == 8


def test_evaluate_hash_describes_scored_snapshot_if_input_changes(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    payload = json.dumps({"prediction_answer": "cat", "reference_answer": "cat"}).encode()
    predictions.write_bytes(payload)
    output = tmp_path / "metrics.json"

    def replace_input(records, **kwargs):
        predictions.write_text('{"different": true}', encoding="utf-8")
        return {"status": "scored"}

    monkeypatch.setattr(evaluate_script, "evaluate_bertscore", replace_input)
    monkeypatch.setattr("sys.argv", [
        "evaluate.py", str(predictions), "--bertscore", "--output", str(output),
    ])
    evaluate_script.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["predictions_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["exact_match"] == 1.0
