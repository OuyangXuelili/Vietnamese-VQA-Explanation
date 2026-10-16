import copy
import json

import pytest

from vqa_explanation.preference.quality import (
    assess_reviews,
    candidate_origin,
    pair_digest,
    review_template,
    text_only_screen,
)


def test_candidate_lineage_uses_the_saved_adapter_files(tmp_path):
    from vqa_explanation.training.preference_data import adapter_fingerprint

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture weights")
    fingerprint = adapter_fingerprint(tmp_path)
    inputs = {"experiment_id": "M1_SFT", "split": "train", "model_id": "model",
              "model_revision": "a" * 40, "adapter_files": fingerprint["files"], "seed": 42}
    assert candidate_origin(inputs)["checkpoint_sha256"] == fingerprint["sha256"]
    with pytest.raises(ValueError, match="pinned train M1"):
        candidate_origin({**inputs, "split": "validation"})
    with pytest.raises(ValueError, match="pinned train M1"):
        candidate_origin({**inputs, "model_revision": "main"})


def test_sample_audit_rejects_missing_or_stale_findings(tmp_path):
    from scripts.audit_preference_quality import sample_findings

    rows = pairs(2)
    record = {
        "schema": "preference_sample_audit_v1", "sample_sha256": "a" * 64,
        "reviewer_type": "ai", "independent_human_verified": False, "blinded": False,
        "columns": ["sample_index", "relevance", "text_conflict", "visual", "reason"],
        "rows": [[1, "direct", "no", "supported", "Visible queried feature."],
                 [2, "unrelated", "no", "supported", "Only a background detail changes."]],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    findings, summary = sample_findings(path, rows, "a" * 64)
    assert summary["needs_followup"] == 1 and summary["training_labels_created"] == 0
    assert findings[0]["pair_sha256"] == pair_digest(rows[0])
    with pytest.raises(ValueError, match="declared sample"):
        sample_findings(path, rows, "b" * 64)
    record["rows"][1][0] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="each sampled pair"):
        sample_findings(path, rows, "a" * 64)


def pairs(count=30):
    return [{
        "pair_id": f"p{i:03}", "source_question_id": f"q{i}",
        "image_id": f"i{i}", "image_path": f"images/{i}.jpg", "image_sha256": "a" * 64,
        "split": "train", "review_status": "accepted", "prompt": "Màu gì?",
        "error_type": "attribute", "generation_method": "fixture",
        "chosen": f"Đáp án: đỏ\nGiải thích: Áo đỏ có đường viền tốt {i}.",
        "rejected": f"Đáp án: đỏ\nGiải thích: Áo xanh có đường viền sai {i}.",
    } for i in range(count)]


def confirmed(rows):
    reviews = review_template(rows)
    for review in reviews:
        review.update(status="confirmed", reviewer_type="human", visible_evidence="Áo đỏ.",
                      rejected_span="Áo xanh", answer_correct=True, chosen_supported=True,
                      rejected_unsupported=True, both_linguistically_plausible=True,
                      language_only_preference="undetermined", contrast_relevant=True,
                      relevance_reason="The changed attribute belongs to the queried clothing.",
                      text_only_reason="Both colors are possible without seeing the image.")
    return reviews


def test_template_cannot_pass_as_completed_review():
    rows = pairs()
    result = assess_reviews(rows, review_template(rows), 20)
    assert not result["ready"] and result["confirmed"] == 0
    assert len(result["needs_review"]) == 30
    assert assess_reviews(rows, confirmed(rows), 20)["ready"]
    assert not assess_reviews(rows, confirmed(rows), 500)["ready"]


@pytest.mark.parametrize("key,value", [
    ("answer_correct", "true"), ("chosen_supported", False),
    ("language_only_preference", "chosen"), ("reviewer_type", None),
    ("rejected_span", "not in explanation"), ("visible_evidence", ""),
    ("contrast_relevant", False), ("relevance_reason", ""), ("text_only_reason", ""),
])
def test_quality_gate_does_not_accept_missing_or_easy_pairs(key, value):
    rows = pairs()
    reviews = confirmed(rows)
    reviews[0][key] = value
    assert not assess_reviews(rows, reviews, 20)["ready"]


def test_main_review_version_rejects_legacy_self_attestations():
    rows = pairs()
    reviews = confirmed(rows)
    for review in reviews:
        for field in ("review_version", "contrast_relevant", "relevance_reason",
                      "text_only_reason"):
            review.pop(field)
    assert assess_reviews(rows, reviews, 20)["ready"]
    result = assess_reviews(rows, reviews, 20, required_version=2)
    assert not result["ready"] and result["confirmed"] == 0
    assert "review_version" in result["issue_details"][0]["failed_checks"]


def test_review_is_bound_to_exact_completions_and_image():
    rows = pairs()
    reviews = confirmed(rows)
    rows[0]["chosen"] += " Updated."
    with pytest.raises(ValueError, match="stale"):
        assess_reviews(rows, reviews, 20)
    changed = copy.deepcopy(rows[0])
    changed["image_sha256"] = "b" * 64
    assert pair_digest(rows[0]) != pair_digest(changed)


def test_quality_review_rejects_extra_and_duplicate_ids():
    rows = pairs()
    with pytest.raises(ValueError, match="duplicate"):
        assess_reviews(rows, confirmed(rows) * 2, 20)
    reviews = confirmed(rows)
    reviews[0]["pair_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        assess_reviews(rows, reviews, 20)


def test_screen_detects_easy_text_signal_and_is_order_independent():
    rows = pairs()
    result = text_only_screen(rows, bootstrap=50)
    assert result == text_only_screen(rows[::-1], bootstrap=50)
    assert result["pair_accuracy"] > 0.9
    assert result["image_bootstrap_ci95"][0] <= result["pair_accuracy"] <= (
        result["image_bootstrap_ci95"][1]
    )


def test_screen_does_not_split_repeated_images_across_folds():
    rows = pairs()
    rows[1]["image_id"] = rows[0]["image_id"]
    rows[2]["image_path"] = rows[1]["image_path"]
    result = text_only_screen(rows, bootstrap=10)
    assignments = {r["pair_id"]: r for r in result["rows"]}
    assert len({assignments[f"p{i:03}"]["fold"] for i in range(3)}) == 1
    assert len({assignments[f"p{i:03}"]["group"] for i in range(3)}) == 1
    assert not assess_reviews(rows, confirmed(rows), 20)["ready"]


@pytest.mark.parametrize("fault", ["split", "answer", "duplicate"])
def test_screen_rejects_noncanonical_pair_inputs(fault):
    rows = pairs()
    if fault == "split":
        rows[0]["split"] = "validation"
    elif fault == "answer":
        rows[0]["rejected"] = "Đáp án: xanh\nGiải thích: Áo xanh."
    else:
        rows[0]["pair_id"] = rows[1]["pair_id"]
    with pytest.raises(ValueError):
        text_only_screen(rows)
