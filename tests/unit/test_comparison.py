import hashlib
import json
from copy import deepcopy

import pytest

from scripts.compare_runs import main as compare_runs_main
from scripts.compare_runs import read_expected_validation_ids
from vqa_explanation.evaluation.comparison import cluster_bootstrap, compare_predictions


def test_cluster_bootstrap_weights_questions_and_preserves_image_clusters():
    result = cluster_bootstrap({'em': [1, 1, -1]}, ['a', 'a', 'b'], seed=42, resamples=500)
    assert result['n_questions'] == 3 and result['n_image_clusters'] == 2
    assert result['metrics']['em']['delta_right_minus_left'] == pytest.approx(1 / 3)
    assert result['metrics']['em']['ci95'] == [-1, 1]
    assert result == cluster_bootstrap({'em': [-1, 1, 1]}, ['b', 'a', 'a'],
                                       seed=42, resamples=500)


@pytest.mark.parametrize('differences,images', [
    ({'em': [1]}, ['a']), ({'em': [float('nan'), 0]}, ['a', 'b']),
    ({'em': [1]}, ['a', 'b']), ({}, ['a', 'b']),
])
def test_cluster_bootstrap_rejects_invalid_inputs(differences, images):
    with pytest.raises(ValueError):
        cluster_bootstrap(differences, images)


def sample(qid="q1"):
    return dict(
        question_id=qid,
        image_id="image1",
        image_path="images/image1.jpg",
        image_sha256="a" * 64,
        question="Con gì?",
        reference_answer="mèo",
        reference_explanations=["Có mèo."],
        prediction_answer="mèo",
        prediction_explanation="Có mèo.",
        split="validation",
        model_id="model",
        model_revision="revision",
        seed=20260918,
        max_pixels=200704,
        min_pixels=200704,
        max_new_tokens=128,
        prompt_sha256="f" * 64,
        generation_config={"do_sample": False},
        load_in_4bit=True,
        compute_dtype="torch.float16",
        experiment_id="M0_PRETRAINED",
        adapter=None,
    )


def adapter_sample(qid="q1"):
    row = sample(qid)
    row["experiment_id"] = "M1_SFT"
    row["adapter"] = "runs/m1/adapter"
    return row


def annotation(qid="q1"):
    row = sample(qid)
    return {
        "question_id": qid,
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "question": row["question"],
        "answer": row["reference_answer"],
        "explanations": row["reference_explanations"],
        "split": "validation",
    }


def test_m1_checkpoint_comparison_keeps_canonical_and_protocol_checks():
    left = [adapter_sample(q) for q in ("q1", "q2")]
    right = [{**r, "adapter": "runs/m1/checkpoint-3683"} for r in left]
    refs = {q: annotation(q) for q in ("q1", "q2")}
    options = dict(expected_annotations=refs, expected_experiments=("M1_SFT", "M1_SFT"),
                   checkpoint_comparison=True)
    report = compare_predictions(left, right[::-1], **options)
    assert report["comparable"] and report["canonical_references_checked"]
    assert report["comparison_kind"] == "m1_checkpoint_validation"
    assert not compare_predictions(left, left, **options)["comparable"]
    assert not compare_predictions(left[:1], right[:1], **options)["comparable"]
    right[0]["max_new_tokens"] = 64
    assert not compare_predictions(left, right, **options)["comparable"]


@pytest.mark.parametrize("options", [
    {"expected_experiments": ("M1_SFT", "M1_SFT")},
    {"expected_annotations": {"q1": annotation()},
     "expected_experiments": ("M3_DPO", "M3_DPO")},
    {"expected_annotations": {"q1": {**annotation(), "split": "test"}},
     "expected_experiments": ("M1_SFT", "M1_SFT"), "expected_split": "test"},
])
def test_checkpoint_mode_rejects_unpinned_other_roles_and_test(options):
    with pytest.raises(ValueError):
        compare_predictions([adapter_sample()], [adapter_sample()],
                            checkpoint_comparison=True, **options)


def test_checkpoint_comparison_cli(tmp_path, monkeypatch):
    from scripts import compare_runs

    refs, left, right = [tmp_path / name for name in
                         ("validation.jsonl", "left.jsonl", "right.jsonl")]
    refs.write_text(json.dumps(annotation()) + "\n", encoding="utf-8")
    left.write_text(json.dumps(adapter_sample()) + "\n", encoding="utf-8")
    right.write_text(json.dumps({**adapter_sample(), "adapter": "runs/m1/checkpoint-3683"})
                     + "\n", encoding="utf-8")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(compare_runs, "FULL_VALIDATION_COUNT", 1)
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", str(left), str(right), "--output", str(output),
        "--expected-annotations", str(refs), "--expected-sha256",
        hashlib.sha256(refs.read_bytes()).hexdigest(), "--split", "validation",
        "--left-experiment", "M1_SFT", "--right-experiment", "M1_SFT",
        "--checkpoint-comparison",
    ])
    with pytest.raises(SystemExit) as done:
        compare_runs.main()
    assert done.value.code == 0
    assert json.loads(output.read_text())["comparison_kind"] == "m1_checkpoint_validation"


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("experiments", [
    ("M0_PRETRAINED", "M1_SFT"), ("M0_PRETRAINED", "M2_SFT_CHOSEN"),
    ("M0_PRETRAINED", "M3_DPO"), ("M1_SFT", "M2_SFT_CHOSEN"),
    ("M1_SFT", "M3_DPO"), ("M2_SFT_CHOSEN", "M3_DPO"),
])
def test_canonical_comparison_all_model_pairs(split, experiments):
    exports = []
    for experiment in experiments:
        exports.append([{**sample(qid), "split": split, "experiment_id": experiment,
                         "adapter": None if experiment == "M0_PRETRAINED"
                         else f"runs/{experiment}/adapter"} for qid in ("q1", "q2")])
    refs = {qid: {**annotation(qid), "split": split} for qid in ("q1", "q2")}
    report = compare_predictions(exports[0], exports[1][::-1], expected_annotations=refs,
                                 expected_split=split, expected_experiments=experiments)
    assert report["comparable"] and report["canonical_references_checked"]
    assert report["expected_split"] == split
    assert report["expected_experiments"] == list(experiments)
    assert report["delta_right_minus_left"]["exact_match"] == 0
    exports[1][0]["reference_answer"] = "chó"
    assert not compare_predictions(exports[0], exports[1], expected_annotations=refs,
                                   expected_split=split,
                                   expected_experiments=experiments)["comparable"]


@pytest.mark.parametrize("fault", ["subset", "role", "adapter", "split"])
def test_test_comparison_rejects_invalid_m2_m3_exports(fault):
    left = [{**adapter_sample(q), "split": "test", "experiment_id": "M2_SFT_CHOSEN"}
            for q in ("q1", "q2")]
    right = [{**r, "experiment_id": "M3_DPO", "adapter": "runs/m3"} for r in left]
    refs = {q: {**annotation(q), "split": "test"} for q in ("q1", "q2")}
    if fault == "subset":
        left.pop()
        right.pop()
    elif fault == "role":
        right[0]["experiment_id"] = "M2_SFT_CHOSEN"
    elif fault == "adapter":
        right[0]["adapter"] = None
    else:
        for row in left + right:
            row["split"] = "validation"
    report = compare_predictions(left, right, expected_annotations=refs, expected_split="test",
                                 expected_experiments=("M2_SFT_CHOSEN", "M3_DPO"))
    assert not report["comparable"]
    assert report["scores"] is None


def test_generic_reference_cli_checks_hash_and_count(tmp_path, monkeypatch):
    from scripts import compare_runs

    left, right, refs = tmp_path / "left.jsonl", tmp_path / "right.jsonl", tmp_path / "test.jsonl"
    annotation_row = {**annotation(), "split": "test"}
    refs.write_text(json.dumps(annotation_row) + "\n", encoding="utf-8")
    for path, experiment in ((left, "M2_SFT_CHOSEN"), (right, "M3_DPO")):
        row = {**adapter_sample(), "split": "test", "experiment_id": experiment,
               "adapter": f"runs/{experiment}"}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    digest = hashlib.sha256(refs.read_bytes()).hexdigest()
    argv = ["compare_runs.py", str(left), str(right), "--output", str(report_path),
            "--expected-annotations", str(refs), "--expected-sha256", digest,
            "--split", "test", "--left-experiment", "M2_SFT_CHOSEN",
            "--right-experiment", "M3_DPO"]
    monkeypatch.setattr("sys.argv", argv)
    # Production gate requires all 1,968 test records, not this one-record fixture.
    with pytest.raises(SystemExit) as error:
        compare_runs.main()
    assert error.value.code == 2 and not report_path.exists()
    monkeypatch.setattr(compare_runs, "FULL_TEST_COUNT", 1)
    with pytest.raises(SystemExit) as done:
        compare_runs.main()
    assert done.value.code == 0
    report = json.loads(report_path.read_text())
    assert report["expected_reference"]["sha256"] == digest
    assert report["expected_reference"]["split"] == "test"
    argv[argv.index("--output") + 1] = str(tmp_path / "bad.json")
    argv[argv.index("--expected-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit) as error:
        compare_runs.main()
    assert error.value.code == 2 and not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("options", [
    {"expected_split": "train"},
    {"expected_split": "test"},
    {"expected_experiments": ("M2_SFT_CHOSEN", "M3_DPO")},
    {"expected_experiments": ("M3_DPO", "M3_DPO")},
    {"expected_annotations": {"wrong-key": annotation()}},
    {"expected_annotations": {"q1": annotation()}, "expected_validation_ids": {"q1"}},
])
def test_reference_api_rejects_ambiguous_protocol(options):
    with pytest.raises(ValueError):
        compare_predictions([sample()], [adapter_sample()], **options)


def test_pairs_by_id_not_order():
    left = [sample(), sample("q2")]
    right = deepcopy(left[::-1])
    right[0]["prediction_answer"] = "chó"
    report = compare_predictions(left, right)
    assert report["comparable"]
    assert not report["canonical_references_checked"]
    assert not report["run_identity_metadata_checked"]
    assert report["delta_right_minus_left"]["exact_match"] == -0.5
    assert report["explanation_rouge_l_f1"]["comparable"]


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("experiment_id", "M1_SFT"), ("adapter", "runs/m1/adapter"),
])
def test_rejects_mixed_run_or_split_even_when_both_files_match(field, value):
    rows = [sample(), sample("q2")]
    rows[1][field] = value
    report = compare_predictions(rows, deepcopy(rows))
    assert not report["comparable"]
    assert report["scores"] is None


def test_generic_comparison_allows_distinct_consistent_models_on_test():
    left = [{**sample(qid), "split": "test"} for qid in ("q1", "q2")]
    right = [{**adapter_sample(qid), "split": "test"} for qid in ("q1", "q2")]
    assert compare_predictions(left, right)["comparable"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_pixels", 802816),
        ("reference_answer", "chó"),
        ("image_sha256", "b" * 64),
        ("question", "Ở đâu?"),
        ("split", "test"),
        ("prompt_sha256", "changed"),
        ("load_in_4bit", False),
        ("compute_dtype", "torch.bfloat16"),
        ("image_path", "images/other.jpg"),
        ("seed", 20260919),
    ],
)
def test_rejects_mismatch(field, value):
    right = sample()
    right[field] = value
    report = compare_predictions([sample()], [right])
    assert not report["comparable"]
    assert report["scores"] is None


def test_rejects_legacy_missing_provenance():
    row = sample()
    del row["prompt_sha256"]
    assert not compare_predictions([row], [row])["comparable"]


def test_rejects_missing_or_invalid_image_digest():
    row = sample()
    del row["image_sha256"]
    assert not compare_predictions([row], [sample()])["comparable"]
    row["image_sha256"] = "not-a-sha256"
    assert not compare_predictions([row], [sample()])["comparable"]


def test_rejects_duplicates_and_partial_sets():
    assert not compare_predictions([sample(), sample()], [sample()])["comparable"]
    assert not compare_predictions([sample()], [sample("q2")])["comparable"]


def test_rejects_empty():
    assert not compare_predictions([], [])["comparable"]


def test_rejects_non_object_rows_without_crashing():
    report = compare_predictions([sample(), None], [sample()])
    assert not report["comparable"]
    assert report["scores"] is None
    assert any("prediction must be an object" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", ""),
        ("reference_answer", " "),
        ("reference_answer", "!!!"),
        ("prompt_sha256", "not-a-hash"),
        ("max_pixels", 0),
        ("min_pixels", 999999),
        ("max_new_tokens", True),
        ("generation_config", {}),
        ("load_in_4bit", "true"),
        ("seed", "42"),
    ],
)
def test_rejects_invalid_shared_metadata_and_references(field, value):
    left = sample()
    right = sample()
    left[field] = right[field] = value
    report = compare_predictions([left], [right])
    assert not report["comparable"]
    assert report["scores"] is None


def test_rejects_mixed_protocol_even_when_both_files_match():
    rows = [sample(), sample("q2")]
    rows[1]["max_pixels"] = 802816
    assert not compare_predictions(rows, deepcopy(rows))["comparable"]


def test_paired_explanation_score_is_separate_from_answer_score():
    left = [sample(), sample("q2")]
    right = deepcopy(left[::-1])
    right[0]["prediction_explanation"] = "Sai hoàn toàn."
    report = compare_predictions(left, right)
    assert report["comparable"]
    assert report["delta_right_minus_left"]["exact_match"] == 0
    assert report["explanation_rouge_l_f1"]["scores"] == {"left": 1.0, "right": 0.5}
    assert report["explanation_rouge_l_f1"]["delta_right_minus_left"] == -0.5


def test_missing_explanation_does_not_hide_answer_result_or_score_subset():
    left = [sample(), sample("q2")]
    right = deepcopy(left)
    del right[1]["prediction_explanation"]
    report = compare_predictions(left, right)
    assert report["comparable"]
    assert report["scores"]["left"]["exact_match"] == 1.0
    assert not report["explanation_rouge_l_f1"]["comparable"]
    assert report["explanation_rouge_l_f1"]["scores"] is None


def test_answer_only_files_remain_answer_comparable():
    left = sample()
    right = deepcopy(left)
    for row in (left, right):
        del row["prediction_explanation"]
        del row["reference_explanations"]
    report = compare_predictions([left], [right])
    assert report["comparable"]
    assert report["scores"]["left"]["exact_match"] == 1.0
    assert not report["explanation_rouge_l_f1"]["comparable"]
    assert report["explanation_rouge_l_f1"]["scores"] is None


def test_reference_explanations_must_match_for_explanation_comparison():
    right = sample()
    right["reference_explanations"] = ["Có chó."]
    report = compare_predictions([sample()], [right])
    assert report["comparable"]
    assert not report["explanation_rouge_l_f1"]["comparable"]


def test_full_validation_gate_requires_exact_ids_and_validation_split():
    left = [sample("q1"), sample("q2")]
    right = deepcopy(left[::-1])
    assert compare_predictions(left, right, expected_validation_ids={"q1", "q2"})["comparable"]

    subset = compare_predictions(left[:1], right[:1], expected_validation_ids={"q1", "q2"})
    assert not subset["comparable"]
    assert subset["scores"] is None
    assert any("missing 1 expected validation" in issue for issue in subset["issues"])

    extra = compare_predictions(left, right, expected_validation_ids={"q1"})
    assert not extra["comparable"]
    assert any("outside the validation manifest" in issue for issue in extra["issues"])

    left[0]["split"] = "test"
    right[1]["split"] = "test"
    wrong_split = compare_predictions(left, right, expected_validation_ids={"q1", "q2"})
    assert not wrong_split["comparable"]
    assert any("not marked validation" in issue for issue in wrong_split["issues"])


def test_official_manifest_rejects_same_file_and_wrong_shared_references():
    canonical = {"q1": annotation()}
    same_file = compare_predictions(
        [sample()], [sample()], expected_validation_annotations=canonical
    )
    assert not same_file["comparable"]
    assert any("expected experiment_id=M1_SFT" in issue for issue in same_file["issues"])

    left = sample()
    right = adapter_sample()
    left["reference_answer"] = right["reference_answer"] = "chó"
    report = compare_predictions(
        [left], [right], expected_validation_annotations=canonical
    )
    assert not report["comparable"]
    assert report["scores"] is None
    assert sum("reference_answer differs" in issue for issue in report["issues"]) == 2


def test_official_manifest_checks_all_references_and_role_metadata():
    canonical = {"q1": annotation()}
    report = compare_predictions(
        [sample()], [adapter_sample()], expected_validation_annotations=canonical
    )
    assert report["comparable"]
    assert report["canonical_references_checked"]
    assert report["run_identity_metadata_checked"]

    right = adapter_sample()
    right["adapter"] = None
    right["reference_explanations"] = ["Sai."]
    right["image_path"] = "images/wrong.jpg"
    report = compare_predictions([sample()], [right], expected_validation_annotations=canonical)
    assert not report["comparable"]
    assert any("M1 requires" in issue for issue in report["issues"])
    assert any("reference_explanations differs" in issue for issue in report["issues"])
    assert any("image_path differs" in issue for issue in report["issues"])


def test_expected_validation_annotations_are_unique_pinned_and_validation_only(tmp_path):
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(annotation(qid)) for qid in ("q1", "q2"))
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert read_expected_validation_ids(manifest, expected_count=2, expected_sha256=digest) == (
        {"q1", "q2"},
        digest,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_expected_validation_ids(manifest, expected_count=2, expected_sha256="0" * 64)

    manifest.write_text(
        json.dumps(annotation()) + "\n" + json.dumps(annotation()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate validation question_id"):
        read_expected_validation_ids(manifest, expected_count=2)
    bad_split = annotation()
    bad_split["split"] = "test"
    manifest.write_text(json.dumps(bad_split) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-validation split"):
        read_expected_validation_ids(manifest, expected_count=1)


def test_compare_runs_cli_full_validation_flag_rejects_shared_subset(tmp_path, monkeypatch):
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(annotation(f"q{index}")) for index in range(1407))
        + "\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(json.dumps(sample("q0")) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_runs.py",
            str(prediction),
            str(prediction),
            "--output",
            str(output),
            "--expected-validation-annotations",
            str(manifest),
            "--expected-validation-sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        compare_runs_main()
    assert exit_info.value.code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scores"] is None
    assert report["expected_validation"]["n"] == 1407
    assert any("missing 1406 expected validation" in issue for issue in report["issues"])

    full_predictions = tmp_path / "full_m0_predictions.jsonl"
    full_predictions.write_text(
        "\n".join(json.dumps(sample(f"q{index}")) for index in range(1407)) + "\n",
        encoding="utf-8",
    )
    full_adapter_predictions = tmp_path / "full_m1_predictions.jsonl"
    full_adapter_predictions.write_text(
        "\n".join(json.dumps(adapter_sample(f"q{index}")) for index in range(1407))
        + "\n",
        encoding="utf-8",
    )
    full_output = tmp_path / "full_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_runs.py",
            str(full_predictions),
            str(full_adapter_predictions),
            "--output",
            str(full_output),
            "--expected-validation-annotations",
            str(manifest),
            "--expected-validation-sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
        ],
    )
    with pytest.raises(SystemExit) as full_exit:
        compare_runs_main()
    assert full_exit.value.code == 0
    full_report = json.loads(full_output.read_text(encoding="utf-8"))
    assert full_report["comparable"]
    assert full_report["n"] == 1407
    assert full_report["canonical_references_checked"]
    assert full_report["run_identity_metadata_checked"]


def test_compare_runs_cli_requires_pinned_official_manifest(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(json.dumps(annotation()) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_runs.py", str(predictions), str(predictions), "--output", str(output),
            "--expected-validation-annotations", str(manifest),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        compare_runs_main()
    assert exit_info.value.code == 2
    assert not output.exists()
    assert "require --expected-validation-sha256" in capsys.readouterr().err


def test_compare_runs_cli_reports_output_write_failure(tmp_path, monkeypatch, capsys):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
    parent_file = tmp_path / "not_a_directory"
    parent_file.write_text("keep", encoding="utf-8")
    output = parent_file / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compare_runs.py", str(predictions), str(predictions), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as exit_info:
        compare_runs_main()
    assert exit_info.value.code == 2
    assert "Could not write comparison report" in capsys.readouterr().err
    assert parent_file.read_text(encoding="utf-8") == "keep"
