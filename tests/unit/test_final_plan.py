import json
from pathlib import Path

import pytest

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation import final_plan as final
from vqa_explanation.evaluation.blind_review import freeze_sample
from vqa_explanation.evaluation.comparison import PROTOCOL_FIELDS
from vqa_explanation.inference.study_validation import ARMS
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rows = [{"question_id": f"{split}{i}", "image_id": f"{split}{i}", "split": split}
            for split, count in (("train", 29459), ("validation", 1407), ("test", 1968))
            for i in range(count)]
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, rows)
    write_jsonl(tmp_path / "test.jsonl", [r for r in rows if r["split"] == "test"])
    sample = tmp_path / "sample.json"
    freeze_sample(annotations, sample)
    practice = tmp_path / "practice"
    practice.mkdir()
    evidence, studies, verified = {}, {}, {}
    for seed in final.SEEDS:
        folder = tmp_path / str(seed)
        (folder / "validation").mkdir(parents=True)
        write_json(folder / "m1_selection.json", {"fixture": "selection"})
        studies[seed] = tmp_path / f"study_{seed}.yaml"
        studies[seed].write_text(f"seed: {seed}", "utf-8")
        evidence[seed] = folder
        paths, predictions, manifests = {}, {}, {}
        for arm, role in ARMS.items():
            paths[arm] = folder / "validation" / f"{arm}.jsonl"
            write_jsonl(paths[arm], [{"experiment_id": role}])
            write_json(paths[arm].with_suffix(".jsonl.manifest.json"), {"fixture": arm})
            protocol = dict.fromkeys(PROTOCOL_FIELDS, "fixture")
            protocol.update(seed=42, max_pixels=200704, min_pixels=200704,
                            max_new_tokens=128, load_in_4bit=True,
                            generation_config={"do_sample": False})
            predictions[arm] = [protocol]
            manifests[arm] = {"inputs": {"source_files": {"runner.py": "a" * 64},
                             "adapter": None if arm == "m0" else "m1" if arm == "m1"
                             else f"{seed}/{arm}",
                             "adapter_files": None if arm == "m0" else {"weights": arm}}}
        verified[seed] = (paths, predictions, {"training_seed": seed,
                                              "training_learning_rate": 1e-5,
                                              "manifests": manifests})
    write_json(practice / "manifest.json", {
        "schema": "rubric_calibration_v1", "status": "awaiting_ratings", "split": "validation",
        "images": 30, "outputs_per_rater": 120, "training_seed": 42, "files": {},
        "sources": {arm: {"sha256": sha256_file(path)}
                    for arm, path in verified[42][0].items()}})
    monkeypatch.setattr(final, "verified_validation",
                        lambda folder, *args: verified[int(Path(folder).name)])
    return evidence, studies, annotations, sample, practice, verified


def test_plan_has_eight_identities_but_cannot_claim_runtime_or_rubric_ready(inputs):
    result = final.build_final_plan(*inputs[:5])
    assert result["status"] == "pending_rubric_and_runtime"
    assert result["expected_predictions"] == 15744 and len(result["runs"]) == 8
    assert result["training_seeds"] == [42, 43, 44]
    assert result["inference_order_seed"] == 42
    assert result["human_review"]["runs"] == ["m0", "m1", "m2_seed42", "m3_seed42"]
    assert result["adapter_weights_verified_in_this_step"] is False
    assert result["test_predictions_generated"] is False
    assert len(result["blockers"]) == 2 and "commands" not in result


def test_missing_seed_is_not_silently_replaced_by_best_seed(inputs):
    missing = {k: v for k, v in inputs[0].items() if k != 43}
    with pytest.raises(ValueError, match="exactly the precommitted"):
        final.build_final_plan(missing, *inputs[1:5])


@pytest.mark.parametrize("fault", ["lr", "seed", "protocol", "source", "baseline"])
def test_sensitivity_or_incompatible_run_cannot_enter_main_test(inputs, fault):
    _, predictions, evidence = inputs[5][43]
    if fault == "lr":
        evidence["training_learning_rate"] = 5e-6
    elif fault == "seed":
        evidence["training_seed"] = 44
    elif fault == "protocol":
        predictions["m3"][0]["max_new_tokens"] = 256
    elif fault == "source":
        evidence["manifests"]["m3"]["inputs"]["source_files"] = {"runner.py": "b" * 64}
    else:
        evidence["manifests"]["m1"]["inputs"]["adapter_files"] = {"weights": "changed"}
    with pytest.raises(ValueError):
        final.build_final_plan(*inputs[:5])


@pytest.mark.parametrize("fault", ["split", "id", "count"])
def test_changed_frozen_test_sample_is_rejected(inputs, fault):
    path = inputs[3]
    sample = json.loads(path.read_text("utf-8"))
    if fault == "split":
        sample["split"] = "validation"
    elif fault == "id":
        sample["question_ids"][0] = "changed"
    else:
        sample["count"] = 30
    write_json(path, sample)
    with pytest.raises(ValueError, match="200-image test sample changed"):
        final.build_final_plan(*inputs[:5])


def test_test_file_must_match_canonical_records(inputs):
    path = inputs[2].parent / "test.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["image_id"] = "different"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Test annotations differ"):
        final.build_final_plan(*inputs[:5])


def test_image_leakage_blocks_preparation(inputs):
    path = inputs[2]
    rows = list(read_jsonl(path))
    rows[-1]["image_id"] = rows[0]["image_id"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Image leakage"):
        final.build_final_plan(*inputs[:5])


def test_calibration_must_use_main_outputs_and_immutable_files(inputs):
    path = inputs[4] / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["sources"]["m2"]["sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="main seed 42 outputs"):
        final.build_final_plan(*inputs[:5])


def test_plan_write_is_new_and_does_not_touch_predictions(inputs):
    result = final.build_final_plan(*inputs[:5])
    output = inputs[2].parent / "new_plan"
    hashes = {p: sha256_file(p) for seed in final.SEEDS for p in inputs[5][seed][0].values()}
    final.save_final_plan(result, output)
    assert json.loads((output / "plan.json").read_text("utf-8")) == result
    assert all(sha256_file(p) == digest for p, digest in hashes.items())
    with pytest.raises(FileExistsError):
        final.save_final_plan(result, output)
