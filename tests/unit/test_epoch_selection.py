import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_m1_epoch as epoch


def completed_execution():
    return {"stage": "sft-full", "status": "passed", "returncode": 0,
            "budget": {"protocol_id": "vqa_nle_m1_epoch1_v1",
                       "completion_mode": "full_epoch", "stop_after_step": 3683}}


@pytest.mark.parametrize("change", [
    {"status": "running"}, {"status": "paused"}, {"returncode": 1},
    {"returncode": False}, {"stage": "audit"}, {"budget": {"stop_after_step": 1200}},
])
def test_selection_rejects_active_or_incomplete_runs(change):
    with pytest.raises(ValueError, match="completed M1 epoch"):
        epoch.require_completed_run(completed_execution() | change)


def test_active_run_stops_before_source_import_or_model_load(tmp_path):
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    (run / "execution.json").write_text(json.dumps(completed_execution() | {"status": "running"}))
    with pytest.raises(ValueError, match="active run"):
        epoch.prepare(SimpleNamespace(source=source, run=run))


def completion():
    return {"selected_eval_loss": 1.02, "selected_step": 3600,
            "selected_checkpoint": "/recorded/training/checkpoint-3600"}


@pytest.mark.parametrize("loss,step", [(1.00, 3683), (1.02, 3600), (1.05, 3600)])
def test_checkpoint_choice_uses_validation_and_keeps_best_on_ties(loss, step):
    result = epoch.select_checkpoint(completion(), loss, Path("/run"))
    assert result["selected_step"] == step
    assert result["selected_validation_loss"] == min(loss, 1.02)
    assert result["selected_adapter"].endswith("checkpoint-3683" if step == 3683 else "adapter")


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), True, "1.0", None])
def test_selection_rejects_invalid_loss(loss):
    with pytest.raises(ValueError, match="finite validation losses"):
        epoch.select_checkpoint(completion(), loss, Path("/run"))
    with pytest.raises(ValueError, match="finite validation losses"):
        epoch.select_checkpoint(completion() | {"selected_eval_loss": loss}, 1.0, Path("/run"))


def test_reloaded_adapter_must_match_exactly():
    torch = pytest.importorskip("torch")
    weights = {"lora_A.weight": torch.ones(2, 3)}
    epoch.verify_weights(weights, {"lora_A.weight": torch.ones(2, 3)})
    for invalid in ({}, {"wrong_key": torch.ones(2, 3)},
                    {"lora_A.weight": torch.zeros(2, 3)},
                    {"lora_A.weight": torch.full((2, 3), float("nan"))}):
        with pytest.raises(ValueError, match="weights do not match"):
            epoch.verify_weights(weights, invalid)


def prepared_job(tmp_path):
    inputs = tmp_path / "finished-run"
    inputs.mkdir()
    marker = inputs / "trainer_state.json"
    marker.write_text('{"global_step": 3683}')
    (inputs / "training/checkpoint-3683").mkdir(parents=True)
    (inputs / "training/checkpoint-3683/adapter_model.safetensors").write_bytes(b"mock weights")
    prepared = {"run": inputs, "source": tmp_path / "source", "completion": completion(),
                "validation": ["v1", "v2"],
                "checks": SimpleNamespace(verify_source=lambda *args: None),
                "evidence": {"file_sha256": {str(marker): epoch.sha256(marker)},
                             "source_files": {}}}
    args = SimpleNamespace(source=tmp_path / "source", run=inputs,
                           output_dir=tmp_path / "selection", check_only=False)
    return args, prepared, marker


def test_post_epoch_job_records_selection_without_changing_training(tmp_path, monkeypatch):
    args, prepared, marker = prepared_job(tmp_path)
    previous = marker.read_bytes()
    monkeypatch.setattr(epoch, "prepare", lambda _: prepared)
    monkeypatch.setattr(epoch, "evaluate_final", lambda *args: {"eval_loss": 1.0})
    epoch.run(args)
    result = json.loads((args.output_dir / "selection.json").read_text("utf-8"))
    assert result["status"] == "passed" and result["selected_step"] == 3683
    assert result["selected_adapter_sha256"] == epoch.sha256(
        args.run / "training/checkpoint-3683/adapter_model.safetensors")
    assert result["test_predictions_generated"] == 0 and result["validation_count"] == 2
    assert marker.read_bytes() == previous
    with pytest.raises(FileExistsError):
        epoch.run(args)


def test_changed_inputs_fail_without_publishing_a_choice(tmp_path, monkeypatch):
    args, prepared, marker = prepared_job(tmp_path)
    monkeypatch.setattr(epoch, "prepare", lambda _: prepared)

    def changed_model(*args):
        marker.write_text("changed")
        return {"eval_loss": 1.0}

    monkeypatch.setattr(epoch, "evaluate_final", changed_model)
    with pytest.raises(ValueError, match="inputs changed"):
        epoch.run(args)
    result = json.loads((args.output_dir / "selection.json").read_text("utf-8"))
    assert result["status"] == "failed" and "selected_step" not in result


def test_check_only_writes_nothing_and_loads_no_model(tmp_path, monkeypatch):
    args, prepared, _ = prepared_job(tmp_path)
    args.check_only = True
    monkeypatch.setattr(epoch, "prepare", lambda _: prepared)
    monkeypatch.setattr(epoch, "evaluate_final", lambda *_: pytest.fail("Model must not load"))
    epoch.run(args)
    assert not args.output_dir.exists()


def test_output_cannot_write_inside_source_or_training(tmp_path, monkeypatch):
    args, _, _ = prepared_job(tmp_path)
    monkeypatch.setattr(
        epoch, "prepare", lambda _: pytest.fail("Must fail before inspecting inputs"),
    )
    for parent in (args.run, args.source):
        args.output_dir = parent / "selection"
        with pytest.raises(ValueError, match="outside the finished run"):
            epoch.run(args)


def test_preloaded_project_source_cannot_mix_with_pinned_source(tmp_path):
    import vqa_explanation  # noqa: F401

    with pytest.raises(ValueError, match="fresh process"):
        epoch.pinned_training(tmp_path)


@pytest.fixture
def selection_inputs(tmp_path, monkeypatch):
    import yaml
    from PIL import Image

    from scripts.resume_pinned_m1 import verify_source
    from vqa_explanation.data.io import read_jsonl
    from vqa_explanation.training.sft_full import plan_data
    from vqa_explanation.utils.environment import load_environment, project_root

    source = tmp_path / "source"
    config_path = source / "configs/training/sft.yaml"
    config_path.parent.mkdir(parents=True)
    cfg = yaml.safe_load((project_root() / "configs/training/sft.yaml").read_text("utf-8"))
    cfg.update(expected_train_count=2, expected_validation_count=2, expected_test_count=1,
               eval_sample_size=1)
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model_path = source / "configs/model/qwen25_vl_3b.yaml"
    model_path.parent.mkdir()
    model_path.write_text("model: {}", encoding="utf-8")
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    records = [{"question_id": f"q{i}", "image_id": f"i{i}",
                "image_path": f"images/i{i}.jpg", "question": "Có gì?", "answer": "mèo",
                "explanations": ["Có mèo trong ảnh."], "split": split}
               for i, split in enumerate(["train", "train", "validation", "validation", "test"])]
    for row in records:
        Image.new("RGB", (4, 4)).save(data / row["image_path"])
    annotations = data / "all.jsonl"
    annotations.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    checksums = data / "checksums.csv"
    checksums.write_text("image_name,size_bytes,sha256\n" + "".join(
        f"{image.name},{image.stat().st_size},{epoch.sha256(image)}\n"
        for image in (data / "images").glob("*.jpg")
    ), encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({"annotations": str(annotations),
                           "image_root": str(data), "image_checksums": str(checksums),
                           "run_root": str(tmp_path / "runs")}), encoding="utf-8")
    run = tmp_path / "finished"
    training = run / "training"
    final = training / "checkpoint-3683"
    final.mkdir(parents=True)
    (training / "adapter").mkdir()
    for path in (final, training / "adapter"):
        (path / "adapter_model.safetensors").write_bytes(b"mock adapter for prepare-only test")
    for name in ("training_result.json", "trainer_state.json", "run_metadata.json"):
        (training / name).write_text("{}")
    _, validation, plan = plan_data(records, data, cfg)
    plan.update(annotations_sha256=epoch.sha256(annotations),
                training_config_sha256=epoch.sha256(config_path),
                model_config_sha256=epoch.sha256(model_path), image_integrity={})
    (training / "data_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    files = {"environment": environment, "annotations": annotations,
             "training_config": config_path, "model_config": model_path,
             "image_checksums": checksums}
    execution = completed_execution()
    execution["provenance"] = {
        "source_files": {"configs/training/sft.yaml": epoch.sha256(config_path)},
        "input_sha256": {name: epoch.sha256(path) for name, path in files.items()},
    }
    (run / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    fake_checks = SimpleNamespace(
        verify_source=verify_source,
        verify_epoch_completion=lambda *_: {
            **completion(), "selected_adapter_sha256": epoch.sha256(
                training / "adapter/adapter_model.safetensors"),
        },
        verify_checkpoint_payloads=lambda *_: {
            "compute_dtype": "torch.bfloat16", "file_sha256": {
                "adapter_model.safetensors": epoch.sha256(final / "adapter_model.safetensors")},
        },
    )
    verified_images = []

    def image_verifier(selected, image_root, manifest):
        verified_images.extend(row["question_id"] for row in selected)
        return {"verified_images": len(selected)}

    fake_full = SimpleNamespace(
        RUNTIME_PACKAGES=("pytest",), validate_resume_runtime=lambda *_: None,
        read_jsonl=read_jsonl, plan_data=plan_data,
    )
    monkeypatch.setattr(epoch, "load_module", lambda *_: fake_checks)
    monkeypatch.setattr(epoch, "pinned_training", lambda *_: (
        None, fake_full, image_verifier, load_environment,
    ))
    return (SimpleNamespace(source=source, run=run, environment=environment),
            validation, verified_images)


def test_prepare_restores_exact_subset_without_image_testing_heldout_test(selection_inputs):
    args, validation, verified_images = selection_inputs
    prepared = epoch.prepare(args)
    assert prepared["validation"] == validation
    assert verified_images == [row["question_id"] for row in validation]
    assert prepared["evidence"]["validation_question_ids"] == verified_images
    assert all(row["split"] == "validation" for row in prepared["validation"])


def test_prepare_refuses_changed_data_plan(selection_inputs):
    args, _, _ = selection_inputs
    path = args.run / "training/data_plan.json"
    plan = json.loads(path.read_text("utf-8"))
    plan["checkpoint_validation_question_ids"] = ["different-question"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="fixed validation subset"):
        epoch.prepare(args)


def test_prepare_refuses_changed_source(selection_inputs):
    args, _, _ = selection_inputs
    path = args.source / "configs/training/sft.yaml"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Pinned source changed"):
        epoch.prepare(args)


def test_prepare_refuses_changed_annotations(selection_inputs):
    args, _, _ = selection_inputs
    annotations = args.environment.parent / "data/all.jsonl"
    annotations.write_text("changed")
    with pytest.raises(ValueError, match="Inputs differ"):
        epoch.prepare(args)
