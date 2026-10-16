import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vqa_explanation.inference import study_validation as study


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"question_id": str(index), "split": "validation"} for index in range(1407)]
    annotations = data / "validation.jsonl"
    annotations.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(study, "VALIDATION_SHA256", study.sha256_file(annotations))
    monkeypatch.setattr(study, "validate_selected_records", lambda *args: None)
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({"annotations": str(data / "all.jsonl"),
                                         "image_root": str(data),
                                         "run_root": str(tmp_path / "runs")}), encoding="utf-8")
    m1 = tmp_path / "m1"
    names = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"m1"}
    m1.mkdir()
    for name, value in names.items():
        (m1 / name).write_bytes(value)
    parent = {"files": {name: digest(value) for name, value in names.items()}}
    group = tmp_path / "training"
    for arm in ("m2", "m3"):
        stage = group / f"{arm}_train"
        adapter = stage / "training/checkpoint-24"
        adapter.mkdir(parents=True)
        for name, value in names.items():
            (adapter / name).write_bytes(value)
        plan = {"pairs": 186, "m1_adapter": parent, "preferences_manifest_sha256": "p",
                "model": {"id": study.MODEL_ID, "revision": study.MODEL_REVISION}}
        plan["sha256"] = digest(json.dumps(plan, sort_keys=True).encode())
        report = {"status": "passed", "global_step": 24, "epoch": 1,
                  "policy_updated": True, "saved_policy_verified": True,
                  "reference_unchanged": arm == "m3", "plan_sha256": plan["sha256"],
                  "output_adapter": parent}
        (stage / "data_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (stage / "training_result.json").write_text(json.dumps(report), encoding="utf-8")
    return environment, group, m1, tmp_path / "evaluation"


def test_inspection_loads_no_model_and_writes_no_predictions(inputs):
    result = study.run_validation(*inputs, inspect_only=True)
    assert result["status"] == "inputs_verified"
    assert result["questions"] == 1407
    assert [command[command.index("--experiment-id") + 1] for command in result["commands"]] == [
        "M2_SFT_CHOSEN", "M3_DPO", "M0_PRETRAINED", "M1_SFT"]
    assert not inputs[-1].exists()
    assert all(command[command.index("--limit") + 1] == "0" for command in result["commands"])


def test_export_is_resumable_but_new_runs_do_not_use_resume(inputs):
    result = study.run_validation(*inputs, inspect_only=True)
    assert all("--resume" not in command for command in result["commands"])
    inputs[-1].mkdir()
    (inputs[-1] / "m2.jsonl.manifest.json").write_text("{}", encoding="utf-8")
    result = study.run_validation(*inputs, inspect_only=True)
    assert "--resume" in result["commands"][0]


@pytest.mark.parametrize("target", ["m1", "m2", "m3"])
def test_changed_adapter_blocks_every_export(inputs, target):
    adapter = inputs[2] if target == "m1" else inputs[1] / f"{target}_train/training/checkpoint-24"
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Adapter changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_unfinished_training_blocks_export(inputs):
    path = inputs[1] / "m3_train/training_result.json"
    report = json.loads(path.read_text())
    report["global_step"] = 5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not complete"):
        study.run_validation(*inputs, inspect_only=True)


def test_changed_plan_blocks_export(inputs):
    path = inputs[1] / "m2_train/data_plan.json"
    plan = json.loads(path.read_text())
    plan["pairs"] = 187
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_changed_validation_blocks_export(inputs):
    (inputs[0].parent / "data/validation.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="annotations changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_full_pipeline_calls_exports_then_four_paired_comparisons(inputs):
    calls = []
    result = study.run_validation(*inputs, runner=lambda cmd, **kw: calls.append(cmd))
    assert result["status"] == "completed" and len(calls) == 8
    assert all(Path(command[2]).name == "run_m0.py" for command in calls[:4])
    assert all("--bootstrap-resamples" in command for command in calls[4:])


@pytest.fixture
def main_inputs(inputs):
    root = Path(__file__).resolve().parents[2]
    study_path = inputs[0].parent / "study.yaml"
    declaration = yaml.safe_load(
        (root / "configs/experiments/preference_study_v2.yaml").read_text())
    for arm, filename in (("m2", "sft_chosen_full.yaml"), ("m3", "dpo_full.yaml")):
        config = inputs[0].parent / filename
        config.write_bytes((root / "configs/training" / filename).read_bytes())
        declaration["training"][f"{arm}_config"] = str(config)
    study_path.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    settings = study.main_study_settings(study_path)
    canonical = inputs[0].parent / "data/all.jsonl"
    canonical.write_text("canonical", encoding="utf-8")
    for arm in ("m2", "m3"):
        stage = inputs[1] / f"{arm}_train"
        (stage / "training/checkpoint-24").rename(stage / "training/checkpoint-375")
        plan_path, report_path = stage / "data_plan.json", stage / "training_result.json"
        plan = json.loads(plan_path.read_text())
        plan.update({
            "pairs": 1000, "config": settings[arm]["config"],
            "comparison_protocol": settings[arm]["protocol"], "expected_updates": 375,
            "annotations_sha256": study.sha256_file(canonical),
            "preferences_files": {"accepted.jsonl": "pairs", "chosen_only.jsonl": "chosen"},
            "m1_selection_sha256": "selection", "source_scope": "trainer_graph",
            "source_files": {"trainer.py": "source"},
            "visual_quality": {"sha256": settings[arm]["quality_sha256"], "ready": True,
                               "confirmed": 1000, "unique_images": 1000,
                               "required_review_version": 2},
        })
        plan.pop("sha256")
        plan["sha256"] = digest(json.dumps(plan, sort_keys=True).encode())
        report = json.loads(report_path.read_text())
        report.update({"global_step": 375, "epoch": 3.0, "plan_sha256": plan["sha256"],
                       "training_loss": 1.0})
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        (stage / "training/trainer_state.json").write_text(json.dumps({
            "global_step": 375, "max_steps": 375, "epoch": 3.0,
        }), encoding="utf-8")
    return inputs, study_path


def test_main_study_exports_final_checkpoint_without_loading_model(main_inputs):
    inputs, config = main_inputs
    result = study.run_validation(*inputs, study_config=config, inspect_only=True)
    assert result["status"] == "inputs_verified"
    commands = result["commands"]
    assert all("checkpoint-375" in command[command.index("--adapter") + 1]
               for command in commands[:2])
    assert not inputs[-1].exists()


def test_main_study_requires_explicit_declaration(main_inputs):
    with pytest.raises(ValueError, match="not complete"):
        study.run_validation(*main_inputs[0], inspect_only=True)


@pytest.mark.parametrize("field", ["global_step", "max_steps", "epoch"])
def test_main_study_rejects_partial_trainer_state(main_inputs, field):
    inputs, config = main_inputs
    state_path = inputs[1] / "m3_train/training/trainer_state.json"
    state = json.loads(state_path.read_text())
    state[field] = 1
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="trainer state is incomplete"):
        study.run_validation(*inputs, study_config=config, inspect_only=True)


@pytest.mark.parametrize("field", ["visual_quality", "comparison_protocol", "config"])
def test_main_study_rejects_different_locked_plan(main_inputs, field):
    inputs, config = main_inputs
    stage = inputs[1] / "m2_train"
    plan_path, report_path = stage / "data_plan.json", stage / "training_result.json"
    plan = json.loads(plan_path.read_text())
    plan[field] = {}
    plan.pop("sha256")
    plan["sha256"] = digest(json.dumps(plan, sort_keys=True).encode())
    report = json.loads(report_path.read_text())
    report["plan_sha256"] = plan["sha256"]
    plan_path.write_text(json.dumps(plan))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="locked study"):
        study.run_validation(*inputs, study_config=config, inspect_only=True)


def test_main_study_rejects_unlocked_data(main_inputs):
    inputs, config = main_inputs
    declaration = yaml.safe_load(config.read_text())
    declaration["preferences"]["data_locked"] = False
    config.write_text(yaml.safe_dump(declaration))
    with pytest.raises(ValueError, match="locked v2 study"):
        study.run_validation(*inputs, study_config=config, inspect_only=True)


@pytest.mark.parametrize("changed", [None, "left", "right", "annotations", "report"])
def test_comparison_cache_requires_unchanged_inputs_and_report(tmp_path, changed):
    files = {name: tmp_path / f"{name}.json" for name in ("left", "right", "annotations")}
    for path in files.values():
        path.write_text("{}")
    report = tmp_path / "comparison.json"
    assert not study.comparison_cache(report, *files.values())
    report.write_text('{"comparable": true}')
    assert study.comparison_cache(report, *files.values(), record=True)
    if changed:
        target = report if changed == "report" else files[changed]
        target.write_text('{"comparable": true, "changed": true}')
        with pytest.raises(ValueError, match="changed"):
            study.comparison_cache(report, *files.values())
    else:
        assert study.comparison_cache(report, *files.values())


def test_unreceipted_comparison_is_not_reused(tmp_path):
    data, report = tmp_path / "data", tmp_path / "comparison.json"
    data.write_text("{}")
    report.write_text('{"comparable": true}')
    with pytest.raises(ValueError, match="changed"):
        study.comparison_cache(report, data, data, data)


@pytest.mark.parametrize("inspect_only", [True, False])
def test_recovery_prepares_every_arm_before_writing(tmp_path, monkeypatch, inspect_only):
    source, destination = tmp_path / "original", tmp_path / "recovered"
    source.mkdir()
    monkeypatch.setattr(study, "prepare_validation", lambda *a, **kw: (
        {"image_root": tmp_path}, tmp_path / "annotations",
        {arm: None for arm in study.ARMS}, {},
    ))
    events = []

    def prepare(path, args):
        assert not destination.exists()
        events.append(("prepare", path.stem))
        return {"receipt": {"source": str(path), "copied_predictions": 500}}

    def write(plan):
        events.append(("write", plan["receipt"]["source"]))
        destination.mkdir(exist_ok=True)
        return plan["receipt"]

    monkeypatch.setattr(study, "prepare_snapshot_recovery", prepare)
    monkeypatch.setattr(study, "write_relocation", write)
    result = study.recover_validation(
        None, None, None, source, destination, study_config="locked", inspect_only=inspect_only,
    )
    assert [event[0] for event in events[:4]] == ["prepare"] * 4
    assert len(events) == (4 if inspect_only else 8)
    assert destination.exists() is not inspect_only
    assert result["status"] == ("recovery_verified" if inspect_only else "recovered")
    if not inspect_only:
        assert (destination / "snapshot_recovery.json").is_file()
