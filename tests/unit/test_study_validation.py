import hashlib
import json
import shutil
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


@pytest.mark.parametrize("inspect_only", [True, False])
def test_verified_fixed_baselines_skip_both_model_exports(inputs, monkeypatch, inspect_only):
    events, calls = [], []
    plan = {"receipt": {"schema": "validation_baseline_reuse_v1"}}
    monkeypatch.setattr(study, "prepare_baseline_reuse", lambda *a: plan)
    monkeypatch.setattr(study, "write_baseline_reuse", lambda p: events.append("copied") or
                        p["receipt"])
    result = study.run_validation(
        *inputs, baseline_source=inputs[0].parent / "baseline", baseline_sha256={},
        inspect_only=inspect_only, runner=lambda cmd, **kw: calls.append(cmd),
    )
    commands = result["commands"] if inspect_only else calls[:2]
    assert [command[command.index("--experiment-id") + 1] for command in commands] == [
        "M2_SFT_CHOSEN", "M3_DPO"]
    assert events == ([] if inspect_only else ["copied"])
    assert len(calls) == (0 if inspect_only else 6)
    assert inputs[-1].exists() is not inspect_only


def test_invalid_baseline_stops_before_any_export_or_receipt(inputs, monkeypatch):
    def invalid(*args):
        raise ValueError("Baseline checksum mismatch")
    monkeypatch.setattr(study, "prepare_baseline_reuse", invalid)
    with pytest.raises(ValueError, match="checksum"):
        study.run_validation(*inputs, baseline_source="baseline", baseline_sha256={},
                             runner=lambda *a, **kw: pytest.fail("must not export"))
    assert not inputs[-1].exists()


@pytest.mark.parametrize("options", [{"baseline_source": "baseline"},
                                     {"baseline_sha256": {}},
                                     {"baseline_source": "baseline", "baseline_sha256": {},
                                      "relocate_from": "old_study"}])
def test_baseline_options_do_not_allow_unpinned_or_old_study_reuse(inputs, options):
    with pytest.raises(ValueError, match="together|relocation"):
        study.run_validation(*inputs, **options)
    assert not inputs[-1].exists()


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


def test_selects_completed_resume_without_changing_interrupted_attempt(main_inputs):
    inputs, config = main_inputs
    initial = inputs[1] / "m3_train"
    resumed = inputs[1] / "m3_train_resume150"
    shutil.copytree(initial, resumed)
    path = initial / "training_result.json"
    report = json.loads(path.read_text())
    report.update(status="paused", global_step=150, epoch=1.2)
    path.write_text(json.dumps(report))
    original_bytes = path.read_bytes()

    with pytest.raises(ValueError, match="not complete"):
        study.run_validation(*inputs, study_config=config, inspect_only=True)
    result = study.run_validation(
        *inputs, study_config=config, inspect_only=True,
        stage_names={"m3": "m3_train_resume150"},
    )
    command = result["commands"][1]
    assert command[command.index("--adapter") + 1] == str(
        resumed / "training/checkpoint-375")
    assert result["training_evidence"]["m3"]["stage"] == resumed.name
    assert path.read_bytes() == original_bytes
    assert not inputs[-1].exists()


@pytest.mark.parametrize("name", ["", "../m3_train", "/m3_train", "m2_train",
                                  "m3_train/nested", "m3_train_resume/../x", "latest", 43])
def test_rejects_invalid_training_attempt_names(main_inputs, name):
    inputs, config = main_inputs
    with pytest.raises(ValueError, match="stage name"):
        study.run_validation(*inputs, study_config=config, inspect_only=True,
                             stage_names={"m3": name})
    assert not inputs[-1].exists()


def test_unknown_or_missing_attempt_is_not_replaced_by_another_run(main_inputs):
    inputs, config = main_inputs
    with pytest.raises(ValueError, match="only m2 or m3"):
        study.run_validation(*inputs, study_config=config, inspect_only=True,
                             stage_names={"m4": "m4_train"})
    with pytest.raises(FileNotFoundError, match="Missing selected training stage"):
        study.run_validation(*inputs, study_config=config, inspect_only=True,
                             stage_names={"m3": "m3_train_resume150"})


def test_resumed_attempt_still_checks_adapter_bytes(main_inputs):
    inputs, config = main_inputs
    resumed = inputs[1] / "m3_train_resume150"
    shutil.copytree(inputs[1] / "m3_train", resumed)
    (resumed / "training/checkpoint-375/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Adapter changed"):
        study.run_validation(*inputs, study_config=config, inspect_only=True,
                             stage_names={"m3": resumed.name})


def test_linked_training_attempt_is_rejected(main_inputs, monkeypatch):
    inputs, config = main_inputs
    stage = inputs[1] / "m3_train"
    monkeypatch.setattr(study.Path, "is_symlink", lambda path: path == stage)
    with pytest.raises(ValueError, match="within the run group"):
        study.run_validation(*inputs, study_config=config, inspect_only=True)


def test_validation_receipt_prevents_switching_training_attempts(inputs):
    resumed = inputs[1] / "m3_train_resume10"
    shutil.copytree(inputs[1] / "m3_train", resumed)
    calls = []
    runner = lambda command, **kwargs: calls.append(command)  # noqa: E731
    study.run_validation(*inputs, runner=runner)
    receipt = inputs[-1] / "training_evidence.json"
    saved = receipt.read_bytes()
    with pytest.raises(ValueError, match="training selection changed"):
        study.run_validation(*inputs, runner=runner, stage_names={"m3": resumed.name})
    assert len(calls) == 8
    assert receipt.read_bytes() == saved


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
