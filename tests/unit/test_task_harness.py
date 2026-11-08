import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from vqa_explanation.utils import task_harness as harness
from vqa_explanation.utils.environment import project_root


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "scripts").mkdir()
    (root / "src/module.py").write_text("value = 1\n", encoding="utf-8")
    (root / "tests/test_sample.py").write_text("def test_example(): pass\n", encoding="utf-8")
    (root / "docs/report.md").write_text("Existing report\n", encoding="utf-8")
    (root / "TIEN_DO.md").write_text("Existing progress\n", encoding="utf-8")
    return root


@pytest.fixture
def real_check(monkeypatch):
    """Use real child exit codes/logging without recursively running this test suite."""
    def configure(code="print('real child check')"):
        monkeypatch.setattr(harness, "check_commands", lambda *_a, **_k: [
            [sys.executable, "-B", "-X", "utf8", "-c", code]])
    configure()
    return configure


def new_task(root, kind="code", criteria=None):
    return harness.start(root, "A bounded task", kind, criteria or ["Behavior verified"])


def read_record(task):
    return json.loads((task / "task.json").read_text(encoding="utf-8"))


def accepted(root, task, evidence=None, criterion=1):
    evidence = evidence or root / "docs/report.md"
    return harness.review(root, task, criterion, [evidence], "Read and checked the saved evidence")


def test_start_keeps_preexisting_work_and_requires_observable_criteria(repository):
    before = harness.worktree_files(repository)
    task = new_task(repository)
    assert read_record(task)["baseline"] == before
    assert (repository / "src/module.py").read_text() == "value = 1\n"
    assert harness.inspect(repository, task)["latest_check"] == "not_run"
    with pytest.raises(ValueError, match="criterion"):
        harness.start(repository, "No criteria", "code", [])


def test_default_plans_delegate_to_existing_cpu_gate(repository):
    full = harness.check_commands(repository, "full", [], False)
    assert full[-1][-1] == str(repository / "scripts/check_project.py")
    assert "--full-data" in harness.check_commands(repository, "data", [], False)[-1]
    assert "--source-only" in harness.check_commands(repository, "full", [], True)[-1]
    assert not any("train" in Path(part).name for command in full for part in command)


def test_focused_checks_use_explicit_test_files_and_reject_unsafe_targets(repository):
    commands = harness.check_commands(repository, "focused", ["tests/test_sample.py::test_example"],
                                      False)
    assert "tests/test_sample.py::test_example" in commands[-1]
    assert "--fail-on-skip" in commands[-1]
    for target in ("--disable-tests", "../outside.py", "src/module.py"):
        with pytest.raises(ValueError):
            harness.check_commands(repository, "focused", [target], False)


@pytest.mark.parametrize("profile,tests,source_only", [
    ("focused", [], False), ("full", ["tests/test_sample.py"], False),
    ("data", [], True), ("focused", ["tests/test_sample.py"], True),
])
def test_ambiguous_check_scope_is_rejected(repository, profile, tests, source_only):
    with pytest.raises(ValueError):
        harness.check_commands(repository, profile, tests, source_only)


def test_real_child_output_exit_and_criterion_allow_completion(repository, real_check):
    task = new_task(repository)
    checked = harness.check(repository, task)
    assert checked["status"] == "passed" and checked["returncode"] == 0
    log = Path(checked["receipt"]).with_name("console.log")
    assert "real child check" in log.read_text(encoding="utf-8")
    accepted(repository, task, Path(checked["receipt"]))
    output = harness.finish(repository, task, "completed", "The task is verified")
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["reviews"][0]["independent_human_review_verified"] is False
    with pytest.raises(ValueError, match="closed"):
        harness.check(repository, task)


def test_pending_criterion_cannot_complete_even_after_success(repository, real_check):
    task = new_task(repository, criteria=["First behavior", "Second behavior"])
    harness.check(repository, task)
    accepted(repository, task)
    with pytest.raises(ValueError, match="criterion 2"):
        harness.finish(repository, task, "completed", "Only half checked")
    assert read_record(task)["status"] == "in_progress"


def test_latest_failure_cannot_reuse_an_earlier_pass(repository, real_check):
    task = new_task(repository)
    first = harness.check(repository, task)
    real_check("print('failure evidence'); raise SystemExit(7)")
    failed = harness.check(repository, task)
    assert failed["status"] == "failed" and failed["returncode"] == 7
    assert Path(first["receipt"]).is_file()
    accepted(repository, task)
    with pytest.raises(ValueError, match="Latest check"):
        harness.finish(repository, task, "completed", "Must not pass")


def test_retry_preserves_failure_and_can_finish_after_success(repository, real_check):
    task = new_task(repository)
    real_check("raise SystemExit(2)")
    failed = harness.check(repository, task)
    real_check()
    passed = harness.check(repository, task)
    accepted(repository, task)
    harness.finish(repository, task, "completed", "Retried and verified")
    assert Path(failed["receipt"]) != Path(passed["receipt"])
    assert json.loads(Path(failed["receipt"]).read_text())["status"] == "failed"


def test_source_edit_after_checks_invalidates_completion(repository, real_check):
    task = new_task(repository)
    harness.check(repository, task)
    (repository / "src/module.py").write_text("value = 2\n", encoding="utf-8")
    accepted(repository, task)
    with pytest.raises(ValueError, match="stale"):
        harness.finish(repository, task, "completed", "Old checks")


def test_environment_change_invalidates_completion(repository, real_check, monkeypatch):
    task = new_task(repository)
    harness.check(repository, task)
    original = harness.importlib.metadata.version
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda name:
                        "changed-version" if name == "pytest" else original(name))
    accepted(repository, task)
    with pytest.raises(ValueError, match="stale"):
        harness.finish(repository, task, "completed", "Environment changed")


def test_input_change_during_real_check_is_not_reported_as_pass(repository, real_check):
    task = new_task(repository)
    real_check("from pathlib import Path; Path('src/module.py').write_text('value = 9\\n')")
    result = harness.check(repository, task)
    assert result["status"] == "stale" and result["returncode"] != 0
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["before"] != receipt["after"]


def test_source_edit_after_review_requires_new_review_even_after_retest(repository, real_check):
    task = new_task(repository)
    accepted(repository, task)
    (repository / "src/module.py").write_text("value = 3\n", encoding="utf-8")
    harness.check(repository, task)
    with pytest.raises(ValueError, match="criterion 1"):
        harness.finish(repository, task, "completed", "Review is old")


def test_reviewed_file_change_is_detected(repository):
    task = new_task(repository, "report")
    accepted(repository, task)
    (repository / "docs/report.md").write_text("Changed after review\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        harness.finish(repository, task, "completed", "Old document review")


def test_document_task_can_finish_from_artifact_review_without_cpu_tests(repository):
    task = new_task(repository, "docs")
    accepted(repository, task)
    handoff = harness.finish(repository, task, "completed", "Read the final document")
    assert json.loads(handoff.read_text())["checks"] == []


def test_code_changes_cannot_be_hidden_as_document_task(repository):
    task = new_task(repository, "docs")
    (repository / "src/module.py").write_text("value = 4\n", encoding="utf-8")
    accepted(repository, task)
    with pytest.raises(ValueError, match="executed check"):
        harness.finish(repository, task, "completed", "Code also changed")


def test_deleting_requirements_cannot_be_hidden_as_document_task(repository):
    dependency_file = repository / "requirements-demo.txt"
    dependency_file.write_text("-e .[demo]\n", encoding="utf-8")
    task = new_task(repository, "docs")
    dependency_file.unlink()
    accepted(repository, task)
    with pytest.raises(ValueError, match="executed check"):
        harness.finish(repository, task, "completed", "Dependencies changed")


def test_data_task_requires_full_data_scope(repository, real_check):
    task = new_task(repository, "data")
    harness.check(repository, task, "full")
    accepted(repository, task)
    with pytest.raises(ValueError, match="full-data"):
        harness.finish(repository, task, "completed", "Only source checked")
    harness.check(repository, task, "data")
    accepted(repository, task)
    harness.finish(repository, task, "completed", "Data scope recorded")


def test_journal_update_does_not_make_code_checks_stale(repository, real_check):
    task = new_task(repository)
    harness.check(repository, task)
    (repository / "TIEN_DO.md").write_text("Actual outcome and next step\n", encoding="utf-8")
    accepted(repository, task)
    output = harness.finish(repository, task, "completed", "Verified with journal update")
    assert "TIEN_DO.md" in json.loads(output.read_text())["changed_paths"]


@pytest.mark.parametrize("status", ["blocked", "needs_input", "in_progress"])
def test_unfinished_handoff_keeps_next_step_and_pending_work(repository, status):
    task = new_task(repository)
    with pytest.raises(ValueError, match="Next step"):
        harness.finish(repository, task, status, "Waiting")
    output = harness.finish(repository, task, status, "Not finished", "Obtain actual human labels")
    receipt = json.loads(output.read_text())
    assert receipt["next_step"] == "Obtain actual human labels"
    assert receipt["reviews"] == []
    assert harness.inspect(repository)["tasks"][0]["status"] == status


def test_evidence_outside_repo_and_task_path_traversal_are_rejected(repository, tmp_path):
    task = new_task(repository)
    outside = tmp_path / "external.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        accepted(repository, task, outside)
    with pytest.raises(ValueError):
        harness.inspect(repository, Path("../external"))
    with pytest.raises(ValueError):
        harness.inspect(repository, repository / "docs")


def test_wrong_checkout_receipts_are_rejected(repository):
    task = new_task(repository)
    data = read_record(task)
    data["repository"] = str(repository.parent / "another_checkout")
    (task / "task.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong repository"):
        harness.inspect(repository, task)


def test_concurrent_check_is_rejected_without_overwriting_task(repository, real_check):
    task = new_task(repository)
    with FileLock(str(task / "task.lock"), timeout=0):
        with pytest.raises(Timeout):
            harness.check(repository, task)
    assert read_record(task)["checks"] == []


def test_interrupt_is_recorded_and_remains_unfinished(repository, real_check, monkeypatch):
    task = new_task(repository)
    def interrupt(*_a, **_k):
        raise KeyboardInterrupt
    monkeypatch.setattr(harness, "run_command", interrupt)
    checked = harness.check(repository, task)
    assert checked["status"] == "interrupted" and checked["returncode"] == 130
    assert read_record(task)["status"] == "in_progress"


def test_missing_program_is_recorded_as_failure(repository, monkeypatch):
    task = new_task(repository)
    monkeypatch.setattr(harness, "check_commands", lambda *_a, **_k: [
        [str(repository / "missing_program.exe")]])
    checked = harness.check(repository, task)
    assert checked["status"] == "failed" and checked["returncode"] != 0
    assert "FileNotFoundError" in Path(checked["receipt"]).read_text()


def test_changed_log_invalidates_check_receipt(repository, real_check):
    task = new_task(repository)
    checked = harness.check(repository, task)
    Path(checked["receipt"]).with_name("console.log").write_text("Replaced log", encoding="utf-8")
    accepted(repository, task)
    with pytest.raises(ValueError, match="log"):
        harness.finish(repository, task, "completed", "Cannot trust replaced log")


def test_relabeling_commands_cannot_create_passed_profile(repository, real_check):
    task = new_task(repository)
    checked = harness.check(repository, task)
    receipt_path = Path(checked["receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["commands"][0]["argv"] = ["not-the-executed-command"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    accepted(repository, task)
    with pytest.raises(ValueError, match="commands"):
        harness.finish(repository, task, "completed", "Wrong command")


def test_cli_refuses_editable_install_from_another_checkout(repository):
    source = project_root() / "scripts/task_harness.py"
    shutil.copy2(source, repository / "scripts/task_harness.py")
    result = subprocess.run([sys.executable, "-B", str(repository / "scripts/task_harness.py"),
                             "status"], cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "another checkout" in result.stderr
    assert not (repository / "artifacts/tasks").exists()


@pytest.mark.parametrize("name,value", [
    ("PYTEST_ADDOPTS", "-k subset"), ("PYTEST_PLUGINS", "unrecorded_plugin"),
    ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"), ("PYTHONOPTIMIZE", "1"),
])
def test_ambient_options_cannot_silently_narrow_checks(repository, monkeypatch, name, value):
    task = new_task(repository)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        harness.check(repository, task)
    assert read_record(task)["checks"] == []


def test_import_probe_cannot_be_disabled_by_python_optimization(repository):
    probe = harness.check_commands(repository, "full", [], False)[0]
    environment = {**os.environ, "PYTHONOPTIMIZE": "1"}
    result = subprocess.run(probe, cwd=repository, env=environment, capture_output=True,
                            text=True, check=False)
    assert result.returncode != 0 and "another checkout" in result.stderr


