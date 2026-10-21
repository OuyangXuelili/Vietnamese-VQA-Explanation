import json
import sys

import pytest
from filelock import FileLock, Timeout

from scripts import run_study_validation as cli


def cli_arguments(tmp_path):
    return ["run_study_validation.py", "--environment", str(tmp_path / "env.yaml"),
            "--run-group", str(tmp_path / "run"), "--m1-adapter", str(tmp_path / "m1"),
            "--output-dir", str(tmp_path / "validation"),
            "--baseline-source", str(tmp_path / "baseline"),
            "--baseline-sha256", json.dumps({"m0.jsonl": "a" * 64}), "--inspect"]


def test_cli_passes_baseline_pins_and_inspection_to_library(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", cli_arguments(tmp_path))
    monkeypatch.setattr(cli, "run_validation", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or {"status": "inputs_verified"})
    cli.main()
    assert calls[0][1]["baseline_source"] == tmp_path / "baseline"
    assert calls[0][1]["baseline_sha256"] == {"m0.jsonl": "a" * 64}
    assert calls[0][1]["inspect_only"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "inputs_verified"


def test_cli_does_not_read_baseline_while_another_export_holds_its_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", cli_arguments(tmp_path))
    monkeypatch.setattr(cli, "run_validation", lambda *args, **kwargs:
                        pytest.fail("Source lock must be acquired before validation"))
    with FileLock(str(tmp_path / "baseline") + ".lock", timeout=0):
        with pytest.raises(Timeout):
            cli.main()
