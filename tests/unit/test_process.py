"""Subprocess diagnostics remain visible when a command fails."""

import subprocess
import sys

import pytest

from vqa_explanation.utils.process import run_command


def test_stdout_and_stderr_are_forwarded(capsys):
    command = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    assert run_command(command) == 0
    output = capsys.readouterr().out
    assert "out" in output and "err" in output


def test_failure_keeps_diagnostics_and_exit_code(capsys):
    command = [sys.executable, "-c", "print('dependency conflict'); raise SystemExit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command(command)
    assert error.value.returncode == 7
    assert "dependency conflict" in capsys.readouterr().out


def test_child_does_not_inherit_parent_python_paths(monkeypatch, capsys):
    monkeypatch.setenv("PYTHONPATH", "outside_environment")
    monkeypatch.setenv("PYTHONHOME", "outside_environment")
    command = [sys.executable, "-c",
               "import os,sys; print(os.getenv('PYTHONPATH')); print(sys.flags.no_user_site)"]
    assert run_command(command) == 0
    assert capsys.readouterr().out.splitlines() == ["None", "1"]


def test_output_callback_receives_complete_lines():
    lines = []
    run_command([sys.executable, "-c", "print('first'); print('second')"], on_output=lines.append)
    assert lines == ["first\n", "second\n"]


def test_callback_failure_stops_child_and_propagates():
    def fail(line):
        raise ValueError("snapshot failed")

    with pytest.raises(ValueError, match="snapshot failed"):
        run_command([sys.executable, "-u", "-c",
                     "import time; print('ready'); time.sleep(30)"], on_output=fail)
