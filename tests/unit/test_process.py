"""Subprocess diagnostics remain visible when a command fails."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from vqa_explanation.utils import process as process_module
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


@pytest.mark.parametrize("nested_helper", [False, True])
@pytest.mark.parametrize("exception", [ValueError, KeyboardInterrupt])
def test_interrupt_stops_grandchild_but_not_unrelated_process(nested_helper, exception):
    psutil = pytest.importorskip("psutil")
    leaf = "import os,time; print('leaf:' + str(os.getpid()), flush=True); time.sleep(30)"
    if nested_helper:
        parent = (
            "import sys; from vqa_explanation.utils.process import run_command; "
            f"run_command([sys.executable, '-u', '-c', {leaf!r}])"
        )
    else:
        parent = (
            "import sys,subprocess; "
            f"child = subprocess.Popen([sys.executable, '-u', '-c', {leaf!r}]); child.wait()"
        )
    descendants = []

    def interrupt(line):
        if line.startswith("leaf:"):
            descendants.append(psutil.Process(int(line.split(":")[1])))
            raise exception("stop requested")

    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as unrelated:
        try:
            with pytest.raises(exception, match="stop requested"):
                run_command([sys.executable, "-u", "-c", parent], on_output=interrupt)
            assert len(descendants) == 1
            _, alive = psutil.wait_procs(descendants, timeout=3)
            assert all(child.status() == psutil.STATUS_ZOMBIE for child in alive)
            assert unrelated.poll() is None
        finally:
            for child in descendants:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
                    child.wait(timeout=5)
            unrelated.kill()
            unrelated.wait(timeout=5)


def test_posix_tree_collects_nested_sessions_without_unrelated_groups(monkeypatch):
    result = SimpleNamespace(stdout="50 1 50\n51 50 50\n52 51 52\n53 52 52\n75 1 75\n")
    monkeypatch.setattr(process_module.subprocess, "run", lambda *a, **kw: result)
    assert process_module._posix_process_groups(50) == {50, 52}


@pytest.mark.parametrize("enumeration_fails", [False, True])
def test_posix_cleanup_always_kills_outer_group(monkeypatch, enumeration_fails):
    signals = []
    monkeypatch.setattr(process_module.signal, "SIGSTOP", 19, raising=False)
    monkeypatch.setattr(process_module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_module, "_signal_group", lambda *args: signals.append(args))

    def groups(pid):
        if enumeration_fails:
            raise OSError("ps unavailable")
        return {pid, 52}

    monkeypatch.setattr(process_module, "_posix_process_groups", groups)
    if enumeration_fails:
        with pytest.raises(OSError, match="unavailable"):
            process_module._stop_posix_tree(SimpleNamespace(pid=50))
    else:
        process_module._stop_posix_tree(SimpleNamespace(pid=50))
    assert signals == ([(50, 19), (50, 9)] if enumeration_fails else
                       [(50, 19), (52, 9), (50, 9)])
