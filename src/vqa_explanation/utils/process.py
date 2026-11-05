"""Forward child-process output to notebooks and terminals."""

import os
import signal
import subprocess
from pathlib import Path


def _posix_process_groups(pid):
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid=,pgid="], capture_output=True, text=True,
        check=True, timeout=5,
    )
    rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    owned = {pid}
    while True:
        children = {child for child, parent, _ in rows if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    return {group for child, _, group in rows if child in owned and group in owned}


def _signal_group(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass


def _stop_posix_tree(process):
    groups = {process.pid}
    try:
        # Freeze the outer session before enumerating nested sessions.
        _signal_group(process.pid, signal.SIGSTOP)
        groups.update(_posix_process_groups(process.pid))
    finally:
        for group in sorted(groups - {process.pid}):
            _signal_group(group, signal.SIGKILL)
        _signal_group(process.pid, signal.SIGKILL)


def _stop_process_tree(process):
    """Stop only this command's descendants, including nested command sessions."""
    if os.name == "nt":
        taskkill = Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"
        result = subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=10, check=False,
        )
        if result.returncode and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
            raise RuntimeError("Could not stop the complete command process tree")
    else:
        _stop_posix_tree(process)
    process.wait(timeout=10)


def run_command(command, *, cwd=None, on_output=None):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1,
                          start_new_session=os.name != "nt") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if on_output is not None:
                    on_output(line)
            returncode = process.wait()
        except BaseException:
            _stop_process_tree(process)
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return returncode
