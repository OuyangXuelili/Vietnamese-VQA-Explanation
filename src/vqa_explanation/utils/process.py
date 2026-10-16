"""Forward child-process output to notebooks and terminals."""

import os
import subprocess


def run_command(command, *, cwd=None, on_output=None):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if on_output is not None:
                    on_output(line)
        except BaseException:
            process.terminate()
            process.wait()
            raise
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return returncode
