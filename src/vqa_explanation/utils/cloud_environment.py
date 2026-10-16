"""Install preference-stage requirements into a separate Python environment."""

import os
import re
import sys
import venv
from pathlib import Path

from vqa_explanation.utils.process import run_command


def environment_python(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_environment(project_dir, environment_dir, *, runner=run_command):
    project_dir = Path(project_dir).resolve()
    environment_dir = Path(environment_dir)
    if environment_dir.is_symlink():
        raise ValueError("Environment directory cannot be a symlink")
    environment_dir = environment_dir.resolve()
    if project_dir.is_relative_to(environment_dir):
        raise ValueError("Environment cannot replace the source directory or its ancestors")
    requirements = project_dir / "requirements-cloud-preference.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    python = environment_python(environment_dir)
    if environment_dir.exists():
        if not (environment_dir / "pyvenv.cfg").is_file() or not python.is_file():
            raise ValueError("Existing environment directory is not a virtual environment")
        config = (environment_dir / "pyvenv.cfg").read_text(encoding="utf-8")
        if not re.search(r"(?mi)^include-system-site-packages\s*=\s*false\s*$", config):
            raise ValueError("Training environment must exclude system site packages")
    else:
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment_dir)
    pip = [sys.executable, "-m", "pip", "--python", str(python)]
    runner(pip + ["install", "--disable-pip-version-check", "--report",
                  str(environment_dir / "install_report.json"), "-r", str(requirements)],
           cwd=project_dir)
    runner(pip + ["check"], cwd=project_dir)
    return python
