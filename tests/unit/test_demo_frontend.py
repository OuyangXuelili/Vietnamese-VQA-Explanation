"""Run browser-state regressions with Node; real GPU/browser QA is separate."""

import shutil
import subprocess
from pathlib import Path


def test_demo_frontend_state_transitions():
    node = shutil.which("node")
    assert node is not None, "Node is required by the local verification environment"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, "--test", "tests/frontend/demo.test.js"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
