import shutil
import subprocess
from pathlib import Path


def test_validation_explorer_frontend():
    node = shutil.which("node")
    assert node is not None, "Node is required for frontend verification"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, "--test", "tests/frontend/validation_explorer.test.js"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
