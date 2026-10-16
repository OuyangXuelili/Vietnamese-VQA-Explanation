from vqa_explanation.utils.run_provenance import preference_training_fingerprints


def test_ui_changes_preserve_trainer_identity_but_loss_dependencies_change_it(tmp_path):
    files = {
        "src/vqa_explanation/training/preference.py":
            "from vqa_explanation.training.loss import loss\n",
        "src/vqa_explanation/training/loss.py": "def loss(): return 1\n",
        "scripts/train_preference.py": "pass\n",
        "requirements-cloud-preference.txt": "torch==2.11.0\n",
        "app/static/demo.css": "body {color: black}\n",
        "src/vqa_explanation/preference/editor.html": "editor\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    original = preference_training_fingerprints(tmp_path)
    (tmp_path / "app/static/demo.css").write_text("new style", encoding="utf-8")
    (tmp_path / "src/vqa_explanation/preference/editor.html").write_text("new editor")
    assert preference_training_fingerprints(tmp_path) == original
    dependency = "src/vqa_explanation/training/loss.py"
    (tmp_path / dependency).write_text("def loss(): return 2\n", encoding="utf-8")
    changed = preference_training_fingerprints(tmp_path)
    assert changed[dependency] != original[dependency]
    assert set(changed) == set(original)


def test_relative_import_is_part_of_training_identity(tmp_path):
    for name, content in {
        "src/vqa_explanation/training/preference.py": "from .loss import loss\n",
        "src/vqa_explanation/training/loss.py": "def loss(): return 1\n",
        "scripts/train_preference.py": "pass\n",
        "requirements-cloud-preference.txt": "torch\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert "src/vqa_explanation/training/loss.py" in preference_training_fingerprints(tmp_path)
