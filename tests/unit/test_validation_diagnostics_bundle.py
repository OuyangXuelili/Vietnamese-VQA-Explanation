import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tests.unit.test_validation_diagnostics import inputs
from vqa_explanation.evaluation import validation_diagnostics_bundle as module


def tree(tmp_path, monkeypatch):
    predictions, annotations, semantic, _ = inputs()
    image_root = tmp_path / "coco"
    (image_root / "images").mkdir(parents=True)
    for i, ref in enumerate(annotations):
        image = image_root / ref["image_path"]
        Image.new("RGB", (20, 15), (i * 30, 40, 50)).save(image)
        digest = module.sha256_file(image)
        for seed in module.SEEDS:
            for rows in predictions[seed].values():
                rows[i]["image_sha256"] = digest
    canonical = tmp_path / "all.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in annotations)
    canonical.write_text(text, encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(text, encoding="utf-8")
    settings, by_seed = {}, {}
    for seed in module.SEEDS:
        folder = tmp_path / f"seed{seed}"
        (folder / "validation").mkdir(parents=True)
        (folder / "training").mkdir()
        (folder / "semantic").mkdir()
        (folder / "m1_selection.json").write_text("{}", encoding="utf-8")
        study = folder / "study.yaml"
        study.write_text("fixture: true", encoding="utf-8")
        paths = {}
        for arm in module.ARMS:
            p = folder / "validation" / f"{arm}.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in predictions[seed][arm]),
                         encoding="utf-8")
            p.with_suffix(".jsonl.manifest.json").write_text("{}", encoding="utf-8")
            paths[arm] = p
            metric = semantic[seed][arm]
            metric["predictions_sha256"] = module.sha256_file(p)
            protocol = metric["explanation_bertscore"]["protocol"]
            protocol.update(model_id=module.DEFAULT_BERTSCORE_MODEL,
                            model_revision=module.DEFAULT_BERTSCORE_REVISION,
                            num_layers=module.DEFAULT_BERTSCORE_LAYERS, idf=False,
                            rescale_with_baseline=False, use_fast_tokenizer=False,
                            reference_aggregation="best_match_across_all_references")
            metric["explanation_bertscore"]["protocol_sha256"] = hashlib.sha256(json.dumps(
                protocol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            (folder / "semantic" / f"{arm}_metrics.json").write_text(json.dumps(metric),
                                                                     encoding="utf-8")
        settings[seed] = {"evidence": str(folder), "study": str(study),
                          "semantic": str(folder / "semantic")}
        by_seed[folder] = paths, predictions[seed], {
            "training_seed": seed, "training_learning_rate": 1e-5}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"schema": "validation_diagnostics_v1", "seeds": settings}),
                      encoding="utf-8")
    monkeypatch.setattr(module, "verified_validation", lambda folder, *_: by_seed[Path(folder)])
    return config, {"annotations": canonical, "image_root": image_root}, predictions


def test_bundle_is_self_contained_and_manifest_covers_every_file(tmp_path, monkeypatch):
    config, environment, predictions = tree(tmp_path, monkeypatch)
    predictions[42]["m3"][0]["raw_output"] = "<script>not markup</script>"
    output = tmp_path / "output"
    result = module.build_diagnostics(config, environment, output, resamples=10)
    assert result["images_verified"] == 4
    assert (output / "index.html").is_file()
    text = (output / "data.js").read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "\\u003cscript>" in text
    files = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert all(module.sha256_file(output / name) == digest for name, digest in files.items())
    assert set(files) == {p.relative_to(output).as_posix() for p in output.rglob("*")
                          if p.is_file() and p.name != "manifest.json"}


def test_rejects_existing_output_without_overwriting_it(tmp_path, monkeypatch):
    config, environment, _ = tree(tmp_path, monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.build_diagnostics(config, environment, output, resamples=10)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize("fault", ["image", "escape", "lr", "source"])
def test_bad_inputs_fail_before_output_is_created(tmp_path, monkeypatch, fault):
    config, environment, predictions = tree(tmp_path, monkeypatch)
    if fault == "image":
        Image.new("RGB", (20, 15), "red").save(environment["image_root"] / "images/0.jpg")
    elif fault == "escape":
        for rows in predictions[42].values():
            rows[0]["image_path"] = "../../outside.jpg"
        annotations = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text(
            encoding="utf-8").splitlines()]
        annotations[0]["image_path"] = "../../outside.jpg"
        (tmp_path / "validation.jsonl").write_text("".join(json.dumps(r) + "\n"
                                                         for r in annotations), encoding="utf-8")
    elif fault == "lr":
        original = module.verified_validation
        def altered(*args):
            paths, rows, evidence = original(*args)
            return paths, rows, {**evidence, "training_learning_rate": 5e-6}
        monkeypatch.setattr(module, "verified_validation", altered)
    else:
        (tmp_path / "seed44/validation/m3.jsonl").unlink()
    output = tmp_path / "new_output"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_diagnostics(config, environment, output, resamples=10)
    assert not output.exists()
