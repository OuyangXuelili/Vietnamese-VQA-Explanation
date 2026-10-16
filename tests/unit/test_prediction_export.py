import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import run_m0
from vqa_explanation.evaluation.comparison import compare_predictions
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner


def metadata_runner():
    runner = object.__new__(Qwen25VLRunner)
    runner.model = SimpleNamespace(
        generation_config=SimpleNamespace(to_dict=lambda: {"num_beams": 1}), dtype="float16"
    )
    runner.processor = SimpleNamespace(chat_template="tested template")
    runner.min_pixels = 200704
    runner.max_pixels = 200704
    runner.load_in_4bit = True
    runner.compute_dtype = "torch.float16"
    return runner


def test_protocol_records_effective_generation_and_template():
    runner = metadata_runner()
    a = runner.protocol_metadata(64)
    assert a["generation_config"] == {"num_beams": 1, "do_sample": False, "max_new_tokens": 64}
    assert a["max_new_tokens"] == 64
    assert a["compute_dtype"] == "torch.float16"
    runner.processor.chat_template = "different template"
    assert a["prompt_sha256"] != runner.protocol_metadata(64)["prompt_sha256"]


def test_export_and_gate_end_to_end_without_gpu(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4)).save(image)
    annotation = tmp_path / "data.jsonl"
    row = dict(
        question_id="q1",
        image_id="im1",
        image_path="image.jpg",
        split="validation",
        question="Con gì?",
        answer="mèo",
        explanations=["Có mèo."],
    )
    annotation.write_text(json.dumps(row), encoding="utf-8")
    runner = metadata_runner()
    runner.generate = lambda *a, **kw: "Đáp án: mèo\nGiải thích: Có mèo."
    captured = []

    def make_runner(*args, **kwargs):
        captured.append(kwargs)
        return runner

    monkeypatch.setattr(run_m0, "Qwen25VLRunner", make_runner)
    args = SimpleNamespace(
        annotations=annotation,
        image_root=tmp_path,
        output=tmp_path / "m0.jsonl",
        split="validation",
        limit=20,
        seed=42,
        model_id="model",
        revision="revision",
        max_new_tokens=64,
        min_pixels=200704,
        max_pixels=200704,
        load_in_4bit=True,
        adapter=None,
        experiment_id="M0_PRETRAINED",
    )
    monkeypatch.setattr(run_m0, "parse_args", lambda: args)
    run_m0.main()
    left = run_m0.read_jsonl(args.output)
    assert len(left[0]["image_sha256"]) == 64
    assert captured[0]["min_pixels"] == 200704
    args.output = tmp_path / "m1.jsonl"
    args.adapter = tmp_path / "fake-adapter"
    args.adapter.mkdir()
    (args.adapter / "adapter_config.json").write_text('{}', encoding='utf-8')
    (args.adapter / "adapter_model.safetensors").write_bytes(b'test weights')
    args.experiment_id = "M1_SFT"
    run_m0.main()
    right = run_m0.read_jsonl(args.output)
    assert compare_predictions(left, right)["comparable"]
    with pytest.raises(SystemExit, match="Output exists"):
        run_m0.main()


def test_read_jsonl_skips_whitespace_and_rejects_non_object_rows(tmp_path):
    annotation = tmp_path / "data.jsonl"
    annotation.write_text('  \n{"question_id":"q1"}\n\n', encoding="utf-8")
    assert run_m0.read_jsonl(annotation) == [{"question_id": "q1"}]
    annotation.write_text('{"question_id":"q1"}\nnull\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 must contain a JSON object"):
        run_m0.read_jsonl(annotation)


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", " "),
        ("answer", "!!!"),
        ("explanations", []),
        ("explanations", [None]),
        ("image_path", "../outside.jpg"),
    ],
)
def test_export_preflight_rejects_bad_data_before_loading_model(
    tmp_path, monkeypatch, field, value
):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4)).save(image)
    row = dict(
        question_id="q1",
        image_id="im1",
        image_path="image.jpg",
        split="validation",
        question="Con gì?",
        answer="mèo",
        explanations=["Có mèo."],
    )
    row[field] = value
    annotation = tmp_path / "data.jsonl"
    annotation.write_text(json.dumps(row), encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    args = SimpleNamespace(
        annotations=annotation,
        image_root=tmp_path,
        output=output,
        split="validation",
        limit=20,
        seed=42,
        model_id="model",
        revision="revision",
        max_new_tokens=64,
        min_pixels=200704,
        max_pixels=200704,
        load_in_4bit=True,
        adapter=None,
        experiment_id="M0_PRETRAINED",
    )
    monkeypatch.setattr(run_m0, "parse_args", lambda: args)

    def fail_if_model_loaded(*unused_args, **unused_kwargs):
        raise AssertionError("model should not load for invalid annotations")

    monkeypatch.setattr(run_m0, "Qwen25VLRunner", fail_if_model_loaded)
    with pytest.raises(SystemExit):
        run_m0.main()
    assert not output.exists()


def resume_fixture(tmp_path, monkeypatch):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.jpg")
    rows = [dict(question_id=f"q{i}", image_id="im1", image_path="image.jpg",
                 split="validation", question="Con gì?", answer="mèo",
                 explanations=["Có mèo."]) for i in range(3)]
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    args = SimpleNamespace(
        annotations=annotations, image_root=tmp_path, output=tmp_path / "predictions.jsonl",
        split="validation", limit=0, seed=42, model_id="model", revision="revision",
        max_new_tokens=64, min_pixels=200704, max_pixels=200704, load_in_4bit=True,
        adapter=None, experiment_id="M0_PRETRAINED", resume=False,
    )
    runner = metadata_runner()
    monkeypatch.setattr(run_m0, "parse_args", lambda: args)
    monkeypatch.setattr(run_m0, "Qwen25VLRunner", lambda *a, **kw: runner)
    return args, runner


@pytest.mark.parametrize(
    "experiment_id,adapter_required",
    [("M1_SFT", True), ("M2_SFT_CHOSEN", True), ("M3_DPO", True),
     ("M3_DOP", False), ("", False)],
)
def test_export_rejects_invalid_experiment_before_loading_model(
    tmp_path, monkeypatch, experiment_id, adapter_required
):
    args, _ = resume_fixture(tmp_path, monkeypatch)
    args.experiment_id = experiment_id

    def fail_if_loaded(*unused_args, **unused_kwargs):
        raise AssertionError("Invalid experiment must fail before loading the model")

    monkeypatch.setattr(run_m0, "Qwen25VLRunner", fail_if_loaded)
    expected = "requires --adapter" if adapter_required else "Unknown experiment"
    with pytest.raises(SystemExit, match=expected):
        run_m0.main()
    assert not args.output.exists()
    assert not args.output.with_suffix(".jsonl.manifest.json").exists()


@pytest.mark.parametrize("experiment_id", ["M1_SFT", "M2_SFT_CHOSEN", "M3_DPO"])
def test_export_accepts_trained_experiment_with_adapter(tmp_path, monkeypatch, experiment_id):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    args.experiment_id = experiment_id
    args.adapter = tmp_path / "adapter"
    args.adapter.mkdir()
    (args.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (args.adapter / "adapter_model.safetensors").write_bytes(b"test weights")
    runner.generate = lambda *a, **kw: "Đáp án: mèo\nGiải thích: Có mèo."
    run_m0.main()
    rows = run_m0.read_jsonl(args.output)
    assert len(rows) == 3
    assert all(row["experiment_id"] == experiment_id for row in rows)
    assert all(row["adapter"] == str(args.adapter.resolve()) for row in rows)


def test_resume_adds_only_missing_ids_and_complete_export_does_not_load_model(
    tmp_path, monkeypatch
):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    calls = []

    def interrupted_generate(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("runtime interrupted")
        return "Đáp án: mèo\nGiải thích: Có mèo."

    runner.generate = interrupted_generate
    with pytest.raises(RuntimeError, match="interrupted"):
        run_m0.main()
    prefix = args.output.read_bytes()
    assert len(run_m0.read_jsonl(args.output)) == 1
    args.resume = True
    calls.clear()

    def finish_generate(*a, **kw):
        calls.append(1)
        return "Đáp án: mèo\nGiải thích: Có mèo."

    runner.generate = finish_generate
    run_m0.main()
    assert len(calls) == 2
    assert args.output.read_bytes().startswith(prefix)
    assert len({row['question_id'] for row in run_m0.read_jsonl(args.output)}) == 3

    def fail_if_loaded(*a, **kw):
        raise AssertionError('Completed export should not load a GPU model')

    monkeypatch.setattr(run_m0, "Qwen25VLRunner", fail_if_loaded)
    complete = args.output.read_bytes()
    run_m0.main()
    assert args.output.read_bytes() == complete


@pytest.mark.parametrize('change', ['image', 'annotations', 'tokens', 'duplicate',
                                   'reference', 'raw', 'truncated', 'protocol', 'source'])
def test_resume_rejects_changed_inputs_or_saved_rows_before_loading_model(
    tmp_path, monkeypatch, change
):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    runner.generate = lambda *a, **kw: "Đáp án: mèo\nGiải thích: Có mèo."
    run_m0.main()
    args.resume = True
    if change == 'image':
        Image.new('RGB', (5, 5)).save(tmp_path / 'image.jpg')
    elif change == 'annotations':
        args.annotations.write_text(args.annotations.read_text() + '\n', encoding='utf-8')
    elif change == 'tokens':
        args.max_new_tokens += 1
    elif change == 'source':
        original = run_m0.export_inputs

        def altered(*a):
            inputs = original(*a)
            inputs['source_files']['scripts/run_m0.py'] = 'changed'
            return inputs

        monkeypatch.setattr(run_m0, 'export_inputs', altered)
    elif change == 'truncated':
        with args.output.open('a', encoding='utf-8') as handle:
            handle.write('{"question_id":')
    else:
        rows = run_m0.read_jsonl(args.output)
        if change == 'duplicate':
            rows.append(rows[0])
        elif change == 'reference':
            rows[0]['reference_answer'] = 'chó'
        elif change == 'raw':
            rows[0]['raw_output'] = 'Đáp án: chó\nGiải thích: Có chó.'
        elif change == 'protocol':
            rows[0]['compute_dtype'] = 'torch.bfloat16'
        args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    preserved = args.output.read_bytes()

    def fail_if_loaded(*a, **kw):
        raise AssertionError('Invalid resume should fail before loading the model')

    monkeypatch.setattr(run_m0, 'Qwen25VLRunner', fail_if_loaded)
    with pytest.raises(SystemExit):
        run_m0.main()
    assert args.output.read_bytes() == preserved


def test_resume_rejects_changed_adapter_weights(tmp_path, monkeypatch):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    args.adapter = tmp_path / 'adapter'
    args.adapter.mkdir()
    (args.adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (args.adapter / 'adapter_model.safetensors').write_bytes(b'original')
    args.experiment_id = 'M1_SFT'
    runner.generate = lambda *a, **kw: 'Đáp án: mèo\nGiải thích: Có mèo.'
    run_m0.main()
    args.resume = True
    (args.adapter / 'adapter_model.safetensors').write_bytes(b'updated')
    with pytest.raises(SystemExit, match='changed'):
        run_m0.main()


def test_export_lock_rejects_second_writer(tmp_path, monkeypatch):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    with run_m0.FileLock(str(args.output) + '.lock'):
        with pytest.raises(SystemExit, match='second writer'):
            run_m0.main()


def test_resume_rejects_loaded_protocol_drift(tmp_path, monkeypatch):
    args, runner = resume_fixture(tmp_path, monkeypatch)
    runner.generate = lambda *a, **kw: 'Đáp án: mèo\nGiải thích: Có mèo.'
    run_m0.main()
    rows = run_m0.read_jsonl(args.output)
    args.output.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    preserved = args.output.read_bytes()
    args.resume = True
    runner.compute_dtype = 'torch.bfloat16'
    with pytest.raises(SystemExit, match='protocol changed'):
        run_m0.main()
    assert args.output.read_bytes() == preserved
