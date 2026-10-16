import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import run_m0
from scripts.run_m0 import DEFAULT_MODEL, DEFAULT_REVISION
from tests.unit.test_prediction_export import metadata_runner
from vqa_explanation.inference import export as prediction_export
from vqa_explanation.preference.candidates import candidate_records, generate_candidates
from vqa_explanation.preference.selection import freeze_selection
from vqa_explanation.training.preference_data import adapter_fingerprint
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def candidate_run(tmp_path):
    rows = []
    for index, split in enumerate(('train', 'train', 'train', 'validation', 'test')):
        Image.new('RGB', (4, 4), color=(index, 0, 0)).save(tmp_path / f'{index}.jpg')
        rows.append(dict(question_id=f'q{index}', image_id=f'i{index}',
                         image_path=f'{index}.jpg', split=split, question='Con gì?',
                         answer='mèo', explanations=['Có mèo.']))
    annotations = tmp_path / 'all.jsonl'
    annotations.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    selection = tmp_path / 'pool'
    freeze_selection(annotations, tmp_path, selection, count=3, seed=42)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
    return SimpleNamespace(
        selection=selection, annotations=annotations, image_root=tmp_path,
        expected_sha256=sha256_file(annotations), count=2,
        output_dir=tmp_path / 'candidates', resume=False, adapter=adapter,
        model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, seed=42,
        min_pixels=200704, max_pixels=200704, max_new_tokens=128, load_in_4bit=True,
    )


def test_candidates_follow_frozen_prefix_and_keep_train_split(candidate_run):
    a = candidate_run
    rows, context = candidate_records(
        a.selection, a.annotations, a.image_root, a.expected_sha256, a.count
    )
    pool = run_m0.read_jsonl(a.selection / 'questions.jsonl')
    assert [r['question_id'] for r in rows] == [r['source_question_id'] for r in pool[:2]]
    assert {r['split'] for r in rows} == {'train'}
    assert context['purpose'] == 'preference_candidates_pending_review'


def test_candidates_resume_interruption_without_duplicates(candidate_run, monkeypatch):
    a = candidate_run
    runner = metadata_runner()
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return 'Đáp án: mèo\nGiải thích: Có mèo.'

    runner.generate = interrupted
    monkeypatch.setattr(prediction_export, 'Qwen25VLRunner', lambda *args, **kwargs: runner)
    with pytest.raises(RuntimeError, match='interrupted'):
        generate_candidates(a)
    predictions = a.output_dir / 'candidates.jsonl'
    prefix = predictions.read_bytes()
    assert len(run_m0.read_jsonl(predictions)) == 1
    a.resume = True
    runner.generate = lambda *args, **kwargs: 'Đáp án: mèo\nGiải thích: Có mèo.'
    report = generate_candidates(a)
    assert report['n'] == 2 and report['status'] == 'generated_pending_review'
    assert predictions.read_bytes().startswith(prefix)
    rows = run_m0.read_jsonl(predictions)
    assert len({r['question_id'] for r in rows}) == 2
    assert not any('chosen' in r or 'rejected' in r or 'accepted' in r for r in rows)

    def must_not_load(*args, **kwargs):
        raise AssertionError('Complete candidates must not load the model again')

    monkeypatch.setattr(prediction_export, 'Qwen25VLRunner', must_not_load)
    assert generate_candidates(a) == report
    a.resume = False
    with pytest.raises(FileExistsError):
        generate_candidates(a)


@pytest.mark.parametrize('mutation', ['annotations', 'image', 'selection'])
def test_changed_inputs_rejected_before_loading_model(candidate_run, monkeypatch, mutation):
    a = candidate_run
    if mutation == 'annotations':
        a.annotations.write_text('{}\n', encoding='utf-8')
    elif mutation == 'selection':
        (a.selection / 'questions.jsonl').write_text('{}\n', encoding='utf-8')
    else:
        for path in a.image_root.glob('*.jpg'):
            path.write_bytes(b'changed')

    def must_not_load(*args, **kwargs):
        raise AssertionError('Invalid inputs must not load a model')

    monkeypatch.setattr(prediction_export, 'Qwen25VLRunner', must_not_load)
    with pytest.raises(ValueError, match='hash'):
        generate_candidates(a)
    assert not a.output_dir.exists()


def test_context_is_recorded_in_export_manifest(candidate_run, monkeypatch):
    runner = metadata_runner()
    runner.generate = lambda *args, **kwargs: 'Đáp án: mèo\nGiải thích: Có mèo.'
    monkeypatch.setattr(prediction_export, 'Qwen25VLRunner', lambda *args, **kwargs: runner)
    a = candidate_run
    generate_candidates(a)
    manifest = json.loads((a.output_dir / 'candidates.jsonl.manifest.json').read_text())
    assert manifest['inputs']['context']['annotations_sha256'] == a.expected_sha256
    assert manifest['inputs']['context']['count'] == 2
    assert manifest['inputs']['split'] == 'train'


def test_changed_m1_selection_rejected_before_output_or_model(candidate_run):
    a = candidate_run
    a.m1_selection = a.image_root / 'm1_selection.json'
    a.m1_selection.write_text('{}', encoding='utf-8')
    a.m1_selection_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='Frozen M1 selection hash changed'):
        generate_candidates(a)
    assert not a.output_dir.exists()


def test_selected_parent_is_validated_and_recorded(candidate_run, monkeypatch):
    from vqa_explanation.preference import candidates

    a = candidate_run
    a.m1_selection = a.image_root / 'm1_selection.json'
    a.m1_selection.write_text('{}', encoding='utf-8')
    a.m1_selection_sha256 = sha256_file(a.m1_selection)
    parent = adapter_fingerprint(a.adapter)
    calls = []

    def verify(path, adapter, model, annotations_hash, canonical):
        calls.append(path)
        assert adapter == parent
        assert model == {'id': a.model_id, 'revision': a.revision}
        assert annotations_hash == a.expected_sha256 and len(canonical) == 5

    monkeypatch.setattr(candidates, 'validate_selection', verify)
    runner = metadata_runner()
    runner.generate = lambda *args, **kwargs: 'Đáp án: mèo\nGiải thích: Có mèo.'
    monkeypatch.setattr(prediction_export, 'Qwen25VLRunner', lambda *args, **kwargs: runner)
    generate_candidates(a)
    manifest = json.loads((a.output_dir / 'candidates.jsonl.manifest.json').read_text())
    assert calls == [a.m1_selection]
    context = manifest['inputs']['context']
    assert context['m1_adapter'] == parent
    assert context['m1_selection_sha256'] == a.m1_selection_sha256


def test_wrong_selected_adapter_is_rejected_before_generation(candidate_run, monkeypatch):
    from vqa_explanation.preference import candidates

    a = candidate_run
    a.m1_selection = a.image_root / 'm1_selection.json'
    a.m1_selection.write_text('{}', encoding='utf-8')
    a.m1_selection_sha256 = sha256_file(a.m1_selection)

    def reject(*args):
        raise ValueError('M1 needs an explicit validation-only selection and matching adapter')

    monkeypatch.setattr(candidates, 'validate_selection', reject)
    with pytest.raises(ValueError, match='matching adapter'):
        generate_candidates(a)
    assert not a.output_dir.exists()


def test_held_out_question_rejected_even_with_matching_selection_hash(candidate_run):
    a = candidate_run
    rows = run_m0.read_jsonl(a.selection / 'questions.jsonl')
    held_out = run_m0.read_jsonl(a.annotations)[-1]
    rows[0].update(source_question_id=held_out['question_id'], split='test')
    questions = a.selection / 'questions.jsonl'
    questions.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    path = a.selection / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files']['questions.jsonl'] = sha256_file(questions)
    manifest['calibration_question_ids'] = [r['source_question_id'] for r in rows]
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='train questions only'):
        candidate_records(a.selection, a.annotations, a.image_root, a.expected_sha256, a.count)
