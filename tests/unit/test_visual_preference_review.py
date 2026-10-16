import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import review_preference_images as review
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.preference.expansion import validate_proposal
from vqa_explanation.utils.run_provenance import sha256_file


def fixture(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (32, 32), 'red').save(images / 'one.jpg')
    annotations = tmp_path / 'annotations.jsonl'
    sample = {'question_id': 'q1', 'image_id': 'i1', 'image_path': 'one.jpg',
              'question': 'Có áo không?', 'answer': 'có', 'explanations': ['Có áo đỏ.'],
              'split': 'train', 'image_sha256': sha256_file(images / 'one.jpg')}
    origin = {'model_id': 'model', 'model_revision': 'c' * 40,
              'checkpoint_sha256': 'b' * 64, 'seed': 42}
    sample.update(original_candidate_origin=origin, candidate_raw='Đáp án: có\nGiải thích: Áo đỏ.')
    write_jsonl(annotations, [sample])
    batch = tmp_path / 'batch'
    batch.mkdir()
    write_jsonl(batch / 'items.jsonl', [sample])
    Image.new('RGB', (32, 32), 'red').save(batch / 'images.png')
    write_json(batch / 'manifest.json', {
        'schema': 'visual_sheet_v2', 'original_candidate_origin': origin,
        'candidates_sha256': 'd' * 64, 'candidate_manifest_sha256': 'e' * 64,
        'annotations_sha256': sha256_file(annotations),
        'items_sha256': sha256_file(batch / 'items.jsonl'),
        'sheet_sha256': sha256_file(batch / 'images.png'),
    })
    decision = {'question_id': 'q1', 'status': 'accepted',
                'chosen': 'Người đứng ở giữa mặc áo đỏ.',
                'rejected': 'Người đứng ở giữa mặc áo xanh.',
                'rejected_span': 'áo xanh', 'evidence': 'Áo đỏ ở giữa ảnh.',
                'error_type': 'attribute', 'answer_correct': True, 'chosen_supported': True,
                'rejected_unsupported': True, 'both_linguistically_plausible': True,
                'language_only_preference': 'undetermined', 'contrast_relevant': True,
                'relevance_reason': 'The contrast describes clothing on the queried person.',
                'text_only_reason': 'A person can wear either color; the words do not decide.'}
    payload = {'schema': 'direct_visual_decisions_v2',
               'decisions': [decision]}
    decisions = batch / 'decisions.json'
    write_json(decisions, payload)
    return SimpleNamespace(annotations=annotations, image_root=images, batch=batch,
                           decisions=decisions, output_dir=tmp_path / 'output'), payload


def test_import_keeps_provenance_and_refuses_duplicate(tmp_path):
    args, _ = fixture(tmp_path)
    review.ingest(args)
    pair = list(read_jsonl(args.output_dir / 'reviewed_pairs.jsonl'))[0]
    assert pair['reviewer_type'] == 'ai' and pair['human_verified'] is False
    assert pair['provenance']['chosen']['source'].startswith('sha256:')
    assert pair['provenance']['original_candidate_origin']['checkpoint_sha256'] == 'b' * 64
    original = sha256_file(args.output_dir / 'decisions.jsonl')
    with pytest.raises(ValueError, match='already imported'):
        review.ingest(args)
    assert sha256_file(args.output_dir / 'decisions.jsonl') == original


@pytest.mark.parametrize('failure', ['assessment', 'sheet', 'image', 'question', 'duplicate',
                                   'lineage', 'relevance'])
def test_invalid_review_cannot_append_journal(tmp_path, failure):
    args, payload = fixture(tmp_path)
    if failure == 'assessment':
        payload['decisions'][0]['chosen_supported'] = False
    elif failure == 'sheet':
        Image.new('RGB', (32, 32), 'blue').save(args.batch / 'images.png')
    elif failure == 'image':
        Image.new('RGB', (32, 32), 'blue').save(args.image_root / 'one.jpg')
    elif failure == 'question':
        payload['decisions'][0]['chosen'] = 'Có áo không?'
    elif failure == 'lineage':
        items = list(read_jsonl(args.batch / 'items.jsonl'))
        items[0]['original_candidate_origin']['checkpoint_sha256'] = 'f' * 64
        write_jsonl(args.batch / 'items.jsonl', items)
        manifest = json.loads((args.batch / 'manifest.json').read_text('utf-8'))
        manifest['items_sha256'] = sha256_file(args.batch / 'items.jsonl')
        write_json(args.batch / 'manifest.json', manifest)
    elif failure == 'relevance':
        payload['decisions'][0]['contrast_relevant'] = False
    else:
        payload['decisions'].append(copy.deepcopy(payload['decisions'][0]))
    write_json(args.decisions, payload)
    with pytest.raises(ValueError):
        review.ingest(args)
    assert not (args.output_dir / 'decisions.jsonl').exists()


def test_shared_assessment_cannot_confirm_a_batch(tmp_path):
    args, payload = fixture(tmp_path)
    payload['assessment'] = {'answer_correct': True}
    write_json(args.decisions, payload)
    with pytest.raises(ValueError, match='per-pair assessments'):
        review.load_decisions(args.decisions)
    del payload['assessment']
    payload['decisions'][0].pop('relevance_reason')
    write_json(args.decisions, payload)
    with pytest.raises(ValueError, match='own complete assessment'):
        review.load_decisions(args.decisions)


@pytest.mark.parametrize('status', ['excluded', 'uncertain'])
def test_batch_without_accepted_pairs_still_records_its_decisions(tmp_path, status):
    args, payload = fixture(tmp_path)
    payload['decisions'] = [{'question_id': 'q1', 'status': status,
                             'reason': 'The image does not settle the answer.'}]
    write_json(args.decisions, payload)
    review.ingest(args)
    saved = list(read_jsonl(args.output_dir / 'decisions.jsonl'))
    assert saved[0]['status'] == status and saved[0]['pair'] is None
    assert list(read_jsonl(args.output_dir / 'reviewed_pairs.jsonl')) == []
    assert json.loads((args.output_dir / 'summary.json').read_text('utf-8'))['accepted'] == 0


@pytest.mark.parametrize('reason', [' ', None, True])
def test_exclusion_needs_a_nonempty_text_reason(tmp_path, reason):
    args, payload = fixture(tmp_path)
    payload['decisions'] = [{'question_id': 'q1', 'status': 'excluded', 'reason': reason}]
    write_json(args.decisions, payload)
    with pytest.raises(ValueError, match='need a reason'):
        review.ingest(args)
    assert not (args.output_dir / 'decisions.jsonl').exists()


def test_exact_saved_m1_text_keeps_model_origin(tmp_path):
    args, payload = fixture(tmp_path)
    items = list(read_jsonl(args.batch / 'items.jsonl'))
    items[0]['candidate_raw'] = 'Đáp án: có\nGiải thích: ' + payload['decisions'][0]['rejected']
    write_jsonl(args.batch / 'items.jsonl', items)
    manifest = json.loads((args.batch / 'manifest.json').read_text('utf-8'))
    manifest['items_sha256'] = sha256_file(args.batch / 'items.jsonl')
    write_json(args.batch / 'manifest.json', manifest)
    review.ingest(args)
    pair = list(read_jsonl(args.output_dir / 'reviewed_pairs.jsonl'))[0]
    assert pair['rejected'] == items[0]['candidate_raw']
    assert pair['provenance']['rejected']['kind'] == 'model'
    assert pair['provenance']['rejected']['seed'] == 42
    assert pair['provenance']['chosen']['kind'] == 'synthetic'


def test_withdrawal_preserves_original_record_and_invalidates_export(tmp_path):
    args, _ = fixture(tmp_path)
    review.ingest(args)
    original = sha256_file(args.output_dir / 'decisions.jsonl')
    args.question_id, args.reason = 'q1', 'Uncertain image-relative direction.'
    review.withdraw(args)
    assert list(read_jsonl(args.output_dir / 'reviewed_pairs.jsonl')) == []
    assert sha256_file(args.output_dir / 'decisions.jsonl') == original
    with pytest.raises(ValueError, match='one active pair'):
        review.withdraw(args)
    original_records = list(read_jsonl(args.output_dir / 'decisions.jsonl'))
    original_records[0]['pair']['chosen'] += ' Changed.'
    with pytest.raises(ValueError, match='stale withdrawal'):
        review.active_records(args.output_dir, original_records)


@pytest.mark.parametrize('mark', ['?', '？'])
def test_question_copy_cannot_be_used_as_explanation(mark):
    with pytest.raises(ValueError, match='statements'):
        validate_proposal({'usable': True, 'chosen': 'Bé có mặc áo mưa không' + mark,
                           'rejected': 'Bé có mặc áo khoác không' + mark,
                           'rejected_span': 'áo khoác', 'evidence': 'Áo mưa trên bé.',
                           'error_type': 'attribute'})
