import hashlib
import json
import zipfile

import pytest

from scripts.prepare_preference_review import render_cases, review_notes, verify_bundle


def test_render_escapes_text_and_keeps_review_pending():
    inputs = [dict(question_id='q1', image_path='images/one.jpg',
                   question='<script>bad</script>', answer='cat', explanations=['<b>cat</b>'])]
    predictions = {'q1': dict(prediction_answer='cat', prediction_explanation='<img onerror=x>')}
    page = render_cases(inputs, predictions)
    assert '<script>bad</script>' not in page
    assert '&lt;script&gt;bad&lt;/script&gt;' in page
    assert '&lt;img onerror=x&gt;' in page
    assert 'chưa duyệt' in page
    assert 'loading="lazy"' in page


@pytest.mark.parametrize('name', ['../outside.json', '/outside.json', 'C:/outside.json'])
def test_bundle_rejects_unsafe_member_paths(tmp_path, name):
    bundle = tmp_path / 'bad.zip'
    payload = b'{}'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr(name, payload)
        archive.writestr('checksums.json', json.dumps({name: hashlib.sha256(payload).hexdigest()}))
    with pytest.raises(ValueError, match='Unsafe ZIP'):
        verify_bundle(bundle, hashlib.sha256(bundle.read_bytes()).hexdigest())


def test_bundle_rejects_changed_download_before_parsing(tmp_path):
    bundle = tmp_path / 'changed.zip'
    bundle.write_bytes(b'not a zip')
    with pytest.raises(ValueError, match='SHA-256'):
        verify_bundle(bundle, '0' * 64)


def test_review_notes_do_not_invent_human_labels():
    inputs = [dict(question_id='q1', image_path='images/one.jpg', answer='cat')]
    predictions = {'q1': dict(prediction_answer='cat', prediction_explanation='A cat is visible.')}
    rows = review_notes(inputs, predictions, 'f' * 64)
    assert rows[0]['source_question_id'] == 'q1'
    assert rows[0]['bundle_sha256'] == 'f' * 64
    assert rows[0]['review_status'] == 'pending'
    assert rows[0]['reviewer'] is None
    assert rows[0]['answer_assessment'] is None
    assert rows[0]['proposed_chosen_explanation'] is None
    assert 'chosen' not in rows[0] and 'rejected' not in rows[0]


def test_page_does_not_hardcode_checkpoint_or_calibration_count():
    inputs = [dict(question_id='q1', image_path='images/one.jpg',
                   question='Animal?', answer='cat', explanations=['Cat.'])]
    predictions = {'q1': dict(prediction_answer='cat', prediction_explanation='Cat.',
                              adapter='/runs/checkpoint-900')}
    page = render_cases(inputs, predictions)
    assert 'checkpoint-900' in page
    assert 'checkpoint-800' not in page
    assert '100 ID' not in page
