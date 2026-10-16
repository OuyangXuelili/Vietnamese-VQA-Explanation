"""Generate unreviewed train candidates through the resumable prediction exporter."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl
from vqa_explanation.evaluation.answer_metrics import exact_match
from vqa_explanation.inference.export import export_predictions, validate_selected_records
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.training.preference_data import adapter_fingerprint, validate_selection
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def candidate_records(selection, annotations, image_root, expected_sha256, count=100):
    selection, annotations = Path(selection), Path(annotations)
    if type(count) is not int or not 1 <= count <= 2000:
        raise ValueError("count must be between 1 and 2000")
    if sha256_file(annotations) != expected_sha256:
        raise ValueError("Canonical annotation hash changed")
    manifest = json.loads((selection / "manifest.json").read_text(encoding="utf-8"))
    questions_path = selection / "questions.jsonl"
    if (manifest.get("schema_version") != 1
            or manifest.get("status") != "questions_selected_not_reviewed"
            or manifest.get("annotations_sha256") != expected_sha256
            or manifest.get("files", {}).get("questions.jsonl") != sha256_file(questions_path)):
        raise ValueError("Selection manifest/hash changed")
    questions = list(read_jsonl(questions_path))
    if len(questions) != manifest.get("count") or count > len(questions):
        raise ValueError("Selection count differs from the manifest or requested count")
    if manifest.get("calibration_question_ids") != [
        r['source_question_id'] for r in questions[:100]
    ]:
        raise ValueError("Calibration IDs differ from the frozen prefix")
    canonical = list(read_jsonl(annotations))
    audit = audit_records(canonical, image_root=image_root)
    if (audit['errors'] or audit['counts'].get('duplicate_question_ids', 0)
            or any(audit['image_overlap_by_split'].values())
            or any(audit['image_path_overlap_by_split'].values())
            or any(not audit['counts'].get(f'split_{split}')
                   for split in ('train', 'validation', 'test'))):
        raise ValueError("Canonical annotations failed the split/ID audit")
    by_id = {r['question_id']: r for r in canonical}
    ids, images, paths = set(), set(), set()
    selected = []
    for index, row in enumerate(questions):
        qid = row.get('source_question_id')
        reference = by_id.get(qid)
        if not reference or reference['split'] != 'train' or row.get('split') != 'train':
            raise ValueError("Selection must contain canonical train questions only")
        expected = {
            'image_id': reference['image_id'], 'image_path': reference['image_path'],
            'question': reference['question'], 'reference_answer': reference['answer'],
            'reference_explanations': reference['explanations'],
            'annotations_sha256': expected_sha256,
            'prompt': build_user_prompt(reference['question']), 'review_status': 'pending',
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Selection reference changed: {qid}")
        if qid in ids or row['image_id'] in images or row['image_path'] in paths:
            raise ValueError("Selection contains duplicate questions or images")
        ids.add(qid)
        images.add(row['image_id'])
        paths.add(row['image_path'])
        if index < count:
            selected.append(reference)
            validate_selected_records([reference], Path(image_root))
            if sha256_file(Path(image_root) / row['image_path']) != row.get('image_sha256'):
                raise ValueError(f"Selected image hash changed: {qid}")
    context = {
        'purpose': 'preference_candidates_pending_review',
        'annotations_sha256': expected_sha256,
        'selection_sha256': sha256_file(questions_path),
        'selection_manifest_sha256': sha256_file(selection / 'manifest.json'),
        'count': count, 'selection_order': 'frozen_prefix',
        'source_files': {
            name: sha256_file(project_root() / name) for name in (
                'scripts/generate_preference_candidates.py',
                'src/vqa_explanation/preference/candidates.py',
                'src/vqa_explanation/preference/selection.py',
            )
        },
    }
    return selected, context


def _write_or_verify(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact changed: {path}")
    else:
        with path.open('xb') as destination:
            destination.write(data)


def generate_candidates(args):
    records, context = candidate_records(
        args.selection, args.annotations, args.image_root, args.expected_sha256, args.count
    )
    m1_selection = getattr(args, 'm1_selection', None)
    if m1_selection is not None:
        if sha256_file(m1_selection) != args.m1_selection_sha256:
            raise ValueError("Frozen M1 selection hash changed")
        parent = adapter_fingerprint(args.adapter)
        validate_selection(
            m1_selection, parent, {'id': args.model_id, 'revision': args.revision},
            args.expected_sha256, list(read_jsonl(args.annotations)),
        )
        context['m1_selection_sha256'] = args.m1_selection_sha256
        context['m1_adapter'] = parent
        context['source_files']['src/vqa_explanation/training/preference_data.py'] = sha256_file(
            project_root() / 'src/vqa_explanation/training/preference_data.py'
        )
    output = args.output_dir
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite candidate run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + '.lock', timeout=0):
        output.mkdir(exist_ok=args.resume)
        inputs = output / 'inputs.jsonl'
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode('utf-8')
        _write_or_verify(inputs, payload)
        predictions = output / 'candidates.jsonl'
        manifest = predictions.with_suffix('.jsonl.manifest.json')
        export_args = SimpleNamespace(
            **{name: getattr(args, name) for name in (
                'model_id', 'revision', 'adapter', 'seed', 'max_new_tokens',
                'min_pixels', 'max_pixels', 'load_in_4bit', 'image_root',
            )},
            annotations=inputs, output=predictions, split='train', limit=0,
            experiment_id='M1_SFT', export_context=context,
            resume=args.resume and manifest.exists(),
        )
        export_predictions(export_args)
        rows = list(read_jsonl(predictions))
        if len(rows) != len(records):
            raise ValueError("Candidate export is incomplete")
        report = {
            'status': 'generated_pending_review', 'n': len(rows),
            'format_valid': sum(r['format_valid'] for r in rows),
            'reference_answer_em': sum(exact_match(r['prediction_answer'], r['reference_answer'])
                                       for r in rows) / len(rows),
            'predictions_sha256': sha256_file(predictions),
            'manifest_sha256': sha256_file(manifest),
            'inputs_sha256': hashlib.sha256(payload).hexdigest(),
            'limitations': 'Train candidates, not held-out evaluation. No preference labels '
                           'or accepted pairs. References and image evidence need human review.',
        }
        _write_or_verify(output / 'summary.json',
                         (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        return report
