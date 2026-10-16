"""Prepare image contact sheets and import explicit visual decisions."""

import argparse
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path

from filelock import FileLock
from PIL import Image, ImageDraw, ImageFont

from vqa_explanation.data.formatting import format_target
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.inference.export import validate_saved_predictions
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.preference.expansion import validate_proposal, write_results
from vqa_explanation.preference.freeze import prepare_exports
from vqa_explanation.preference.quality import assess_reviews, candidate_origin, pair_digest
from vqa_explanation.utils.run_provenance import sha256_file, utc_now

RULES = "direct_visual_v2: inspect full image; verify canonical answer; declarative grounded " \
        "chosen; plausible image-dependent false detail in rejected; no text-only contradiction; " \
        "explain question relevance per pair; retain uncertainty; no human-review claim"
RECIPE = hashlib.sha256(RULES.encode()).hexdigest()
QUALITY_FIELDS = ('answer_correct', 'chosen_supported', 'rejected_unsupported',
                  'both_linguistically_plausible', 'language_only_preference',
                  'contrast_relevant', 'relevance_reason', 'text_only_reason')


def load_decisions(path):
    value = json.loads(path.read_text('utf-8'))
    if isinstance(value, dict):
        if value.get('schema') != 'direct_visual_decisions_v2' or 'assessment' in value:
            raise ValueError('Use v2 decisions with per-pair assessments, not shared approval')
        value = value.get('decisions')
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError('Supply a decision list')
    if any(row.get('status') == 'accepted' and not set(QUALITY_FIELDS).issubset(row)
           for row in value):
        raise ValueError('Each accepted pair needs its own complete assessment')
    return value


def prepare(args):
    if args.start < 0 or args.count < 1:
        raise ValueError('Sheet start must be non-negative and count must be positive')
    canonical = {r['question_id']: r for r in read_jsonl(args.annotations)}
    candidates = list(read_jsonl(args.candidates))
    candidate_manifest_path = Path(str(args.candidates) + '.manifest.json')
    candidate_manifest = json.loads(candidate_manifest_path.read_text('utf-8'))
    origin = candidate_origin(candidate_manifest['inputs'])
    try:
        references = [canonical[row['question_id']] for row in candidates]
    except KeyError as error:
        raise ValueError('Candidate is absent from canonical annotations') from error
    validate_saved_predictions(candidates, references, candidate_manifest)
    selected = candidates[args.start:args.start + args.count]
    if len(selected) != args.count:
        raise ValueError('Requested batch exceeds candidates')
    folder = args.output_dir / f'batch_{args.start:05d}'
    folder.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(args.font, 21)
    board = Image.new('RGB', (1800, 540 * ((args.count + 2) // 3)), 'white')
    draw = ImageDraw.Draw(board)
    rows = []
    for index, candidate in enumerate(selected):
        sample = canonical[candidate['question_id']]
        if sample['split'] != 'train' or sample['image_path'] != candidate['image_path']:
            raise ValueError('Candidate does not match canonical train')
        path = args.image_root / sample['image_path']
        digest = sha256_file(path)
        if digest != candidate['image_sha256']:
            raise ValueError('Candidate image changed')
        x, y = (index % 3) * 600, (index // 3) * 540
        with Image.open(path) as opened:
            picture = opened.convert('RGB')
            picture.thumbnail((580, 380))
            board.paste(picture, (x + (600 - picture.width) // 2, y + 25))
        lines = [f'{index + 1}. ID {sample["question_id"]}']
        lines += textwrap.wrap(sample['question'], 45)
        lines += textwrap.wrap('Đáp án: ' + sample['answer'], 45)
        draw.multiline_text((x + 12, y + 410), '\n'.join(lines), font=font, fill='black', spacing=3)
        rows.append({**sample, 'image_sha256': digest, 'candidate_raw': candidate['raw_output'],
                     'original_candidate_origin': origin})
    board.save(folder / 'images.png')
    write_jsonl(folder / 'items.jsonl', rows)
    write_json(folder / 'manifest.json', {'schema': 'visual_sheet_v2', 'start': args.start,
               'annotations_sha256': sha256_file(args.annotations),
               'candidates_sha256': sha256_file(args.candidates),
               'candidate_manifest_sha256': sha256_file(candidate_manifest_path),
               'original_candidate_origin': origin,
               'items_sha256': sha256_file(folder / 'items.jsonl'),
               'sheet_sha256': sha256_file(folder / 'images.png'), 'rules': RULES})
    print(folder)


def ingest(args):
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir) + '.lock', timeout=0):
        return _ingest(args)


def active_records(output, records):
    path = output / 'withdrawals.jsonl'
    withdrawn = list(read_jsonl(path)) if path.exists() else []
    known = {r['question_id']: r for r in records}
    reasons = {}
    for row in withdrawn:
        record = known.get(row['question_id'])
        if (record is None or record.get('pair') is None
                or pair_digest(record['pair']) != row['pair_sha256']
                or row['question_id'] in reasons or not row.get('reason')):
            raise ValueError('Invalid or stale withdrawal')
        reasons[row['question_id']] = row['reason']
    return [{**r, 'pair': None, 'quality': None, 'status': 'withdrawn',
             'withdrawal_reason': reasons[r['question_id']]} if r['question_id'] in reasons
            else r for r in records]


def withdraw(args):
    with FileLock(str(args.output_dir) + '.lock', timeout=0):
        records = list(read_jsonl(args.output_dir / 'decisions.jsonl'))
        current = active_records(args.output_dir, records)
        matched = [r for r in current if r['question_id'] == args.question_id and r.get('pair')]
        if len(matched) != 1 or not args.reason.strip():
            raise ValueError('Withdrawal needs one active pair and an explicit reason')
        row = {'question_id': args.question_id, 'pair_sha256': pair_digest(matched[0]['pair']),
               'reason': args.reason, 'reviewer_type': 'ai', 'human_verified': False,
               'reviewed_at': utc_now()}
        with (args.output_dir / 'withdrawals.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = write_results(args.output_dir, active_records(args.output_dir, records),
                               1000, args.image_root)
        print(json.dumps(result, indent=2))


def _ingest(args):
    folder = args.batch
    manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
    if manifest.get('schema') != 'visual_sheet_v2':
        raise ValueError('Prepare a v2 sheet with verified candidate lineage before import')
    origin = manifest.get('original_candidate_origin')
    if (not isinstance(origin, dict)
            or not isinstance(origin.get('model_id'), str)
            or not origin['model_id'].strip()
            or not isinstance(origin.get('model_revision'), str)
            or not re.fullmatch(r'[0-9a-f]{40}', origin['model_revision'])
            or not isinstance(origin.get('checkpoint_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', origin['checkpoint_sha256'])
            or type(origin.get('seed')) is not int or origin['seed'] < 0
            or any(not isinstance(manifest.get(key), str)
                   or not re.fullmatch(r'[0-9a-f]{64}', manifest[key])
                   for key in ('candidates_sha256', 'candidate_manifest_sha256'))):
        raise ValueError('Review sheet is missing candidate lineage')
    if (manifest['annotations_sha256'] != sha256_file(args.annotations)
            or manifest['items_sha256'] != sha256_file(folder / 'items.jsonl')
            or manifest['sheet_sha256'] != sha256_file(folder / 'images.png')):
        raise ValueError('Review inputs changed')
    items = {r['question_id']: r for r in read_jsonl(folder / 'items.jsonl')}
    decisions = load_decisions(args.decisions)
    if len(decisions) != len(items) or {r['question_id'] for r in decisions} != set(items):
        raise ValueError('Supply one explicit decision per displayed image')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    journal = output / 'decisions.jsonl'
    previous = list(read_jsonl(journal)) if journal.exists() else []
    if any(r['question_id'] in items for r in previous):
        raise ValueError('Batch was already imported')
    checked = []
    decision_hash = sha256_file(args.decisions)
    for decision in decisions:
        sample = items[decision['question_id']]
        if sample.get('original_candidate_origin') != origin:
            raise ValueError('Review item candidate lineage differs from sheet')
        record = {'question_id': sample['question_id'], 'status': decision['status'],
                  'decision': decision, 'sheet_sha256': manifest['sheet_sha256'],
                  'decision_file_sha256': decision_hash, 'pair': None,
                  'quality': None, 'reviewed_at': utc_now()}
        if decision['status'] in {'excluded', 'uncertain'}:
            if (not isinstance(decision.get('reason'), str)
                    or not decision['reason'].strip()):
                raise ValueError('Excluded or uncertain decisions need a reason')
            checked.append(record)
            continue
        if decision['status'] != 'accepted':
            raise ValueError('Unknown visual decision')
        proposal = {'usable': True, **decision}
        validate_proposal(proposal)
        if decision['chosen'].strip().casefold() == sample['question'].strip().casefold():
            raise ValueError('Chosen cannot be a copy of the question')
        if sha256_file(args.image_root / sample['image_path']) != sample['image_sha256']:
            raise ValueError('Image changed after review')
        pair = {'pair_id': 'v2-' + sample['question_id'],
                'source_question_id': sample['question_id'], 'image_id': sample['image_id'],
                'image_path': sample['image_path'], 'split': 'train',
                'question': sample['question'], 'prompt': build_user_prompt(sample['question']),
                'chosen': format_target(sample['answer'], decision['chosen']),
                'rejected': format_target(sample['answer'], decision['rejected']),
                'annotations_sha256': manifest['annotations_sha256'],
                'image_sha256': sample['image_sha256'], 'error_type': decision['error_type'],
                'generation_method': 'ai_review_candidate', 'reviewer_type': 'ai',
                'reviewer': 'direct_visual_review', 'human_verified': False,
                'review_status': 'accepted', 'reviewed_at': record['reviewed_at'],
                'evidence': decision['evidence'], 'rejected_span': decision['rejected_span'],
                'provenance': {side: {'kind': 'synthetic', 'recipe_sha256': RECIPE,
                                     'source': f'sha256:{decision_hash}#{sample["question_id"]}'}
                               for side in ('chosen', 'rejected')}}
        pair['provenance']['original_candidate_origin'] = {
            **origin, 'candidates_sha256': manifest['candidates_sha256'],
            'candidate_manifest_sha256': manifest['candidate_manifest_sha256'],
            'candidate_raw_sha256': hashlib.sha256(sample['candidate_raw'].encode()).hexdigest(),
        }
        for side in ('chosen', 'rejected'):
            if pair[side] == sample['candidate_raw']:
                pair['provenance'][side] = {
                    'kind': 'model', 'source': f"sha256:{manifest['candidates_sha256']}"
                    f"#{sample['question_id']}", 'model_id': origin['model_id'],
                    'checkpoint_sha256': origin['checkpoint_sha256'], 'seed': origin['seed'],
                    'recipe_sha256': manifest['candidate_manifest_sha256'],
                    'recipe_scope': 'candidate_export_manifest',
                }
        quality = {'pair_id': pair['pair_id'], 'pair_sha256': pair_digest(pair),
                   'review_version': 2,
                   'status': 'confirmed', 'reviewer_type': 'ai',
                   'visible_evidence': decision['evidence'],
                   'rejected_span': decision['rejected_span'],
                   'method': 'direct_image_inspection', 'independent_human_verified': False}
        for field in QUALITY_FIELDS:
            quality[field] = decision.get(field)
        if not assess_reviews([pair], [quality], 1, required_version=2)['ready']:
            raise ValueError('Explicit quality decision is incomplete or failed')
        record.update(pair=pair, quality=quality)
        checked.append(record)
    all_records = active_records(output, previous + checked)
    pairs = [r['pair'] for r in all_records if r['pair']]
    if len({p['image_sha256'] for p in pairs}) != len(pairs):
        raise ValueError('Repeated image bytes in accepted pairs')
    if pairs:
        prepare_exports(pairs, list(read_jsonl(args.annotations)), args.image_root,
                        manifest['annotations_sha256'])
    if sha256_file(args.decisions) != decision_hash:
        raise ValueError('Decisions changed during import')
    with journal.open('a', encoding='utf-8') as handle:
        for record in checked:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(write_results(output, all_records, 1000, args.image_root), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    sheet = modes.add_parser('sheet')
    sheet.add_argument('--candidates', type=Path, required=True)
    sheet.add_argument('--start', type=int, required=True)
    sheet.add_argument('--count', type=int, default=12)
    sheet.add_argument('--font', required=True)
    import_mode = modes.add_parser('import')
    import_mode.add_argument('--batch', type=Path, required=True)
    import_mode.add_argument('--decisions', type=Path, required=True)
    withdraw_mode = modes.add_parser('withdraw')
    withdraw_mode.add_argument('--question-id', required=True)
    withdraw_mode.add_argument('--reason', required=True)
    withdraw_mode.add_argument('--image-root', type=Path, required=True)
    withdraw_mode.add_argument('--output-dir', type=Path, required=True)
    for mode in (sheet, import_mode):
        mode.add_argument('--annotations', type=Path, required=True)
        mode.add_argument('--image-root', type=Path, required=True)
        mode.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    {'sheet': prepare, 'import': ingest, 'withdraw': withdraw}[args.mode](args)


if __name__ == '__main__':
    main()
