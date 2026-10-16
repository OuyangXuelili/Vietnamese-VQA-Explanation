"""Build a read-only image review page from a verified candidate bundle."""

import argparse
import json
from html import escape
from pathlib import Path

from vqa_explanation.evaluation.answer_metrics import exact_match
from vqa_explanation.preference.bundle import verify_bundle
from vqa_explanation.preference.review import blank_notes, render_editor

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self';
style-src 'unsafe-inline'">
<title>Duyệt ứng viên M1</title><style>
body{font:16px/1.5 Arial,sans-serif;color:#1b2935;background:#f4f6f8;margin:0}
main{max-width:1160px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 12px}
header,article{background:white;border:1px solid #d5dce3;border-radius:8px;padding:20px;
margin-bottom:18px}h2{font-size:19px;margin:0 0 14px}h3{font-size:15px;margin:12px 0 4px}
.case{display:grid;grid-template-columns:minmax(240px,1fr) 1fr;gap:24px}
img{width:100%;height:320px;object-fit:contain;background:#eef1f4}
.muted{font-size:14px;color:#596675}.flag{color:#9b4300}.text{white-space:pre-wrap}
li{margin:6px 0}code{overflow-wrap:anywhere}@media(max-width:760px){.case{display:block}}
</style></head><body><main>__CONTENT__</main></body></html>"""


def review_notes(inputs, predictions, bundle_sha256):
    """Create blank review records, not trainable preference pairs."""
    return [
        {
            'schema': 'candidate_review_v1', 'source_question_id': row['question_id'],
            'image_path': row['image_path'], 'bundle_sha256': bundle_sha256,
            'reference_answer': row['answer'],
            'candidate_answer': predictions[row['question_id']]['prediction_answer'],
            'candidate_explanation': predictions[row['question_id']]['prediction_explanation'],
            'review_status': 'pending', 'reviewer': None, 'reviewed_at': None,
            'answer_assessment': None, 'explanation_support': None,
            'unsupported_span': None, 'image_evidence': None,
            'proposed_chosen_explanation': None, 'exclusion_reason': None,
        }
        for row in inputs
    ]


def render_cases(inputs, predictions):
    adapters = {row.get('adapter') for row in predictions.values()}
    checkpoint = next(iter(adapters)) if len(adapters) == 1 else None
    checkpoint = Path(checkpoint).name if checkpoint else 'xem manifest'
    sections = [f'''<header><h1>{len(inputs)} ứng viên M1 — chưa duyệt</h1>
<p>Xem ảnh để kiểm đáp án và lời giải thích. Khác nhãn tham chiếu không tự động có nghĩa
là model sai; lời giải thích tham chiếu cũng có thể sai hoặc dịch chưa đúng.</p>
<p>Trang này chỉ để đọc. Chưa tạo chosen/rejected hoặc nhãn accepted. Khi duyệt cần ghi
bằng chứng trong ảnh, cụm sai và lý do loại. Không train M2/M3 bằng các ứng viên thô.</p>
<p>Ghi nhãn vào <code>review_notes.jsonl</code> trong thư mục này. Các trường người
duyệt và đánh giá đang trống; file ghi chú không phải đầu vào trainer/validator cặp.</p>
<p class="muted">Train-only · {escape(checkpoint)} · thứ tự ID đầu của pool cố định</p>
</header>''']
    for index, row in enumerate(inputs, 1):
        prediction = predictions[row['question_id']]
        answer = prediction['prediction_answer']
        match = bool(exact_match(answer, row['answer']))
        flag = 'Khớp đáp án tham chiếu' if match else 'Đáp án khác tham chiếu — cần kiểm ảnh'
        refs = ''.join(f'<li>{escape(text)}</li>' for text in row['explanations'])
        sections.append(f'''<article id="case-{index}">
<h2>{index}. {escape(row['question'])}</h2><div class="case">
<div><img loading="lazy" src="{escape(row['image_path'], quote=True)}" alt="Ảnh câu hỏi">
<p class="muted">ID: <code>{escape(row['question_id'])}</code> · chưa duyệt</p></div><div>
<h3>Đáp án M1</h3><div class="text">{escape(answer)}</div><p class="flag">{flag}</p>
<h3>Giải thích M1</h3><div class="text">{escape(prediction['prediction_explanation'])}</div>
<h3>Đáp án tham chiếu</h3><div class="text">{escape(row['answer'])}</div>
<details><summary>Giải thích tham chiếu</summary><ul>{refs}</ul></details>
</div></div></article>''')
    return PAGE.replace('__CONTENT__', '\n'.join(sections))


def prepare_review(bundle, expected_sha256, output_dir, *, editable=False):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite review artifacts: {output}')
    data, inputs, predictions = verify_bundle(bundle, expected_sha256)
    page = render_cases(inputs, predictions)
    notes = review_notes(inputs, predictions, expected_sha256)
    if editable:
        notes = blank_notes(inputs, predictions, expected_sha256)
        start = page.index('<header>')
        end = page.index('</header>', start) + len('</header>')
        page = page[:start] + render_editor(notes) + page[end:]
        page = page.replace("style-src 'unsafe-inline'", "style-src 'unsafe-inline'; "
                            "script-src 'unsafe-inline'; connect-src 'none'")
        page = page.replace('</style>', '#review-editor{background:white;padding:20px;'
                            'margin-bottom:18px}#review-form{display:grid;gap:10px}'
                            '#review-form label{display:grid}textarea{min-height:55px}'
                            'select,textarea,button{font:inherit}</style>')
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in data.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / 'review_notes.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in notes), encoding='utf-8'
    )
    (output / 'index.html').write_text(page, encoding='utf-8')
    return output / 'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--editable', action='store_true',
                        help='Offline editor; export labels as JSONL')
    args = parser.parse_args()
    print(prepare_review(args.bundle, args.expected_sha256, args.output_dir,
                         editable=args.editable))


if __name__ == '__main__':
    main()
