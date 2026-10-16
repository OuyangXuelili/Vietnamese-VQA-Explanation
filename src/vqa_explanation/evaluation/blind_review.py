"""Freeze image samples, hide model identities and summarize independent ratings."""

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation.comparison import (
    EXPERIMENT_IDS,
    cluster_bootstrap,
    compare_predictions,
)
from vqa_explanation.utils.review_time import parse_review_time
from vqa_explanation.utils.run_provenance import sha256_file

LABEL_FIELDS = {
    "review_status", "reviewer", "reviewed_at", "answer_assessment",
    "explanation_support", "unsupported_detail", "image_evidence",
}


def _rating_metrics(row):
    return {
        "support": float(row["explanation_support"]),
        "correct_and_support2_rate": float(row["answer_assessment"] == "correct"
                                          and row["explanation_support"] == 2),
        "unsupported_yes_rate": float(row["unsupported_detail"] == "yes"),
        "unsupported_uncertain_rate": float(row["unsupported_detail"] == "uncertain"),
    }


def select_sample(records, *, count=200, seed=42):
    if type(count) is not int or count < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Use at least two images and a nonnegative seed")
    groups, ids = {}, set()
    for row in records:
        if row["split"] != "test":
            continue
        if row["question_id"] in ids:
            raise ValueError("Duplicate test question ID")
        ids.add(row["question_id"])
        groups.setdefault(row["image_id"], []).append(row)
    if len(groups) < count:
        raise ValueError("Not enough test images")
    rng = random.Random(seed)
    images = sorted(groups)
    rng.shuffle(images)
    return [rng.choice(sorted(groups[image], key=lambda row: row["question_id"]))
            for image in images[:count]]


def freeze_sample(annotations, output, *, count=200, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = select_sample(list(read_jsonl(annotations)), count=count, seed=seed)
    sample = {
        "schema": "blind_review_sample_v1", "split": "test", "seed": seed,
        "count": count, "annotations_sha256": sha256_file(Path(annotations)),
        "question_ids": [row["question_id"] for row in rows],
        "image_ids": [row["image_id"] for row in rows],
        "method": "seeded image shuffle, one seeded question per selected image",
    }
    write_json(output, sample)
    return sample


def render_blind_editor(notes):
    payload = json.dumps(notes, ensure_ascii=False).replace("<", "\\u003c")
    identity = hashlib.sha256(
        json.dumps(notes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self';
style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'">
<title>Đánh giá lời giải thích</title><style>
body{font:16px/1.5 Arial;max-width:1000px;margin:24px auto;padding:0 16px}
img{width:100%;height:360px;object-fit:contain}label{display:block;margin:10px 0}
textarea,select,button{font:inherit}textarea{width:100%;min-height:65px}
#answer,#explanation{white-space:pre-wrap}[hidden]{display:none}
</style><h1>Đánh giá theo ảnh</h1>
<p>Chấm độc lập. Mức hỗ trợ: 0 = sai hoặc không hỗ trợ; 1 = một phần;
2 = bằng chứng rõ và đủ hỗ trợ đáp án. Giữ nhãn mơ hồ khi ảnh chưa đủ xác định.
Ghi chi tiết không được ảnh hỗ trợ riêng. Trang tự lưu nháp trong trình duyệt;
xuất JSONL để lưu bền vững hoặc chuyển máy.</p>
<select id="picker" aria-label="Chọn đầu ra"></select>
<button id="export">Xuất JSONL</button>
<label>Nhập nhãn<input id="import" type="file" accept=".jsonl"></label>
<p id="status" role="status"></p><h2 id="question"></h2><img id="image" alt="Ảnh câu hỏi">
<h3>Đáp án</h3><p id="answer"></p><h3>Giải thích</h3><p id="explanation"></p>
<div id="form"></div><script type="application/json" id="data">""" + payload + """</script>
<script>(()=>{
const source=JSON.parse(document.getElementById('data').textContent);
const rows=source.map(row=>({...row}));
const storageKey='vqa-blind-draft-v1:""" + identity + """';
let dirty=false;
const picker=document.getElementById('picker'),status=document.getElementById('status');
const fields={review_status:'Trạng thái',reviewer:'Người chấm',
answer_assessment:'Đáp án theo ảnh',explanation_support:'Mức hỗ trợ (0–2)',
unsupported_detail:'Có chi tiết không được ảnh hỗ trợ?',image_evidence:'Bằng chứng / ghi chú'};
const options={review_status:['pending','reviewed'],
answer_assessment:['','correct','incorrect','uncertain'],explanation_support:['','0','1','2'],
unsupported_detail:['','yes','no','uncertain']};
const editable=[...Object.keys(fields),'reviewed_at'];
function validate(input,{complete=true}={}){
if(!Array.isArray(input)||input.length!==rows.length)throw Error('Số dòng không khớp');
input.forEach((row,i)=>{
if(!row||typeof row!=='object'||Array.isArray(row))throw Error('Dòng nhãn không hợp lệ');
Object.keys(source[i]).filter(k=>!editable.includes(k)).forEach(k=>{
if(JSON.stringify(row[k])!==JSON.stringify(source[i][k]))throw Error('Nguồn đầu ra không khớp');});
if(!['pending','reviewed'].includes(row.review_status))throw Error('Trạng thái không hợp lệ');
['reviewer','reviewed_at','image_evidence'].forEach(k=>{
if(row[k]!=null&&typeof row[k]!=='string')throw Error('Trường văn bản không hợp lệ');});
['answer_assessment','unsupported_detail'].forEach(k=>{
if(row[k]!=null&&row[k]!==''&&!options[k].includes(row[k]))throw Error('Nhãn không hợp lệ');});
if(row.explanation_support!=null&&![0,1,2].includes(row.explanation_support))
throw Error('Mức hỗ trợ không hợp lệ');
if(complete&&row.review_status==='reviewed'&&(!row.reviewer?.trim()||!row.image_evidence?.trim()
||!['correct','incorrect','uncertain'].includes(row.answer_assessment)
||![0,1,2].includes(row.explanation_support)
||!['yes','no','uncertain'].includes(row.unsupported_detail)))throw Error('Còn thiếu đánh giá');
if(complete&&row.review_status==='reviewed'&&(!row.reviewed_at
||!/(Z|[+-]\\d{2}:\\d{2})$/.test(row.reviewed_at)
||!Number.isFinite(Date.parse(row.reviewed_at))))throw Error('Thời gian cần có múi giờ');
});}
function replaceEditable(input){input.forEach((r,i)=>editable.forEach(k=>rows[i][k]=r[k]));}
function saveDraft(){
try{localStorage.setItem(storageKey,JSON.stringify(rows));
status.textContent='Đã lưu nháp trong trình duyệt; xuất JSONL để lưu bền vững.';}
catch(error){status.textContent='Không lưu được nháp; hãy Xuất JSONL ngay.';}}
function restoreDraft(){try{
const saved=localStorage.getItem(storageKey);if(!saved)return;
const input=JSON.parse(saved);validate(input,{complete:false});replaceEditable(input);dirty=true;
status.textContent='Đã khôi phục nháp; xuất JSONL để lưu bền vững.';
}catch(error){status.textContent='Nháp không hợp lệ hoặc không khớp nguồn; không khôi phục. '+
'Dữ liệu lưu cũ không bị xóa. Có thể nhập JSONL đã xuất.';}}
function refresh(){const old=picker.value;picker.replaceChildren();rows.forEach((r,i)=>{
const option=document.createElement('option');option.value=i;
option.textContent=`${i+1}/${rows.length} — ${r.review_status}`;picker.append(option);});
picker.value=old||'0';}
function draw(){const i=Number(picker.value),r=rows[i];
document.getElementById('question').textContent=r.question;
document.getElementById('image').src=r.image_path;
document.getElementById('answer').textContent=r.prediction_answer;
document.getElementById('explanation').textContent=r.prediction_explanation;
const form=document.getElementById('form');form.replaceChildren();
Object.entries(fields).forEach(([k,text])=>{const label=document.createElement('label');
label.textContent=text;const input=document.createElement(options[k]?'select':'textarea');
if(options[k])options[k].forEach(v=>{const o=document.createElement('option');o.value=v;
o.textContent=v||'Chưa ghi';input.append(o);});input.value=r[k]??'';
input.addEventListener('input',()=>{r[k]=k==='explanation_support'&&input.value!==''
?Number(input.value):input.value.trim()||null;r.reviewed_at=new Date().toISOString();
dirty=true;refresh();saveDraft();});label.append(input);
form.append(label);});}
picker.addEventListener('change',draw);
document.getElementById('export').addEventListener('click',()=>{try{validate(rows);
const url=URL.createObjectURL(new Blob([rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],
{type:'application/x-ndjson'}));const link=document.createElement('a');link.href=url;
link.download=rows[0].rater_id+'_ratings.jsonl';link.click();
setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='Kiểm tra file trong Downloads.';
dirty=false;
}catch(error){status.textContent=error.message;}});
document.getElementById('import').addEventListener('change',async event=>{try{
const text=await event.target.files[0].text();const input=text.split(/\\r?\\n/)
.filter(line=>line.trim()).map(JSON.parse);validate(input);
replaceEditable(input);dirty=true;refresh();draw();saveDraft();
}catch(error){status.textContent=error.message;}});
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
restoreDraft();refresh();draw();})();</script></html>"""


def prepare_blind_review(annotations, sample_path, prediction_paths, image_root, output_dir,
                         *, raters=("rater1", "rater2")):
    output, root = Path(output_dir), Path(image_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    if (len(raters) != len(set(raters)) or not raters
            or any(not r.isidentifier() for r in raters)):
        raise ValueError("Rater IDs must be distinct identifiers")
    canonical = list(read_jsonl(annotations))
    sample = json.loads(Path(sample_path).read_text("utf-8"))
    selected = select_sample(canonical, count=sample["count"], seed=sample["seed"])
    if (sample.get("schema") != "blind_review_sample_v1"
            or sample.get("annotations_sha256") != sha256_file(Path(annotations))
            or sample.get("question_ids") != [r["question_id"] for r in selected]
            or sample.get("image_ids") != [r["image_id"] for r in selected]):
        raise ValueError("Frozen human sample differs from canonical test")
    if set(prediction_paths) != set(EXPERIMENT_IDS):
        raise ValueError("Need all four M0–M3 exports")
    expected = {r["question_id"]: r for r in canonical if r["split"] == "test"}
    predictions = {role: list(read_jsonl(path)) for role, path in prediction_paths.items()}
    for role in EXPERIMENT_IDS[1:]:
        comparison = compare_predictions(
            predictions[EXPERIMENT_IDS[0]], predictions[role], expected_annotations=expected,
            expected_split="test", expected_experiments=(EXPERIMENT_IDS[0], role),
        )
        if not comparison["comparable"] or not comparison["explanation_rouge_l_f1"]["comparable"]:
            raise ValueError(f"Incompatible test exports: {role}: {comparison['issues']}")
    indexed = {role: {r["question_id"]: r for r in rows}
               for role, rows in predictions.items()}
    image_sources = []
    for row in selected:
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError("Missing review image or path outside image root")
        if sha256_file(image) != indexed[EXPERIMENT_IDS[0]][row["question_id"]]["image_sha256"]:
            raise ValueError("Review image differs from inference image")
        image_sources.append(image)
    mapping, pages = [], {}
    for rater in raters:
        rng = random.Random(f"{sample['seed']}:{rater}")
        notes = []
        for case, row in enumerate(selected):
            roles = list(EXPERIMENT_IDS)
            rng.shuffle(roles)
            for slot, role in enumerate(roles):
                entry = hashlib.sha256(f"{rater}:{case}:{slot}".encode()).hexdigest()[:20]
                prediction = indexed[role][row["question_id"]]
                notes.append({"entry_id": entry, "rater_id": rater,
                              "image_path": f"../images/{case:04d}.jpg",
                              "question": row["question"],
                              "prediction_answer": prediction["prediction_answer"],
                              "prediction_explanation": prediction["prediction_explanation"],
                              **dict.fromkeys(LABEL_FIELDS), "review_status": "pending"})
                mapping.append({"entry_id": entry, "rater_id": rater, "experiment_id": role,
                                "question_id": row["question_id"], "image_id": row["image_id"]})
        pages[rater] = notes
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for case, image in enumerate(image_sources):
        shutil.copyfile(image, output / "images" / f"{case:04d}.jpg")
    for rater, notes in pages.items():
        write_jsonl(output / rater / "blank_ratings.jsonl", notes)
        (output / rater / "index.html").write_text(render_blind_editor(notes), "utf-8")
    write_jsonl(output / "private/model_key.jsonl", mapping)
    write_json(output / "private/manifest.json", {
        "schema": "blind_review_bundle_v1", "sample_sha256": sha256_file(Path(sample_path)),
        "predictions": {role: sha256_file(Path(path)) for role, path in prediction_paths.items()},
        "raters": list(raters), "files": {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in output.rglob("*") if path.is_file()
        },
    })
    return output


def cohen_kappa(left, right, labels, *, quadratic=False):
    if not left or len(left) != len(right):
        raise ValueError("Agreement needs paired nonempty ratings")
    n = len(labels)
    confusion = np.zeros((n, n), dtype=float)
    for a, b in zip(left, right, strict=True):
        confusion[labels.index(a), labels.index(b)] += 1
    confusion /= len(left)
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0))
    weights = ((np.arange(n)[:, None] - np.arange(n)) / (n - 1)) ** 2 if quadratic else (
        1 - np.eye(n)
    )
    denominator = float((weights * expected).sum())
    return None if denominator == 0 else 1 - float((weights * confusion).sum()) / denominator


def summarize_blind_review(bundle, rating_paths):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "private/manifest.json").read_text("utf-8"))
    for relative, digest in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256_file(path) != digest:
            raise ValueError("Blind review bundle changed")
    key = {r["entry_id"]: r for r in read_jsonl(bundle / "private/model_key.jsonl")}
    if not rating_paths or not set(rating_paths).issubset(manifest["raters"]):
        raise ValueError("Unknown or empty rater set")
    results, by_rater, identities = {}, {}, set()
    for rater, path in rating_paths.items():
        baseline = {r["entry_id"]: r for r in read_jsonl(bundle / rater / "blank_ratings.jsonl")}
        ratings = list(read_jsonl(path))
        if len(ratings) != len(baseline) or {r["entry_id"] for r in ratings} != set(baseline):
            raise ValueError("Ratings must cover every blinded output exactly once")
        reviewers = set()
        grouped = {role: {} for role in EXPERIMENT_IDS}
        for row in ratings:
            source = baseline[row["entry_id"]]
            if any(row.get(k) != v for k, v in source.items() if k not in LABEL_FIELDS):
                raise ValueError("Immutable blinded output changed")
            if (row.get("review_status") != "reviewed"
                    or row.get("answer_assessment") not in {"correct", "incorrect", "uncertain"}
                    or type(row.get("explanation_support")) is not int
                    or row["explanation_support"] not in {0, 1, 2}
                    or row.get("unsupported_detail") not in {"yes", "no", "uncertain"}
                    or any(not isinstance(row.get(k), str) or not row[k].strip()
                           for k in ("reviewer", "reviewed_at", "image_evidence"))):
                raise ValueError("Human rating is pending or incomplete")
            parse_review_time(row["reviewed_at"])
            reviewers.add(row["reviewer"].strip())
            mapping = key[row["entry_id"]]
            grouped[mapping["experiment_id"]][mapping["question_id"]] = row
        if len(reviewers) != 1 or identities & reviewers:
            raise ValueError("Each rating file needs one distinct independent reviewer")
        identities |= reviewers
        by_rater[rater] = grouped
        results[rater] = {}
        for role, indexed in grouped.items():
            rows = list(indexed.values())
            correct = [r for r in rows if r["answer_assessment"] == "correct"]
            metrics = [_rating_metrics(row) for row in rows]
            results[rater][role] = {
                "n": len(rows),
                "answer_counts": dict(Counter(r["answer_assessment"] for r in rows)),
                "mean_support": float(np.mean([r["explanation_support"] for r in rows])),
                "support_given_correct_n": len(correct),
                "support_given_correct": float(np.mean([r["explanation_support"] for r in correct]))
                if correct else None,
                **{name: float(np.mean([values[name] for values in metrics]))
                   for name in metrics[0] if name != "support"},
                "unsupported_counts": dict(Counter(r["unsupported_detail"] for r in rows)),
            }
        ids = sorted(grouped[EXPERIMENT_IDS[2]])
        left, right = grouped[EXPERIMENT_IDS[2]], grouped[EXPERIMENT_IDS[3]]
        image_ids = {r["question_id"]: r["image_id"] for r in key.values()}
        paired = [(_rating_metrics(left[q]), _rating_metrics(right[q])) for q in ids]
        results[rater]["M3_minus_M2"] = cluster_bootstrap(
            {name: [b[name] - a[name] for a, b in paired] for name in paired[0][0]},
            [image_ids[q] for q in ids],
        )
    agreement = None
    if len(by_rater) == 2:
        a, b = list(by_rater.values())
        agreement = {}
        for role in EXPERIMENT_IDS:
            ids = sorted(a[role])
            agreement[role] = {
                field: cohen_kappa([a[role][q][field] for q in ids],
                                   [b[role][q][field] for q in ids], labels,
                                   quadratic=field == "explanation_support")
                for field, labels in {
                    "answer_assessment": ["correct", "incorrect", "uncertain"],
                    "explanation_support": [0, 1, 2],
                    "unsupported_detail": ["yes", "no", "uncertain"],
                }.items()
            }
    return {"schema": "blind_review_results_v1", "status": "complete", "raters": results,
            "agreement": agreement, "metric_protocol": {
                "contrast": "M3_minus_M2",
                "primary": "support",
                "secondary": ["correct_and_support2_rate", "unsupported_yes_rate",
                              "unsupported_uncertain_rate"],
                "rate_denominator": "all sampled outputs per model and rater",
                "rate_unit": "proportion, not percentage points",
                "uncertainty": "Keep uncertain labels in the denominator and report separately. "
                "A lower yes rate alone does not establish fewer unsupported details.",
                "confidence_intervals": "Pointwise paired image-bootstrap intervals; "
                "secondary metrics are descriptive, not multiplicity-adjusted claims.",
            }, "limitations":
            "Reviewer metadata cannot prove independence. Bootstrap measures sampling uncertainty, "
            "not training-seed stability. Null kappa means zero expected disagreement."}
