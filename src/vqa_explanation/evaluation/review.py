"""Prepare a fixed image-based validation review without assigning labels."""

import hashlib
import json
import shutil
from html import escape
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.data.schema import VQASample
from vqa_explanation.inference.export import select_records, validate_saved_predictions
from vqa_explanation.utils.run_provenance import sha256_file

RECIPE = "validation_review_image_hash_v1"
REVIEW_FIELDS = (
    "review_status", "reviewer", "reviewed_at", "answer_assessment",
    "explanation_support", "image_evidence", "unsupported_span", "error_type",
)


def select_review_records(records, *, count=100, seed=42):
    """Select one question per image, independently of model outputs."""
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Use a positive count and a nonnegative integer seed")
    groups, seen = {}, set()
    for row in records:
        sample = VQASample.from_dict(row)
        if sample.split != "validation" or sample.question_id in seen:
            raise ValueError("Review requires unique validation questions")
        seen.add(sample.question_id)
        groups.setdefault(sample.image_id, []).append(row)
    if len(groups) < count:
        raise ValueError("Not enough distinct validation images")

    def rank(kind, identifier):
        return hashlib.sha256(f"{RECIPE}:{seed}:{kind}:{identifier}".encode()).hexdigest()

    images = sorted(groups, key=lambda image: (rank("image", image), image))[:count]
    return [min(groups[image], key=lambda row: (rank("question", row["question_id"]),
                                              row["question_id"])) for image in images]


def render_review(notes):
    payload = json.dumps(notes, ensure_ascii=False).replace("<", "\\u003c")
    cards = []
    for index, row in enumerate(notes):
        refs = "".join(f"<li>{escape(text)}</li>" for text in row["reference_explanations"])
        cards.append(f"""<article data-case="{index}" {'hidden' if index else ''}>
<h2>{index + 1}. {escape(row['question'])}</h2>
<div class="case"><div><img src="{escape(row['image_path'], quote=True)}" alt="Ảnh câu hỏi">
<p class="muted">ID {escape(row['question_id'])}</p></div><div>
<h3>Đáp án M1</h3><p>{escape(row['prediction_answer'])}</p>
<h3>Giải thích M1</h3><p>{escape(row['prediction_explanation'])}</p>
<details><summary>Tham chiếu — có thể có lỗi</summary>
<p>{escape(row['reference_answer'])}</p><ul>{refs}</ul></details>
</div></div></article>""")
    return """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self';
style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'">
<title>Rà chất lượng M1</title><style>
body{font:16px/1.5 Arial,sans-serif;background:#f4f6f8;color:#20303d;margin:0}
main{max-width:1120px;margin:auto;padding:24px}header,article,section{background:white;
padding:20px;margin-bottom:16px;border:1px solid #d5dce3;border-radius:8px}
h1{font-size:25px}h2{font-size:21px}h3{font-size:16px}p{white-space:pre-wrap}
.case{display:grid;grid-template-columns:1fr 1fr;gap:24px}
img{width:100%;height:360px;object-fit:contain;background:#eef1f4}
.muted{color:#596675;font-size:14px}#form{display:grid;gap:10px}
label{display:grid}select,textarea,button{font:inherit}textarea{min-height:48px}
[hidden]{display:none!important}@media(max-width:760px){.case{display:block}}
</style></head><body><main><header><h1>Rà chất lượng M1 — validation</h1>
<p>Ảnh cố định, một câu mỗi ảnh. Đây là kiểm tra chất lượng, không phải dữ liệu train.</p>
<p>Nhìn ảnh để chấm đáp án và giải thích. Mức hỗ trợ: 0 = sai hoặc không hỗ trợ đáp án;
1 = hỗ trợ một phần, còn thiếu hoặc có chi tiết không chắc; 2 = hỗ trợ rõ, không có
chi tiết sai đáng kể. Khác tham chiếu không tự động có nghĩa là model sai.</p>
<p>Nhãn chưa được gán. Bấm Xuất JSONL để lưu, rồi nhập lại file khi tiếp tục.</p>
<select id="picker" aria-label="Chọn câu"></select>
<button id="export">Xuất JSONL</button>
<label>Nhập nhãn đã lưu<input id="import" type="file" accept=".jsonl"></label>
<p id="status" role="status"></p></header>""" + "\n".join(cards) + """
<section><h2>Ghi đánh giá</h2><div id="form"></div></section>
<script type="application/json" id="data">""" + payload + """</script><script>
(() => {
  const notes=JSON.parse(document.getElementById('data').textContent);
  const picker=document.getElementById('picker'), form=document.getElementById('form');
  const status=document.getElementById('status');
  const fields={review_status:'Trạng thái',reviewer:'Người duyệt',
    answer_assessment:'Đáp án theo ảnh',explanation_support:'Mức hỗ trợ (0–2)',
    image_evidence:'Bằng chứng trong ảnh',unsupported_span:'Cụm sai hoặc không được ảnh hỗ trợ',
    error_type:'Loại lỗi / ghi chú'};
  const options={review_status:['pending','reviewed'],
    answer_assessment:['','correct','incorrect','uncertain'],explanation_support:['','0','1','2']};
  const editable=[...Object.keys(fields),'reviewed_at'];
  function refresh() {
    const old=picker.value; picker.replaceChildren();
    notes.forEach((row,i)=>{const item=document.createElement('option');item.value=i;
      item.textContent=`${i+1}. ${row.question_id} — ${row.review_status}`;picker.append(item);});
    picker.value=old || '0';
  }
  function draw() {
    const index=Number(picker.value), row=notes[index];form.replaceChildren();
    document.querySelectorAll('article').forEach((card,i)=>{card.hidden=i!==index;});
    Object.entries(fields).forEach(([key,label])=>{
      const wrapper=document.createElement('label');wrapper.textContent=label;
      const input=document.createElement(options[key] ? 'select' : 'textarea');
      if(options[key]) options[key].forEach(value=>{const item=document.createElement('option');
        item.value=value;item.textContent=value || 'Chưa ghi';input.append(item);});
      input.value=row[key] == null ? '' : String(row[key]);
      input.addEventListener('change',()=>{
        const value=input.value.trim();row[key]=key==='explanation_support' && value!==''
          ? Number(value) : (value || null);row.reviewed_at=new Date().toISOString();
        picker.options[index].textContent=`${index+1}. ${row.question_id} — ${row.review_status}`;
        status.textContent='Chưa lưu ra file. Bấm Xuất JSONL để giữ nhãn.';
      });wrapper.append(input);form.append(wrapper);
    });
  }
  function validate(rows) {
    rows.forEach(row=>{
      if(!['pending','reviewed'].includes(row.review_status))
        throw Error('Trạng thái không hợp lệ');
      if(row.review_status==='reviewed' && (!row.reviewer || !row.reviewer.trim()
        || !['correct','incorrect','uncertain'].includes(row.answer_assessment)
        || ![0,1,2].includes(row.explanation_support) || !row.image_evidence
        || !row.image_evidence.trim())) throw Error(`Câu ${row.question_id} còn thiếu đánh giá`);
    });
  }
  picker.addEventListener('change',draw);
  document.getElementById('export').addEventListener('click',()=>{
    try {validate(notes);const text=notes.map(row=>JSON.stringify(row)).join('\\n')+'\\n';
      const url=URL.createObjectURL(new Blob([text],{type:'application/x-ndjson'}));
      const link=document.createElement('a');link.href=url;link.download='m1_review_100.jsonl';
      link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      status.textContent='Đã yêu cầu tải file nhãn; kiểm tra Downloads.';
    } catch(error) {status.textContent=error.message;}
  });
  document.getElementById('import').addEventListener('change',async event=>{
    try {const text=await event.target.files[0].text();
      const rows=text.split(/\\r?\\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));
      if(rows.length!==notes.length) throw Error('Số câu không khớp');
      rows.forEach((row,i)=>Object.keys(notes[i]).filter(key=>!editable.includes(key))
        .forEach(key=>{if(JSON.stringify(row[key])!==JSON.stringify(notes[i][key]))
          throw Error('Nguồn câu hoặc phiên bản M1 không khớp');}));
      validate(rows);rows.forEach((row,i)=>editable.forEach(key=>{notes[i][key]=row[key];}));
      refresh();draw();status.textContent='Đã nhập nhãn. Chưa xác nhận việc người duyệt nhìn ảnh.';
    } catch(error) {status.textContent=error.message;}
  });
  refresh();draw();
})();</script></main></body></html>"""


def prepare_validation_review(*, annotations, annotations_sha256, predictions,
                              predictions_sha256, manifest, manifest_sha256,
                              image_root, output_dir, count=100, seed=42,
                              expected_count=1407):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review: {output}")
    for path, digest in ((annotations, annotations_sha256), (predictions, predictions_sha256),
                         (manifest, manifest_sha256)):
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Input SHA-256 mismatch: {path}")
    records = list(read_jsonl(annotations))
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} validation records")
    selected = select_review_records(records, count=count, seed=seed)
    saved, export = list(read_jsonl(predictions)), json.loads(Path(manifest).read_text("utf-8"))
    inputs = export.get("inputs", {})
    if (export.get("schema_version") != 1 or inputs.get("split") != "validation"
            or inputs.get("annotations_sha256") != annotations_sha256
            or inputs.get("experiment_id") != "M1_SFT" or not inputs.get("adapter")
            or not inputs.get("adapter_files") or inputs.get("limit") != 0
            or type(inputs.get("seed")) is not int or inputs["seed"] < 0
            or not isinstance(export.get("protocol"), dict) or not export["protocol"]
            or any(not isinstance(row, dict) for row in saved)):
        raise ValueError("Manifest must identify a complete M1 validation export")
    export_records = select_records(records, "validation", 0, inputs["seed"])
    selection_digest = hashlib.sha256(json.dumps(
        export_records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if inputs.get("selected_sha256") != selection_digest:
        raise ValueError("Export record selection changed")
    completed = validate_saved_predictions(saved, records, export)
    if len(completed) != len(records):
        raise ValueError("Validation prediction coverage is incomplete")
    indexed = {row["question_id"]: row for row in saved}
    root, images, notes = Path(image_root).resolve(), {}, []
    for row in selected:
        prediction = indexed[row["question_id"]]
        source = (root / row["image_path"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing image or path outside image root: {source}")
        image_digest = sha256_file(source)
        if image_digest != prediction["image_sha256"]:
            raise ValueError(f"Review image differs from inference: {row['question_id']}")
        destination = f"images/{row['image_id']}.jpg"
        # The destination name is data-derived; reject path separators before copying.
        if Path(destination).name != f"{row['image_id']}.jpg" or "\\" in row["image_id"]:
            raise ValueError("Unsafe image ID")
        images[destination] = (source, image_digest)
        notes.append({
            "schema": "m1_validation_review_v1", "question_id": row["question_id"],
            "image_id": row["image_id"], "image_path": destination,
            "image_sha256": image_digest, "question": row["question"],
            "prediction_answer": prediction["prediction_answer"],
            "prediction_explanation": prediction["prediction_explanation"],
            "reference_answer": row["answer"], "reference_explanations": row["explanations"],
            "predictions_sha256": predictions_sha256, **dict.fromkeys(REVIEW_FIELDS),
            "review_status": "pending",
        })
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    for name, (source, _) in images.items():
        shutil.copyfile(source, output / name)
    write_jsonl(output / "review_notes.jsonl", notes)
    (output / "index.html").write_text(render_review(notes), encoding="utf-8")
    write_json(output / "manifest.json", {
        "schema": "m1_validation_review_bundle_v1", "status": "pending_human_review",
        "selection": {"recipe": RECIPE, "seed": seed, "count": count,
                      "question_ids": [row["question_id"] for row in notes],
                      "image_ids": [row["image_id"] for row in notes]},
        "inputs": {"annotations_sha256": annotations_sha256,
                   "predictions_sha256": predictions_sha256, "manifest_sha256": manifest_sha256,
                   "adapter_files": inputs["adapter_files"]},
        "files": {name: sha256_file(output / name) for name in
                  ("review_notes.jsonl", "index.html", *images)},
    })
    return output / "index.html"
