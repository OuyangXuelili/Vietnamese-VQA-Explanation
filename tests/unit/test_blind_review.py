import json
import re
import shutil
import subprocess

import pytest
from PIL import Image

from vqa_explanation.data.io import read_jsonl, write_jsonl
from vqa_explanation.evaluation.blind_review import (
    cohen_kappa,
    freeze_sample,
    prepare_blind_review,
    render_blind_editor,
    select_sample,
    summarize_blind_review,
)
from vqa_explanation.evaluation.comparison import EXPERIMENT_IDS
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def blind_bundle(tmp_path):
    records, paths = [], {}
    for i in range(4):
        image = tmp_path / f"images/{i}.jpg"
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), color=(i * 30, 0, 0)).save(image)
        for j in range(2):
            records.append({"question_id": f"q{i}_{j}", "image_id": str(i),
                            "image_path": f"images/{i}.jpg", "question": "Animal?",
                            "answer": "cat", "explanations": ["Fur."], "split": "test"})
    annotation = tmp_path / "all.jsonl"
    write_jsonl(annotation, records)
    sample = tmp_path / "sample.json"
    freeze_sample(annotation, sample, count=3)
    for role in EXPERIMENT_IDS:
        fields = ("question_id", "image_id", "image_path", "question", "split")
        rows = [{**{k: r[k] for k in fields},
                 "reference_answer": r["answer"], "reference_explanations": r["explanations"],
                 "prediction_answer": "cat", "prediction_explanation": "Fur.",
                 "image_sha256": sha256_file(tmp_path / r["image_path"]),
                 "model_id": "hidden-backbone", "model_revision": "hidden-revision",
                 "experiment_id": role, "adapter": None if role == EXPERIMENT_IDS[0] else role,
                 "seed": 42, "max_pixels": 200704, "min_pixels": 200704,
                 "max_new_tokens": 256, "prompt_sha256": "a" * 64,
                 "generation_config": {"do_sample": False}, "load_in_4bit": True,
                 "compute_dtype": "float16"} for r in records]
        paths[role] = tmp_path / f"{role}.jsonl"
        write_jsonl(paths[role], rows)
    bundle = tmp_path / "bundle"
    prepare_blind_review(annotation, sample, paths, tmp_path, bundle)
    return bundle


def ratings(bundle):
    key = {r["entry_id"]: r for r in read_jsonl(bundle / "private/model_key.jsonl")}
    files = {}
    for rater in ("rater1", "rater2"):
        rows = list(read_jsonl(bundle / rater / "blank_ratings.jsonl"))
        for row in rows:
            score = int(key[row["entry_id"]]["question_id"][1]) % 3
            row.update(review_status="reviewed", reviewer=rater, explanation_support=score,
                       answer_assessment="correct" if score == 2 else "uncertain",
                       unsupported_detail="no" if score == 2 else "yes",
                       reviewed_at="2026-10-11T12:00:00+07:00", image_evidence="Visible fur.")
        files[rater] = bundle.parent / f"{rater}_ratings.jsonl"
        write_jsonl(files[rater], rows)
    return files


def test_sample_is_prediction_independent_and_one_question_per_image():
    rows = [{"question_id": f"q{i}_{j}", "image_id": str(i), "split": "test"}
            for i in range(8) for j in range(3)]
    first = select_sample(rows, count=4)
    assert first == select_sample(list(reversed(rows)), count=4)
    assert len({r["image_id"] for r in first}) == 4


def test_pages_hide_model_keys_and_leave_labels_pending(blind_bundle):
    for rater in ("rater1", "rater2"):
        page = (blind_bundle / rater / "index.html").read_text("utf-8")
        assert "hidden-backbone" not in page and "hidden-revision" not in page
        assert all(role not in page for role in EXPERIMENT_IDS)
        rows = list(read_jsonl(blind_bundle / rater / "blank_ratings.jsonl"))
        assert len(rows) == 12 and all(r["review_status"] == "pending" for r in rows)
    assert len(list(read_jsonl(blind_bundle / "private/model_key.jsonl"))) == 24


def test_complete_independent_ratings_get_scores_and_agreement(blind_bundle):
    report = summarize_blind_review(blind_bundle, ratings(blind_bundle))
    assert report["status"] == "complete"
    assert report["raters"]["rater1"]["M3_DPO"]["n"] == 3
    assert report["agreement"]["M3_DPO"]["explanation_support"] == pytest.approx(1)
    one = summarize_blind_review(blind_bundle, {"rater1": ratings(blind_bundle)["rater1"]})
    assert one["agreement"] is None


def test_paired_secondary_metrics_do_not_replace_primary_support(blind_bundle):
    files = ratings(blind_bundle)
    key = {r["entry_id"]: r for r in read_jsonl(blind_bundle / "private/model_key.jsonl")}
    rows = list(read_jsonl(files["rater1"]))
    for row in rows:
        role = key[row["entry_id"]]["experiment_id"]
        if role in {"M2_SFT_CHOSEN", "M3_DPO"}:
            row.update(explanation_support=2,
                       answer_assessment="correct" if role == "M3_DPO" else "incorrect",
                       unsupported_detail="no" if role == "M3_DPO" else "yes")
    write_jsonl(files["rater1"], rows)
    report = summarize_blind_review(blind_bundle, {"rater1": files["rater1"]})
    rater = report["raters"]["rater1"]
    scores = rater["M3_minus_M2"]["metrics"]
    for name, delta in {"support": 0, "correct_and_support2_rate": 1,
                        "unsupported_yes_rate": -1, "unsupported_uncertain_rate": 0}.items():
        assert scores[name]["delta_right_minus_left"] == delta
        assert scores[name]["ci95"] == [delta, delta]
    assert rater["M2_SFT_CHOSEN"]["support_given_correct"] is None
    assert rater["M2_SFT_CHOSEN"]["correct_and_support2_rate"] == 0
    assert rater["M3_DPO"]["correct_and_support2_rate"] == 1
    assert report["metric_protocol"]["primary"] == "support"


def test_uncertain_detail_labels_remain_in_denominator(blind_bundle):
    files = ratings(blind_bundle)
    key = {r["entry_id"]: r for r in read_jsonl(blind_bundle / "private/model_key.jsonl")}
    rows = list(read_jsonl(files["rater1"]))
    for row in rows:
        role = key[row["entry_id"]]["experiment_id"]
        if role == "M2_SFT_CHOSEN":
            row["unsupported_detail"] = "yes"
        elif role == "M3_DPO":
            row["unsupported_detail"] = "uncertain"
    write_jsonl(files["rater1"], rows)
    report = summarize_blind_review(blind_bundle, {"rater1": files["rater1"]})
    rater = report["raters"]["rater1"]
    assert rater["M3_DPO"]["n"] == 3
    assert rater["M3_DPO"]["unsupported_counts"] == {"uncertain": 3}
    assert rater["M3_DPO"]["unsupported_yes_rate"] == 0
    assert rater["M3_DPO"]["unsupported_uncertain_rate"] == 1
    scores = rater["M3_minus_M2"]["metrics"]
    assert scores["unsupported_yes_rate"]["delta_right_minus_left"] == -1
    assert scores["unsupported_uncertain_rate"]["delta_right_minus_left"] == 1
    assert rater["M3_minus_M2"]["n_questions"] == 3


def test_paired_scores_are_invariant_to_rating_order(blind_bundle):
    files = ratings(blind_bundle)
    before = summarize_blind_review(blind_bundle, files)
    for path in files.values():
        write_jsonl(path, reversed(list(read_jsonl(path))))
    after = summarize_blind_review(blind_bundle, files)
    for rater in files:
        assert before["raters"][rater]["M3_minus_M2"] == after["raters"][rater]["M3_minus_M2"]


@pytest.mark.parametrize("fault", ["pending", "duplicate", "immutable", "same_reviewer"])
def test_invalid_labels_cannot_be_scored(blind_bundle, fault):
    files = ratings(blind_bundle)
    rows = list(read_jsonl(files["rater2"]))
    if fault == "pending":
        rows[0]["review_status"] = "pending"
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "immutable":
        rows[0]["prediction_explanation"] = "Different output"
    else:
        for row in rows:
            row["reviewer"] = "rater1"
    write_jsonl(files["rater2"], rows)
    with pytest.raises(ValueError):
        summarize_blind_review(blind_bundle, files)


def test_bundle_changes_are_rejected(blind_bundle):
    files = ratings(blind_bundle)
    path = blind_bundle / "private/model_key.jsonl"
    path.write_text("{}\n", "utf-8")
    with pytest.raises(ValueError, match="bundle changed"):
        summarize_blind_review(blind_bundle, files)


def test_kappa_handles_order_and_degenerate_ratings():
    assert cohen_kappa([0, 1, 2], [0, 1, 2], [0, 1, 2], quadratic=True) == 1
    assert cohen_kappa([0, 0], [0, 0], [0, 1, 2], quadratic=True) is None
    assert cohen_kappa([0, 1], [1, 0], [0, 1]) == -1


def run_editor_regression(page, checks):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the blind-editor lifecycle checks")
    payload = json.loads(page.split('id="data">')[1].split("</script>")[0])
    code = page.split("<script>(()=>{")[1].split("</script>")[0]
    code = "(()=>{" + code.replace(
        "restoreDraft();refresh();draw();})();",
        "restoreDraft();refresh();draw();"
        "globalThis.reviewTest={rows,storageKey,validate,saveDraft};})();",
    )
    script = """
const assert=require('node:assert/strict'),vm=require('node:vm');
const source=SOURCE;
const code=CODE;
function mount(store=new Map(),blocked=false){
  class Element{
    constructor(){this.children=[];this.listeners={};this.value='';}
    replaceChildren(){this.children=[];}
    append(child){this.children.push(child);}
    addEventListener(name,fn){this.listeners[name]=fn;}
    click(){this.clicked=true;}
  }
  const elements={},events={};
  const document={
    getElementById(id){return elements[id]??=new Element();},
    createElement(){return new Element();}
  };
  document.getElementById('data').textContent=JSON.stringify(source);
  let exported;
  const scope={document,window:{addEventListener(name,fn){events[name]=fn;}},
    localStorage:{getItem(key){return store.get(key)??null;},
      setItem(key,value){if(blocked)throw Error('quota');store.set(key,value);}},
    Blob,URL:{createObjectURL(blob){exported=blob;return 'blob:fixture';},
      revokeObjectURL(){}},setTimeout(fn){fn();}};
  vm.runInNewContext(code,scope);
  return {scope,elements,events,store,exported:()=>exported};
}
(async()=>{CHECKS})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("SOURCE", json.dumps(payload)).replace("CODE", json.dumps(code)).replace(
        "CHECKS", checks,
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True,
                            check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_blind_draft_survives_reload_before_field_blur(blind_bundle):
    page = (blind_bundle / "rater1/index.html").read_text("utf-8")
    run_editor_regression(page, """
const first=mount();
const field=first.elements.form.children[5].children[0];
field.value='Fixture evidence still being typed';field.listeners.input();
const key=first.scope.reviewTest.storageKey;
assert.equal(JSON.parse(first.store.get(key))[0].image_evidence,field.value);
const second=mount(first.store);
assert.equal(second.scope.reviewTest.rows[0].image_evidence,field.value);
assert.equal(second.scope.reviewTest.rows[0].review_status,'pending');
assert.equal(second.scope.reviewTest.rows[0].reviewer,null);
assert.ok(second.elements.status.textContent.includes('khôi phục'));
let warned=false;
second.events.beforeunload({preventDefault(){warned=true;}});
assert.equal(warned,true);
""")


@pytest.mark.parametrize("fault", ["source", "support", "json"])
def test_blind_editor_rejects_bad_drafts_without_deleting_them(blind_bundle, fault):
    page = (blind_bundle / "rater1/index.html").read_text("utf-8")
    changes = {
        "source": "rows[0].prediction_explanation='forged source';",
        "support": "rows[0].explanation_support=77;",
        "json": "raw='not JSON';",
    }
    run_editor_regression(page, """
const first=mount(),key=first.scope.reviewTest.storageKey;
const rows=JSON.parse(JSON.stringify(source));let raw;
CHANGES
raw=raw??JSON.stringify(rows);const store=new Map([[key,raw]]);
const second=mount(store);
assert.equal(second.scope.reviewTest.rows[0].image_evidence,null);
assert.equal(second.scope.reviewTest.rows[0].prediction_explanation,
             source[0].prediction_explanation);
assert.equal(store.get(key),raw);
assert.ok(second.elements.status.textContent.includes('không khôi phục'));
""".replace("CHANGES", changes[fault]))


def test_blind_editor_warns_on_storage_failure_and_exports_exact_jsonl(blind_bundle):
    page = (blind_bundle / "rater1/index.html").read_text("utf-8")
    run_editor_regression(page, r"""
const first=mount(new Map(),true);
const field=first.elements.form.children[5].children[0];
field.value='Fixture evidence';field.listeners.input();
assert.ok(first.elements.status.textContent.includes('Không lưu được nháp'));
first.elements.export.listeners.click();
const exported=await first.exported().text();
const rows=exported.trim().split(/\r?\n/).map(JSON.parse);
assert.equal(rows.length,source.length);
assert.equal(rows[0].image_evidence,field.value);
assert.equal(rows[0].entry_id,source[0].entry_id);
let warned=false;first.events.beforeunload({preventDefault(){warned=true;}});
assert.equal(warned,false);
""")


def test_blind_editor_restores_incomplete_work_but_does_not_export_it(blind_bundle):
    page = (blind_bundle / "rater1/index.html").read_text("utf-8")
    run_editor_regression(page, """
const first=mount();
const field=first.elements.form.children[0].children[0];
field.value='reviewed';field.listeners.input();
const second=mount(first.store);
assert.equal(second.scope.reviewTest.rows[0].review_status,'reviewed');
assert.equal(second.scope.reviewTest.rows[0].reviewer,null);
second.elements.export.listeners.click();
assert.equal(second.exported(),undefined);
assert.ok(second.elements.status.textContent.includes('Còn thiếu đánh giá'));
""")


def test_blind_draft_identity_is_separate_for_each_rater_and_source(blind_bundle):
    first = (blind_bundle / "rater1/index.html").read_text("utf-8")
    second = (blind_bundle / "rater2/index.html").read_text("utf-8")
    notes = list(read_jsonl(blind_bundle / "rater1/blank_ratings.jsonl"))
    notes[0]["prediction_explanation"] = "Changed fixture output"
    third = render_blind_editor(notes)

    def key(page):
        return re.search(r"const storageKey='([^']+)'", page).group(1)

    assert len({key(first), key(second), key(third)}) == 3
