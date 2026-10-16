import copy
import hashlib
import json
import shutil
import subprocess

import pytest
from PIL import Image

from vqa_explanation.data.io import write_jsonl
from vqa_explanation.preference.audit import audit_review_notes
from vqa_explanation.preference.review import blank_notes, convert_notes, render_editor
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def review_case(tmp_path):
    image = tmp_path / "images/one.jpg"
    image.parent.mkdir()
    Image.new("RGB", (8, 8)).save(image)
    row = dict(
        question_id="q1",
        image_id="1",
        image_path="images/one.jpg",
        question="Animal?",
        answer="cat",
        explanations=["Fur."],
        split="train",
    )
    predictions = {"q1": dict(prediction_answer="cat", prediction_explanation="Wings.")}
    manifest = {
        "inputs": {
            "model_id": "model",
            "seed": 42,
            "adapter_files": {"adapter_model.safetensors": "c" * 64},
            "image_hashes": {"q1": sha256_file(image)},
        },
        "protocol": {},
    }
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(annotations, [row])
    manifest["inputs"]["context"] = {"annotations_sha256": sha256_file(annotations)}
    notes = blank_notes([row], predictions, "b" * 64)
    return notes, [row], predictions, manifest, "b" * 64, annotations, tmp_path


def accepted(notes):
    notes = copy.deepcopy(notes)
    notes[0].update(
        review_status="accepted",
        reviewer="test fixture",
        reviewed_at="2026-10-06T10:00:00+07:00",
        answer_assessment="correct",
        explanation_support=0,
        canonical_answer_verified=True,
        unsupported_span="Wings",
        image_evidence="Fur is visible.",
        proposed_chosen_explanation="Fur.",
        proposed_rejected_explanation="Wings.",
        error_type="attribute",
    )
    return notes


def test_pending_labels_cannot_be_exported(review_case):
    with pytest.raises(ValueError, match="No accepted"):
        convert_notes(*review_case)


def test_verified_notes_produce_identical_answer_and_explicit_origin(review_case):
    notes, *other = review_case
    pairs = convert_notes(accepted(notes), *other)
    assert len(pairs) == 1 and pairs[0]["review_status"] == "accepted"
    assert pairs[0]["provenance"]["rejected"]["kind"] == "model"
    assert pairs[0]["provenance"]["chosen"]["kind"] == "manual"
    assert pairs[0]["chosen"].splitlines()[0] == pairs[0]["rejected"].splitlines()[0]


def ai_accepted(notes):
    rows = accepted(notes)
    rows[0].update(
        reviewer="codex-visual-review", reviewer_type="ai",
        chosen_origin_kind="synthetic", chosen_origin_source="recipe.json:q1",
        chosen_recipe_sha256="d" * 64,
    )
    return rows


def test_ai_review_does_not_claim_human_verification(review_case):
    notes, *other = review_case
    pair = convert_notes(ai_accepted(notes), *other)[0]
    assert pair["generation_method"] == "ai_review_candidate"
    assert pair["reviewer_type"] == "ai" and pair["human_verified"] is False
    assert pair["provenance"]["chosen"]["kind"] == "synthetic"
    assert pair["provenance"]["rejected"]["kind"] == "model"
    assert pair["provenance"]["original_candidate_answer"] == "cat"


@pytest.mark.parametrize("kind", [None, "manual"])
def test_ai_written_chosen_cannot_claim_human_authorship(review_case, kind):
    notes, *other = review_case
    rows = ai_accepted(notes)
    rows[0].update(chosen_origin_kind=kind, chosen_recipe_sha256=None)
    with pytest.raises(ValueError, match="AI-written chosen"):
        convert_notes(rows, *other)


def test_ai_edited_negative_requires_synthetic_origin(review_case):
    notes, *other = review_case
    rows = ai_accepted(notes)
    rows[0].update(proposed_rejected_explanation="Large wings.", rejected_origin_kind="manual")
    with pytest.raises(ValueError, match="AI-written rejected"):
        convert_notes(rows, *other)


@pytest.mark.parametrize("actor", ["anonymous", True, [], {}])
def test_invalid_review_actor_is_rejected_by_audit_and_export(review_case, actor):
    notes, *other = review_case
    rows = ai_accepted(notes)
    rows[0]["reviewer_type"] = actor
    audit = audit_review_notes(rows, notes)
    assert audit["status"] == "failed"
    with pytest.raises(ValueError, match="reviewer_type"):
        convert_notes(rows, *other)


def test_edited_negative_is_not_misrepresented_as_raw_model_output(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0]["proposed_rejected_explanation"] = "Large wings."
    pairs = convert_notes(notes, *other)
    assert pairs[0]["provenance"]["rejected"]["kind"] == "manual"
    assert pairs[0]["provenance"]["rejected_explanation_edited"] is True


@pytest.mark.parametrize("edited", [False, True])
def test_existing_notes_without_optional_origin_fields_remain_supported(review_case, edited):
    notes, *other = review_case
    notes = accepted(notes)
    for side in ("chosen", "rejected"):
        for suffix in ("origin_kind", "origin_source", "recipe_sha256"):
            notes[0].pop(f"{side}_{suffix}")
    if edited:
        notes[0]["proposed_rejected_explanation"] = "Large wings."
    pairs = convert_notes(notes, *other)
    assert pairs[0]["provenance"]["rejected"]["kind"] == ("manual" if edited else "model")
    assert pairs[0]["provenance"]["chosen"]["kind"] == "manual"


def test_synthetic_negative_preserves_its_recipe_and_draft_source(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        proposed_rejected_explanation="Large wings.",
        rejected_origin_kind="synthetic",
        rejected_origin_source="  synthetic_calibration_v1/drafts.jsonl:q1  ",
        rejected_recipe_sha256="d" * 64,
    )
    pairs = convert_notes(notes, *other)
    assert pairs[0]["provenance"]["rejected"] == {
        "kind": "synthetic",
        "source": "synthetic_calibration_v1/drafts.jsonl:q1",
        "recipe_sha256": "d" * 64,
    }
    assert pairs[0]["provenance"]["rejected_explanation_edited"] is True
    assert pairs[0]["provenance"]["original_candidate_answer"] == "cat"
    assert pairs[0]["chosen"].splitlines()[0] == pairs[0]["rejected"].splitlines()[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("rejected_origin_source", None),
        ("rejected_origin_source", "  "),
        ("rejected_recipe_sha256", None),
        ("rejected_recipe_sha256", "d" * 63),
        ("rejected_recipe_sha256", "D" * 64),
        ("rejected_recipe_sha256", 123),
    ],
)
def test_synthetic_negative_requires_source_and_valid_recipe(review_case, field, value):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        proposed_rejected_explanation="Large wings.",
        rejected_origin_kind="synthetic",
        rejected_origin_source="drafts.jsonl:q1",
        rejected_recipe_sha256="d" * 64,
    )
    notes[0][field] = value
    with pytest.raises(ValueError, match="synthetic"):
        convert_notes(notes, *other)


def test_raw_m1_negative_cannot_be_claimed_as_synthetic(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        rejected_origin_kind="synthetic",
        rejected_origin_source="drafts.jsonl:q1",
        rejected_recipe_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="unchanged rejected"):
        convert_notes(notes, *other)


def test_edited_negative_cannot_claim_raw_m1_origin(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(proposed_rejected_explanation="Large wings.", rejected_origin_kind="model")
    with pytest.raises(ValueError, match="edited rejected"):
        convert_notes(notes, *other)


@pytest.mark.parametrize("kind", ["reference", "ai", [], 123])
def test_rejected_origin_must_be_supported(review_case, kind):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(proposed_rejected_explanation="Large wings.", rejected_origin_kind=kind)
    with pytest.raises(ValueError, match="rejected_origin_kind"):
        convert_notes(notes, *other)


def test_recipe_cannot_be_silently_dropped_from_manual_negative(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        proposed_rejected_explanation="Large wings.",
        rejected_origin_kind="manual",
        rejected_recipe_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="requires synthetic origin"):
        convert_notes(notes, *other)


def test_human_verification_preserves_synthetic_authorship_on_both_sides(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        proposed_rejected_explanation="Large wings.",
        chosen_origin_kind="synthetic",
        chosen_origin_source="  drafts.jsonl:q1:chosen  ",
        chosen_recipe_sha256="d" * 64,
        rejected_origin_kind="synthetic",
        rejected_origin_source="drafts.jsonl:q1:rejected",
        rejected_recipe_sha256="d" * 64,
    )
    pairs = convert_notes(notes, *other)
    assert pairs[0]["reviewer"] == "test fixture"
    assert pairs[0]["provenance"]["chosen"] == {
        "kind": "synthetic",
        "source": "drafts.jsonl:q1:chosen",
        "recipe_sha256": "d" * 64,
    }
    assert pairs[0]["provenance"]["rejected"] == {
        "kind": "synthetic",
        "source": "drafts.jsonl:q1:rejected",
        "recipe_sha256": "d" * 64,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("chosen_origin_source", None),
        ("chosen_origin_source", "  "),
        ("chosen_recipe_sha256", None),
        ("chosen_recipe_sha256", "d" * 63),
        ("chosen_recipe_sha256", "D" * 64),
        ("chosen_recipe_sha256", 123),
    ],
)
def test_synthetic_chosen_requires_source_and_valid_recipe(review_case, field, value):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(
        chosen_origin_kind="synthetic",
        chosen_origin_source="drafts.jsonl:q1:chosen",
        chosen_recipe_sha256="d" * 64,
    )
    notes[0][field] = value
    with pytest.raises(ValueError, match="synthetic chosen"):
        convert_notes(notes, *other)


@pytest.mark.parametrize("kind", ["reference", "model", "ai", [], 123])
def test_chosen_origin_must_be_supported(review_case, kind):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0]["chosen_origin_kind"] = kind
    with pytest.raises(ValueError, match="chosen_origin_kind"):
        convert_notes(notes, *other)


def test_recipe_cannot_be_silently_dropped_from_manual_chosen(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0].update(chosen_origin_kind="manual", chosen_recipe_sha256="d" * 64)
    with pytest.raises(ValueError, match="requires synthetic origin"):
        convert_notes(notes, *other)


def test_editor_exposes_synthetic_origin_inputs_for_both_sides(review_case):
    page = render_editor(review_case[0])
    for side in ("chosen", "rejected"):
        for suffix in ("origin_kind", "origin_source", "recipe_sha256"):
            assert f"{side}_{suffix}" in page


def test_editor_groups_fields_and_collapses_provenance(review_case):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the editor layout regression test")
    page = render_editor(review_case[0])
    layout = page[page.index("  const reviewSections = ["):
                  page.index("  function validateRows(")]
    script = """
const assert=require('node:assert/strict');
function element(tag) {
  return {tag,children:[],open:false,append(...items){this.children.push(...items);}};
}
const document={createElement:element}, form=element('form');
""" + layout + """
const containers=createReviewSections();
assert.equal(form.children.length,4);
assert.deepEqual(form.children.slice(0,3).map(item=>item.tag),
                 ['fieldset','fieldset','fieldset']);
const details=form.children[3];
assert.equal(details.tag,'details');
assert.equal(details.id,'review-provenance');
assert.equal(details.open,false);
assert.equal(details.children[0].tag,'summary');
assert.equal(containers.size,18);
const core=reviewSections.flatMap(section=>section.fields);
assert.equal(new Set([...core,...technicalFields]).size,18);
for (const key of technicalFields) assert.equal(containers.get(key),details.children[2]);
for (const key of core) assert.notEqual(containers.get(key),details.children[2]);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    assert "containers.get(key).append(wrapper)" in page
    assert "optionLabels[key]?.[value]" in page
    assert "item.value=value" in page


def test_editor_layout_keeps_storage_key_and_pending_labels(review_case):
    notes = review_case[0]
    before = copy.deepcopy(notes)
    page = render_editor(notes)
    payload = page.split('id="review-data">')[1].split("</script>")[0]
    queue = page.split('id="queue-data">')[1].split("</script>")[0]
    expected = hashlib.sha256((payload + "\n" + queue + "\nvisible").encode()).hexdigest()
    assert f"const storageKey='vqa-review-draft-v1:{expected}';" in page
    assert json.loads(payload)[0]["review_status"] == "pending"
    assert json.loads(payload)[0]["reviewer"] is None
    assert notes == before


def test_editor_picker_selects_first_question_on_first_load(review_case):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the editor picker regression test")
    page = render_editor(review_case[0])
    helper = page[page.index("  function refreshPicker()"):
                  page.index("  function updateProgress()")]
    script = """
const assert=require('node:assert/strict');
const queueIndices=[0,1], notes=[{source_question_id:'q1',review_status:'pending'},
                               {source_question_id:'q2',review_status:'pending'}];
const picker={value:'',replaceChildren(){},append(){}};
const document={createElement(){return {};}};
function updateProgress(){}
""" + helper + """
refreshPicker(); assert.equal(picker.value,'0');
picker.value='1'; refreshPicker(); assert.equal(picker.value,'1');
picker.value='999'; refreshPicker(); assert.equal(picker.value,'0');
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_editor_normalizes_legacy_optional_fields_without_mutating_notes(review_case):
    notes = copy.deepcopy(review_case[0])
    for side in ("chosen", "rejected"):
        for suffix in ("origin_kind", "origin_source", "recipe_sha256"):
            notes[0].pop(f"{side}_{suffix}")
    before = copy.deepcopy(notes)
    page = render_editor(notes)
    payload = json.loads(page.split('id="review-data">')[1].split("</script>")[0])
    for side in ("chosen", "rejected"):
        for suffix in ("origin_kind", "origin_source", "recipe_sha256"):
            assert f"{side}_{suffix}" in payload[0]
            assert payload[0][f"{side}_{suffix}"] is None
    assert notes == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_explanation", "changed"),
        ("bundle_sha256", "f" * 64),
        ("canonical_answer_verified", False),
        ("answer_assessment", "incorrect"),
        ("unsupported_span", "absent"),
        ("image_evidence", ""),
        ("reviewer", None),
        ("reviewed_at", "2026-10-06T10:00:00"),
        ("error_type", "unknown"),
    ],
)
def test_invalid_or_forged_notes_are_rejected(review_case, field, value):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0][field] = value
    with pytest.raises(ValueError):
        convert_notes(notes, *other)


def test_editor_data_cannot_close_the_json_script(review_case):
    notes = review_case[0]
    notes[0]["candidate_explanation"] = "</script><script>alert(1)</script>"
    page = render_editor(notes)
    assert "</script><script>alert(1)" not in page
    assert "\\u003c/script>" in page
    assert "export-notes" in page and "import-notes" in page
    assert ".textContent=" in page


def test_editor_keeps_a_source_bound_browser_draft(review_case):
    page = render_editor(review_case[0])
    assert "localStorage.setItem(storageKey" in page
    assert "localStorage.getItem(storageKey)" in page
    assert "vqa-review-draft-v1:" in page
    assert "validateRows(rows)" in page
    assert "Trang tự lưu bản nháp" in page

    changed = copy.deepcopy(review_case[0])
    changed[0]["reviewer"] = "another baseline"
    other_page = render_editor(changed)
    key = page.split("const storageKey='")[1].split("';")[0]
    other_key = other_page.split("const storageKey='")[1].split("';")[0]
    assert key != other_key


def test_editor_reconciles_restored_picker_without_erasing_current_form(review_case):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the editor lifecycle regression test")
    page = render_editor(review_case[0])
    helper = page[page.index("  function syncSelection()"):
                  page.index("  picker.addEventListener('input',syncSelection)")]
    script = """
const assert=require('node:assert/strict');
const notes=[{source_question_id:'q1'},{source_question_id:'q2'}];
const queueIndices=[0,1], picker={value:'1'}, form={dataset:{questionId:'q1'}};
let redraws=0;
function draw(){redraws++; form.dataset.questionId=notes[Number(picker.value)].source_question_id;}
""" + helper + """
syncSelection();
assert.equal(form.dataset.questionId,'q2');
assert.equal(redraws,1);
syncSelection();
assert.equal(redraws,1);
picker.value='999'; syncSelection();
assert.equal(picker.value,'0');
assert.equal(form.dataset.questionId,'q1');
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    assert "window.addEventListener('pageshow',syncSelection)" in page
    assert "picker.addEventListener('input',syncSelection)" in page
    assert "picker.addEventListener('change',syncSelection)" in page
    assert "input.addEventListener('input', () => {\n        picker.value=String(i);" in page


def test_editor_shows_ai_advisory_without_inventing_a_review(review_case):
    notes = review_case[0]
    advisory = {
        "source_question_id": "q1",
        "ai_verdict": "repair_explanation",
        "rationale": "The visible fur contradicts the candidate.",
        "image_evidence_ai": "Fur is visible.",
    }
    page = render_editor(notes, advisories=[advisory])
    payload = page.split('id="advisory-data">')[1].split("</script>")[0]

    assert json.loads(payload) == [advisory]
    assert 'id="next-ai-flagged"' in page
    assert "AI triage — không phải nhãn người" in page
    assert notes[0]["review_status"] == "pending"
    assert notes[0]["reviewer"] is None


def test_editor_can_hide_ai_help_until_the_reviewer_requests_it(review_case):
    advisory = {
        "source_question_id": "q1",
        "ai_verdict": "repair_explanation",
        "rationale": "The visible fur contradicts the candidate.",
    }
    page = render_editor(
        review_case[0],
        advisories=[advisory],
        assistance_mode="on_request",
    )

    assert "const assistanceMode='on_request'" in page
    assert 'id="reveal-assistance"' in page
    assert "revealedAssistance.has(row.source_question_id)" in page
    assert "nextDraft.hidden=assistanceMode==='on_request'" in page
    assert review_case[0][0]["review_status"] == "pending"


def test_editor_rejects_unknown_assistance_mode(review_case):
    with pytest.raises(ValueError, match="assistance_mode"):
        render_editor(review_case[0], assistance_mode="always_apply")


@pytest.mark.parametrize(
    "advisories",
    [
        [
            {"source_question_id": "q1", "ai_verdict": "pass_grounded"},
            {"source_question_id": "q1", "ai_verdict": "exclude_ambiguous"},
        ],
        [{"source_question_id": "unknown", "ai_verdict": "pass_grounded"}],
        [{"source_question_id": "q1", "ai_verdict": "accepted"}],
    ],
)
def test_editor_rejects_invalid_ai_advisories(review_case, advisories):
    with pytest.raises(ValueError, match="advisories"):
        render_editor(review_case[0], advisories=advisories)


def test_blank_notes_do_not_invent_labels(review_case):
    notes = review_case[0]
    assert notes[0]["review_status"] == "pending"
    assert notes[0]["reviewer"] is None
    assert notes[0]["canonical_answer_verified"] is None
    assert notes[0]["proposed_rejected_explanation"] is None


def test_accepted_review_requires_a_real_support_rating(review_case):
    notes, *other = review_case
    notes = accepted(notes)
    notes[0]["explanation_support"] = None
    with pytest.raises(ValueError, match="explanation_support"):
        convert_notes(notes, *other)


def test_candidate_cannot_be_rebound_to_a_different_canonical_version(review_case):
    notes, *other = review_case
    other[2]["inputs"]["context"]["annotations_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="annotation version"):
        convert_notes(accepted(notes), *other)


def test_json_data_roundtrip_is_exact(review_case):
    notes = review_case[0]
    page = render_editor(notes)
    payload = page.split('id="review-data">')[1].split("</script>")[0]
    assert json.loads(payload) == notes
    assert hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize(
    "kind,edited,valid",
    [
        (None, False, True),
        ("", False, True),
        ("model", False, True),
        ("manual", False, False),
        ("model", True, False),
        ("manual", True, True),
        ([], True, False),
    ],
)
def test_audit_and_export_agree_on_rejected_authorship(review_case, kind, edited, valid):
    baseline, *other = review_case
    notes = accepted(baseline)
    notes[0]["rejected_origin_kind"] = kind
    if edited:
        notes[0]["proposed_rejected_explanation"] = "Large wings."
    report = audit_review_notes(notes, baseline)
    assert (report["status"] == "passed") is valid
    if valid:
        assert len(convert_notes(notes, *other)) == 1
    else:
        with pytest.raises(ValueError):
            convert_notes(notes, *other)


def run_draft_restore(page, storage):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the browser-draft regression test")
    payload = json.loads(page.split('id="review-data">')[1].split("</script>")[0])
    key = page.split("const storageKey='")[1].split("';")[0]
    legacy = json.loads(page.split("const legacyStorageKeys=")[1].split(";")[0])
    helpers = page[page.index("  function validateRows(") : page.index("  function refreshPicker(")]
    script = (
        """
const fs=require('node:fs');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const sourceNotes=input.notes, notes=sourceNotes.map(row=>({...row}));
const storageKey=input.key, legacyStorageKeys=input.legacy;
const immutable=['schema','source_question_id','image_path','bundle_sha256',
                 'reference_answer','candidate_answer','candidate_explanation'];
const state=input.storage;
const localStorage={getItem:key=>state[key] ?? null, setItem:(key,value)=>{state[key]=value;}};
const message={textContent:''}; let dirty=false,saveTimer=null;
"""
        + helpers
        + """
const restored=restoreBrowserDraft();
process.stdout.write(JSON.stringify({restored,notes,state,message:message.textContent}));
"""
    )
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps({"notes": payload, "key": key, "legacy": legacy, "storage": storage}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout), key, legacy


@pytest.mark.parametrize("mode", ["visible", "on_request"])
def test_legacy_browser_draft_is_migrated_without_deleting_original(review_case, mode):
    page = render_editor(review_case[0], assistance_mode=mode)
    legacy = json.loads(page.split("const legacyStorageKeys=")[1].split(";")[0])
    saved = accepted(review_case[0])
    result, key, _ = run_draft_restore(page, {legacy[0]: json.dumps(saved)})
    assert result["restored"] is True
    assert result["notes"] == saved
    assert json.loads(result["state"][key]) == saved
    assert json.loads(result["state"][legacy[0]]) == saved


def test_conflicting_legacy_drafts_are_preserved_for_explicit_import(review_case):
    page = render_editor(review_case[0])
    legacy = json.loads(page.split("const legacyStorageKeys=")[1].split(";")[0])
    storage = {
        legacy[0]: json.dumps(accepted(review_case[0])),
        legacy[1]: json.dumps(review_case[0]),
    }
    result, key, _ = run_draft_restore(page, storage)
    assert result["restored"] is False
    assert result["state"] == storage
    assert key not in result["state"]
    assert "khác nhau" in result["message"]


def test_invalid_browser_draft_is_not_erased_or_replaced(review_case):
    page = render_editor(review_case[0])
    key = page.split("const storageKey='")[1].split("';")[0]
    saved = accepted(review_case[0])
    saved[0]["bundle_sha256"] = "wrong-source"
    storage = {key: json.dumps(saved)}
    result, _, _ = run_draft_restore(page, storage)
    assert result["restored"] is False
    assert result["notes"] == review_case[0]
    assert result["state"] == storage


def test_current_draft_takes_priority_over_older_drafts(review_case):
    page = render_editor(review_case[0])
    key = page.split("const storageKey='")[1].split("';")[0]
    legacy = json.loads(page.split("const legacyStorageKeys=")[1].split(";")[0])
    saved = accepted(review_case[0])
    result, _, _ = run_draft_restore(
        page, {key: json.dumps(saved), legacy[0]: json.dumps(review_case[0])}
    )
    assert result["restored"] is True
    assert result["notes"] == saved
