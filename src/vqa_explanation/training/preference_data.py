"""CPU input gates shared by chosen-SFT and DPO."""

import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.inference.export import validate_saved_predictions
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.preference.freeze import prepare_exports
from vqa_explanation.preference.quality import assess_reviews
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.run_provenance import preference_training_fingerprints, sha256_file


def adapter_fingerprint(path):
    path = Path(path)
    names = ["adapter_config.json", "adapter_model.safetensors"]
    if any(not (path / name).is_file() or not (path / name).stat().st_size for name in names):
        raise ValueError("M1 adapter needs nonempty config and safetensors weights")
    files = {name: sha256_file(path / name) for name in names}
    return {
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = {
        "stage",
        "experiment_id",
        "seed",
        "expected_pairs",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "min_pixels",
        "max_pixels",
        "disable_dropout",
        "checkpoint_selection",
        "logging_steps",
        "save_total_limit",
        "save_steps",
        "dtype",
        "attn_implementation",
        "max_length",
    }
    extra = {"beta", "preflight_max_length", "precompute_ref_log_probs"}
    if not isinstance(cfg, dict) or cfg.get("stage") not in {"sft_chosen", "dpo"}:
        raise ValueError("Expected sft_chosen or dpo config")
    fields = common | (extra if cfg["stage"] == "dpo" else set())
    if set(cfg) != fields:
        raise ValueError(f"Config fields differ: {sorted(set(cfg) ^ fields)}")
    expected_id = "M2_SFT_CHOSEN" if cfg["stage"] == "sft_chosen" else "M3_DPO"
    if (
        cfg["experiment_id"] != expected_id
        or cfg["checkpoint_selection"] != "final_epoch"
        or cfg["per_device_train_batch_size"] != 1
        or cfg["gradient_accumulation_steps"] != 8
        or cfg["disable_dropout"] is not True
        or cfg["gradient_checkpointing"] is not True
    ):
        raise ValueError("Protocol requires batch 1 x 8 and dropout off")
    for key in (
        "expected_pairs",
        "seed",
        "min_pixels",
        "max_pixels",
        "logging_steps",
        "save_steps",
        "num_train_epochs",
    ):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    if cfg["save_total_limit"] is not None:
        raise ValueError("Keep preference checkpoints; save_total_limit must be null")
    if (
        type(cfg["learning_rate"]) not in (int, float)
        or not math.isfinite(cfg["learning_rate"])
        or cfg["learning_rate"] <= 0
    ):
        raise ValueError("Invalid learning_rate")
    if cfg["dtype"] not in {"auto", "float16", "bfloat16"}:
        raise ValueError("dtype must be auto, float16 or bfloat16")
    if cfg["attn_implementation"] != "sdpa":
        raise ValueError("This single-GPU protocol uses sdpa")
    length = cfg["max_length"] if cfg["stage"] == "sft_chosen" else cfg["preflight_max_length"]
    if type(length) is not int or length <= 0:
        raise ValueError("Invalid token length guard")
    if cfg["stage"] == "dpo":
        if cfg["max_length"] is not None or cfg["precompute_ref_log_probs"] is not False:
            raise ValueError("Do not truncate image tokens or use an unverified reference cache")
        if (
            type(cfg["beta"]) not in (int, float)
            or not math.isfinite(cfg["beta"])
            or cfg["beta"] <= 0
        ):
            raise ValueError("Invalid DPO beta")
    return cfg


def validate_study_arm(config_path, study_path):
    """Bind a run to both declared arms before reading data or loading a model."""
    study_path = Path(study_path)
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    if not isinstance(study, dict) or not isinstance(study.get("training"), dict):
        raise ValueError("Study needs a training comparison")
    if not isinstance(study.get("protocol_id"), str) or not study["protocol_id"].strip():
        raise ValueError("Study needs a protocol_id")
    training = study["training"]
    paths = {}
    for arm in ("m2", "m3"):
        relative = training.get(f"{arm}_config")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"Study needs {arm}_config")
        paths[arm] = (project_root() / relative).resolve()
    sft, dpo = (load_config(paths[arm]) for arm in ("m2", "m3"))
    if sft["stage"] != "sft_chosen" or dpo["stage"] != "dpo":
        raise ValueError("Study arms must be chosen-SFT and DPO, in that order")
    matched = (
        "seed", "expected_pairs", "learning_rate", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "gradient_checkpointing", "min_pixels", "max_pixels", "disable_dropout",
        "checkpoint_selection", "dtype", "attn_implementation", "save_steps",
    )
    for field in matched:
        if sft[field] != dpo[field]:
            raise ValueError(f"Comparison arms differ: {field}")
    if sft["max_length"] != dpo["preflight_max_length"]:
        raise ValueError("Comparison arms differ: token length guard")
    declared = {
        "seed": study.get("seed"),
        "num_train_epochs": training.get("epochs_per_branch"),
        "learning_rate": training.get("learning_rate"),
        "checkpoint_selection": training.get("checkpoint_selection"),
    }
    preferences = study.get("preferences", {})
    if preferences.get("data_locked") is True:
        locked_pairs = preferences.get("locked_pairs")
        if type(locked_pairs) is not int or locked_pairs <= 0:
            raise ValueError("Locked study requires a positive locked_pairs")
        declared["expected_pairs"] = locked_pairs
    for field, value in declared.items():
        if value is None or sft[field] != value:
            raise ValueError(f"Config differs from study: {field}")
    batch = sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
    if batch != training.get("effective_batch"):
        raise ValueError("Config differs from study: effective_batch")
    selected = load_config(config_path)
    arm = "m2" if selected["stage"] == "sft_chosen" else "m3"
    if selected != (sft if arm == "m2" else dpo):
        raise ValueError(f"Selected config is not the declared {arm} arm")
    return {
        "protocol_id": study["protocol_id"],
        "study_sha256": sha256_file(study_path),
        "arm": arm,
        "config_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "matched_fields": {field: sft[field] for field in matched},
        "token_length_guard": sft["max_length"],
        "effective_batch": batch,
    }


def _evidence_path(selection, base, name):
    evidence = selection.get("evidence", {}).get(name, {})
    relative = evidence.get("path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ":" in relative
        or "\\" in relative
    ):
        raise ValueError(f"Missing portable M1 evidence: {name}")
    path = (base / relative).resolve()
    if sha256_file(path) != evidence.get("sha256"):
        raise ValueError(f"M1 selection evidence changed: {name}")
    return path


def validate_selection(selection_path, parent, model_cfg, annotations_hash, canonical):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return _validate_selection(selection, selection_path.parent, parent, model_cfg,
                               annotations_hash, canonical)


def _validate_selection(selection, base, parent, model_cfg, annotations_hash, canonical):
    if (
        selection.get("status") != "selected"
        or selection.get("selection_split") != "validation"
        or selection.get("annotations_sha256") != annotations_hash
        or selection.get("adapter") != parent
        or selection.get("model_id") != model_cfg["id"]
        or selection.get("model_revision") != model_cfg["revision"]
    ):
        raise ValueError("M1 needs an explicit validation-only selection and matching adapter")
    paths = {
        name: _evidence_path(selection, base, name)
        for name in (
            "epoch_result",
            "trainer_state",
            "loss_selection",
            "validation_predictions",
            "validation_manifest",
            "validation_annotations",
        )
    }
    epoch = json.loads(paths["epoch_result"].read_text(encoding="utf-8"))
    state = json.loads(paths["trainer_state"].read_text(encoding="utf-8"))
    if (
        epoch.get("status") != "passed"
        or epoch.get("global_step") != 3683
        or state.get("global_step") != 3683
        or not math.isclose(state.get("epoch", -1), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("M1 one-epoch result has not passed")
    loss_selection = json.loads(paths["loss_selection"].read_text(encoding="utf-8"))
    if (
        loss_selection.get("status") != "passed"
        or loss_selection.get("selected_adapter_sha256")
        != parent["files"]["adapter_model.safetensors"]
    ):
        raise ValueError("Post-epoch loss selection differs from the selected adapter")
    records = [r for r in canonical if r["split"] == "validation"]
    val_annotations = list(read_jsonl(paths["validation_annotations"]))
    if sorted(val_annotations, key=lambda r: r["question_id"]) != sorted(
        records, key=lambda r: r["question_id"]
    ):
        raise ValueError("Validation annotations differ from canonical validation")
    predictions = list(read_jsonl(paths["validation_predictions"]))
    manifest = json.loads(paths["validation_manifest"].read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", {})
    if (
        inputs.get("adapter_files") != parent["files"]
        or inputs.get("model_id") != model_cfg["id"]
        or inputs.get("model_revision") != model_cfg["revision"]
        or inputs.get("annotations_sha256") != sha256_file(paths["validation_annotations"])
        or inputs.get("split") != "validation"
    ):
        raise ValueError("Validation evidence does not belong to the selected M1")
    completed = validate_saved_predictions(predictions, records, manifest)
    if len(records) != 1407 or len(completed) != len(records):
        raise ValueError("M1 selection requires all 1407 validation predictions")
    # Diagnostic labels are optional; supplied labels must still be verifiable.
    if "human_review" not in selection.get("evidence", {}):
        return selection
    human_path = _evidence_path(selection, base, "human_review")
    human = list(read_jsonl(human_path))
    by_id = {r["question_id"]: r for r in records}
    if (
        not human
        or len({r.get("question_id") for r in human}) != len(human)
        or any(
            r.get("question_id") not in by_id
            or r.get("review_status") != "reviewed"
            or not isinstance(r.get("reviewer"), str)
            or not r["reviewer"].strip()
            or type(r.get("explanation_support")) is not int
            or r["explanation_support"] not in {0, 1, 2}
            or r.get("answer_assessment") not in {"correct", "incorrect", "uncertain"}
            for r in human
        )
    ):
        raise ValueError("Supplied M1 diagnostic review has invalid or pending labels")
    if len({by_id[r["question_id"]]["image_id"] for r in human}) != len(human):
        raise ValueError("M1 diagnostic review must use distinct images")
    return selection


def freeze_selection(args):
    """Pin an M1 adapter to its completed training and validation evidence."""
    if sha256_file(args.annotations) != args.expected_sha256:
        raise ValueError("Canonical annotation hash changed")
    parent = adapter_fingerprint(args.adapter)
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))["model"]
    evidence_paths = {
        name: getattr(args, name)
        for name in (
            "epoch_result", "trainer_state", "loss_selection", "validation_predictions",
            "validation_manifest", "validation_annotations",
        )
    }
    if args.human_review is not None:
        evidence_paths["human_review"] = args.human_review
    selection = {
        "schema_version": 2,
        "protocol": "validation_loss_full_validation_v2",
        "status": "selected",
        "selection_split": "validation",
        "annotations_sha256": args.expected_sha256,
        "model_id": model_cfg["id"],
        "model_revision": model_cfg["revision"],
        "adapter": parent,
        "diagnostic_review_required": False,
        "evidence": {
            name: {
                "path": Path(os.path.relpath(
                    path.resolve(), args.output.parent.resolve()
                )).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in evidence_paths.items()
        },
    }
    canonical = list(read_jsonl(args.annotations))
    _validate_selection(selection, args.output.parent, parent, model_cfg,
                        args.expected_sha256, canonical)
    payload = (json.dumps(selection, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise ValueError("Existing M1 selection differs; use a new output path")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as destination:
            destination.write(payload)
    return selection


def prepare_stage(args):
    """Revalidate frozen data and identical chosen targets before loading a model."""
    study_path = getattr(args, "study_config", None)
    comparison = validate_study_arm(args.config, study_path) if study_path else None
    study = yaml.safe_load(Path(study_path).read_text(encoding="utf-8")) if study_path else {}
    quality_policy = study.get("preferences", {})
    if quality_policy.get("require_visual_quality") is True:
        if quality_policy.get("data_locked") is not True:
            raise ValueError("Study data is not locked; finish visual quality review first")
    env = load_environment(args.environment)
    cfg = load_config(args.config)
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))["model"]
    manifest_path = args.preferences / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated_metadata" or manifest.get("schema_version") != 1:
        raise ValueError("Preferences must be exported by the freeze validator")
    annotations_hash = sha256_file(env["annotations"])
    if manifest.get("annotations_sha256") != annotations_hash:
        raise ValueError("Preference/canonical annotation hash mismatch")
    for name in ("accepted.jsonl", "chosen_only.jsonl"):
        if sha256_file(args.preferences / name) != manifest.get("files", {}).get(name):
            raise ValueError(f"Frozen preference file changed: {name}")
    pairs = list(read_jsonl(args.preferences / "accepted.jsonl"))
    chosen = list(read_jsonl(args.preferences / "chosen_only.jsonl"))
    canonical = list(read_jsonl(env["annotations"]))
    accepted, expected_chosen, _ = prepare_exports(
        pairs, canonical, env["image_root"], annotations_hash
    )
    if accepted != pairs or expected_chosen != chosen:
        raise ValueError("M2 chosen rows must exactly match M3 accepted pairs")
    if len(pairs) != manifest.get("pairs") or len(pairs) != cfg["expected_pairs"]:
        raise ValueError("Pair count differs from the declared protocol; no silent subset")
    quality = None
    if quality_policy.get("require_visual_quality") is True:
        path = args.preferences / "quality_review.jsonl"
        digest = sha256_file(path)
        if digest != quality_policy.get("quality_review_sha256"):
            raise ValueError("Visual quality review hash differs from the locked study")
        quality = assess_reviews(
            pairs, list(read_jsonl(path)), quality_policy.get("minimum_pairs"),
            required_version=quality_policy.get("required_review_version", 1),
        )
        if not quality["ready"]:
            raise ValueError("Visual quality review is incomplete or the dataset is too small")
        quality = {"sha256": digest, **quality}
    parent = adapter_fingerprint(args.m1_adapter)
    validate_selection(args.m1_selection, parent, model_cfg, annotations_hash, canonical)
    validate_pair_origins(pairs, parent, model_cfg)
    parent_cfg = json.loads((args.m1_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if parent_cfg.get("base_model_name_or_path") != model_cfg["id"]:
        raise ValueError("M1 adapter backbone differs from model config")
    plan = {
        "schema_version": 1,
        "stage": cfg["stage"],
        "config": cfg,
        "model": model_cfg,
        "annotations_sha256": annotations_hash,
        "preferences_manifest_sha256": sha256_file(manifest_path),
        "preferences_files": manifest["files"],
        "pairs": len(pairs),
        "m1_adapter": parent,
        "m1_selection_sha256": sha256_file(args.m1_selection),
        "source_scope": "preference_training_import_graph_v1",
        "source_files": preference_training_fingerprints(project_root()),
        "expected_updates": (math.ceil(len(pairs) / cfg["gradient_accumulation_steps"])
                             * cfg["num_train_epochs"]),
    }
    if comparison is not None:
        plan["comparison_protocol"] = comparison
    if quality is not None:
        plan["visual_quality"] = quality
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return env, cfg, plan, pairs, chosen


def validate_pair_origins(pairs, parent, model_cfg):
    """Keep candidate lineage even when a reviewer rewrites both completions."""
    expected = {"model_id": model_cfg["id"], "checkpoint_sha256": parent["sha256"]}
    for pair in pairs:
        provenance = pair["provenance"]
        candidate = provenance.get("original_candidate_origin")
        if not isinstance(candidate, dict) or any(
            candidate.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"{pair['pair_id']}: original candidate differs from selected M1")
        for side in ("chosen", "rejected"):
            origin = provenance[side]
            if origin["kind"] == "model" and any(
                origin.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(f"{pair['pair_id']}: {side} model origin differs from selected M1")


class PairCollator:
    """Use SFT's exact prompt boundary and image processing for both completions."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("Preference collator requires batch size 1")
        pair = rows[0]
        result = {}
        for side in ("chosen", "rejected"):
            parsed = parse_prediction(pair[side])
            if not parsed.format_valid:
                raise ValueError("Invalid preference completion")
            row = {
                **pair,
                "question_id": pair["source_question_id"],
                "answer": parsed.answer,
                "explanations": [parsed.explanation],
            }
            result[side] = self.collator([row])
        return result
