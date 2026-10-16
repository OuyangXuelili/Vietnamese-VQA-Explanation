"""New preference trainers; the pinned M1 runtime is unchanged."""

import importlib.metadata
import json
import math
from contextlib import contextmanager
from pathlib import Path

from vqa_explanation.data.io import write_json
from vqa_explanation.training.preference_data import (
    PairCollator,
    adapter_fingerprint,
    prepare_stage,
)
from vqa_explanation.training.preference_resume import (
    checkpoint_callback,
    resume_identity,
    validate_resume,
)
from vqa_explanation.training.sft_data import VQACollator
from vqa_explanation.utils.run_provenance import source_fingerprints, utc_now

RUNTIME = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.2",
}


def use_bf16(cfg, cuda):
    from vqa_explanation.training.sft import supports_native_bf16

    native = supports_native_bf16(cuda)
    if cfg["dtype"] == "bfloat16" and not native:
        raise ValueError("GPU has no native BF16; use auto/float16 for new stages")
    return cfg["dtype"] == "bfloat16" or (cfg["dtype"] == "auto" and native)


def completion_logp(model, batch):
    """Sum next-token log-probabilities over assistant tokens only."""
    import torch

    labels = batch["labels"][:, 1:]
    output = model(
        **{key: value for key, value in batch.items() if key != "labels"}, use_cache=False
    )
    mask = labels != -100
    if labels.shape[0] != 1 or not torch.any(mask):
        raise ValueError("DPO requires one sample with assistant tokens")
    logits = output.logits[:, :-1][mask].float()
    targets = labels[mask]
    return (logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)).sum()


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    import torch.nn.functional as functional

    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -functional.logsigmoid(margin).mean()


@contextmanager
def reference_adapter(model):
    """Restore policy even if a reference forward fails."""
    try:
        model.set_adapter("reference")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)
        yield
    finally:
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)


def trainable_policy_only(model):
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".default." not in name or "lora_" not in name for name in trainable):
        raise ValueError("Only the M1-derived policy LoRA may be trainable")


def reference_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if ".reference." in name
    }


def verify_reference(model, before):
    import torch

    after = reference_state(model)
    if (
        not before
        or set(before) != set(after)
        or any(not torch.equal(value, after[name]) for name, value in before.items())
    ):
        raise ValueError("Frozen M1 reference weights changed")
    trainable_policy_only(model)


def verify_saved_policy(model, path):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    expected = get_peft_model_state_dict(model, adapter_name="default")
    saved = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    if set(expected) != set(saved) or any(
        not torch.equal(value.detach().cpu(), saved[name]) for name, value in expected.items()
    ):
        raise ValueError("Saved policy weights differ from the trained adapter")


def runtime_versions():
    versions = {}
    for name, expected in RUNTIME.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "Install requirements-cloud-preference.txt in a new GPU runtime"
            ) from error
        if actual != expected:
            raise RuntimeError(f"{name}={actual}; preference runtime requires {expected}")
        versions[name] = actual
    return versions


def load_policy(cfg, model_cfg, adapter, *, with_reference):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a new single-GPU runtime; local CPU is for data and tests")
    bf16 = use_bf16(cfg, torch.cuda)
    dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(
        model_cfg["id"],
        revision=model_cfg["revision"],
        trust_remote_code=False,
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["id"],
        revision=model_cfg["revision"],
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation=cfg["attn_implementation"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True, adapter_name="default")
    if with_reference:
        model.load_adapter(str(adapter), adapter_name="reference", is_trainable=False)
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)
        for name, value in reference_state(model).items():
            policy = model.get_parameter(name.replace(".reference.", ".default."))
            if not torch.equal(value, policy.detach().cpu()):
                raise ValueError("Policy/reference must start from identical M1 weights")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    trainable_policy_only(model)
    return processor, model, bf16


def make_trainer_class(cfg):
    import torch
    from transformers import Trainer

    class PreferenceTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Trainer, not our loss, scales each microbatch by accumulation steps.
            self.model_accepts_loss_kwargs = False

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.torch import load_file

            policy = model if model is not None else self.model
            path = Path(resume_from_checkpoint)
            saved = load_file(str(path / "adapter_model.safetensors"), device="cpu")
            if set(saved) != set(get_peft_model_state_dict(policy, adapter_name="default")):
                raise ValueError("Resume policy adapter keys differ")
            # Trainer 4.57.6 skips the root policy when a reference subfolder exists.
            set_peft_model_state_dict(policy, saved, adapter_name="default")
            policy.set_adapter("default")
            for name, parameter in policy.named_parameters():
                if ".reference." in name:
                    parameter.requires_grad_(False)
            trainable_policy_only(policy)
            verify_saved_policy(policy, path)

        def compute_loss(self, policy, inputs, return_outputs=False, num_items_in_batch=None):
            if cfg["stage"] == "sft_chosen":
                loss, outputs = super().compute_loss(policy, inputs, return_outputs=True)
            else:
                chosen_logp = completion_logp(policy, inputs["chosen"])
                rejected_logp = completion_logp(policy, inputs["rejected"])
                with torch.no_grad(), reference_adapter(policy):
                    ref_chosen = completion_logp(policy, inputs["chosen"])
                    ref_rejected = completion_logp(policy, inputs["rejected"])
                loss = dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, cfg["beta"])
                outputs = {"loss": loss}
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            return (loss, outputs) if return_outputs else loss

    return PreferenceTrainer


def run_stage(args):
    env, cfg, plan, pairs, chosen = prepare_stage(args)
    if args.preflight_only:
        print(
            json.dumps(
                {"status": "cpu_metadata_passed", "plan": plan, "gpu_training_verified": False},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    import torch
    from transformers import TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU; preflight-only needs no model weights")
    versions = runtime_versions()
    dtype = "bfloat16" if use_bf16(cfg, torch.cuda) else "float16"
    gpu = {"name": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0))}
    budget = args.pilot_steps or plan["expected_updates"]
    stop = getattr(args, "stop_after_step", None)
    resume_path = getattr(args, "resume_from_checkpoint", None)
    identity = resume_identity(plan, versions, dtype, gpu, budget)
    resume = validate_resume(resume_path, identity) if resume_path else None
    start_step = resume["global_step"] if resume else 0
    if stop is not None and (type(stop) is not int or not start_step < stop < budget):
        raise ValueError("stop-after-step must be between the resume step and original budget")
    if args.pilot_steps and not 1 <= args.pilot_steps <= 10:
        raise ValueError("GPU gate may use 1–10 updates")
    if not args.pilot_steps:
        if args.gpu_gate is None:
            raise ValueError("Run one short GPU gate for this new trainer before its full stage")
        gate = json.loads(args.gpu_gate.read_text(encoding="utf-8"))
        if gate.get("status") != "gpu_gate_passed" or gate.get("plan_sha256") != plan["sha256"]:
            raise ValueError("GPU gate does not match current source/data/M1/config")
        if gate.get("runtime") != versions:
            raise ValueError("GPU gate runtime differs")
        if gate.get("gpu") != gpu or gate.get("policy_updated") is not True:
            raise ValueError("GPU gate hardware or policy update differs")
        if cfg["stage"] == "dpo" and gate.get("reference_unchanged") is not True:
            raise ValueError("DPO gate must verify the frozen M1 reference")
        if gate.get("dtype") != dtype or gate.get("saved_policy_verified") is not True:
            raise ValueError("GPU gate dtype/save verification differs")
        if cfg["stage"] == "dpo" and gate.get("estimated_full_train_hours", math.inf) > 8:
            raise ValueError("DPO estimate exceeds 8 hours; review the shared budget")
    output = args.output_dir or env["run_root"] / cfg["experiment_id"] / utc_now().replace(":", "")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    processor, model, bf16 = load_policy(
        cfg, plan["model"], args.m1_adapter, with_reference=cfg["stage"] == "dpo"
    )
    length = cfg["max_length"] if cfg["stage"] == "sft_chosen" else cfg["preflight_max_length"]
    sft_collator = VQACollator(processor, env["image_root"], max_length=length)
    collator = PairCollator(sft_collator) if cfg["stage"] == "dpo" else sft_collator
    rows = pairs if cfg["stage"] == "dpo" else chosen
    # Check the real collator once before training, without truncating completions.
    for row in rows:
        collator([row])
    initial = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }
    frozen = reference_state(model) if cfg["stage"] == "dpo" else None
    output.mkdir(parents=True, exist_ok=False)
    release_files = source_fingerprints(Path(__file__).resolve().parents[3])
    write_json(output / "data_plan.json", plan)
    write_json(
        output / "execution.json",
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "runtime": versions,
            "plan_sha256": plan["sha256"],
            "release_source_files": release_files,
            "resume": resume,
        },
    )
    trainer_class = make_trainer_class(cfg)
    try:
        training_args = TrainingArguments(
            output_dir=str(output / "training"),
            seed=cfg["seed"],
            data_seed=cfg["seed"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["num_train_epochs"],
            max_steps=args.pilot_steps if args.pilot_steps else -1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=not bf16,
            bf16=bf16,
            remove_unused_columns=False,
            report_to="none",
            logging_steps=cfg["logging_steps"],
            save_strategy="steps",
            save_steps=cfg["save_steps"],
            eval_strategy="no",
            save_total_limit=None,
            optim="adamw_torch",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            warmup_steps=0,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
        )
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=rows,
            data_collator=collator,
            processing_class=processor,
            callbacks=[checkpoint_callback(identity, stop)],
        )
        result = trainer.train(resume_from_checkpoint=resume["checkpoint"] if resume else None)
        expected = stop or budget
        if trainer.state.global_step != expected or not math.isfinite(result.training_loss):
            raise ValueError("Stage stopped before its declared budget")
        if not stop and not args.pilot_steps and not math.isclose(
            trainer.state.epoch, float(cfg["num_train_epochs"]), abs_tol=1e-6
        ):
            raise ValueError("Full stage did not finish its declared epochs")
        if not any(
            not torch.equal(value, model.get_parameter(name).detach().cpu())
            for name, value in initial.items()
        ):
            raise ValueError("Policy weights did not update")
        if frozen is not None:
            verify_reference(model, frozen)
        if adapter_fingerprint(args.m1_adapter) != plan["m1_adapter"]:
            raise ValueError("Input M1 adapter changed")
        if prepare_stage(args)[2] != plan:
            raise ValueError("Source, selection or preference inputs changed during training")
        model.save_pretrained(str(output / "adapter"), selected_adapters=["default"])
        processor.save_pretrained(str(output / "adapter"))
        verify_saved_policy(model, output / "adapter")
        trainer.save_state()
        report = {
            "status": "paused" if stop else "gpu_gate_passed" if args.pilot_steps else "passed",
            "plan_sha256": plan["sha256"],
            "release_source_files": release_files,
            "runtime": versions,
            "gpu": gpu,
            "global_step": trainer.state.global_step,
            "resume": resume,
            "updates_this_session": expected - start_step,
            "loss_note": "Trainer summary; use log_history for per-update loss after resume.",
            "epoch": trainer.state.epoch,
            "training_loss": result.training_loss,
            "metrics": result.metrics,
            "dtype": "bfloat16" if bf16 else "float16",
            "policy_updated": True,
            "reference_unchanged": True if frozen is not None else None,
            "saved_policy_verified": True,
            "estimated_full_train_hours": result.metrics["train_runtime"]
            * plan["expected_updates"]
            / (expected - start_step)
            / 3600,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "output_adapter": adapter_fingerprint(output / "adapter"),
            "last_checkpoint": str(output / "training" / f"checkpoint-{expected}"),
            "finished_at_utc": utc_now(),
        }
        write_json(output / "training_result.json", report)
        write_json(output / "execution.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as error:
        write_json(
            output / "execution.json",
            {
                "status": "failed",
                "error": str(error),
                "plan_sha256": plan["sha256"],
                "finished_at_utc": utc_now(),
            },
        )
        raise
