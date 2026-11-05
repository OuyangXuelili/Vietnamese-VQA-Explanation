"""Request-scoped checks of raw model tensors, before generation token masks."""

from collections.abc import Mapping
from contextlib import contextmanager


class NumericalInferenceError(RuntimeError):
    """The model produced NaN or infinity; no generated answer is trustworthy."""


def _check_tensors(value, torch, stage):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise NumericalInferenceError(f"Non-finite model output at {stage}")
    elif isinstance(value, Mapping):
        for item in value.values():
            _check_tensors(item, torch, stage)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_tensors(item, torch, stage)


@contextmanager
def guard_model_outputs(model, torch):
    """Observe vision features and raw logits without changing generation settings.

    PEFT's generate delegates to its base model, bypassing wrapper forward hooks.
    Named modules cover both Qwen's older visual layout and model.visual layout.
    Processed scores are excluded: their -inf masks can be legal.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    handles = []

    def check_raw(_module, _args, output):
        _check_tensors(output, torch, "vision/lm_head")

    def check_logits(_module, _args, output):
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(
            output, "logits", None)
        _check_tensors(logits, torch, "logits")

    try:
        for name, module in base.named_modules():
            if name.rsplit(".", 1)[-1] in {"visual", "lm_head"}:
                handles.append(module.register_forward_hook(check_raw))
        handles.append(base.register_forward_hook(check_logits))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
