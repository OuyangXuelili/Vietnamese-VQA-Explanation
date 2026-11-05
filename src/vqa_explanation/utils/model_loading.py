"""Explicit precision and quantization settings for new inference runs."""


def validate_loading_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {"dtype", "double_quant", "attention"}:
        raise ValueError("Loading profile requires dtype, double_quant and attention")
    if (not isinstance(profile["dtype"], str)
            or profile["dtype"] not in {"auto", "float16", "bfloat16", "float32"}):
        raise ValueError("Unsupported loading dtype")
    if type(profile["double_quant"]) is not bool:
        raise ValueError("double_quant must be a boolean")
    if not isinstance(profile["attention"], str) or profile["attention"] not in {"sdpa", "eager"}:
        raise ValueError("Use sdpa or eager attention")
    return dict(profile)


def resolve_dtype(torch, name, *, native_bf16):
    if name not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported loading dtype")
    if name == "auto":
        name = "bfloat16" if native_bf16 else "float16"
    if name == "bfloat16" and not native_bf16:
        raise ValueError("GPU has no native BF16; choose float16 or float32")
    return getattr(torch, name)


def quantization_kwargs(dtype, *, double_quant):
    if type(double_quant) is not bool:
        raise ValueError("double_quant must be a boolean")
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": double_quant,
        "bnb_4bit_compute_dtype": dtype,
    }
