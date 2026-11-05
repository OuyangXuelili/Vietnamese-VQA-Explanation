from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from vqa_explanation.inference.generation_safety import guard_model_outputs
from vqa_explanation.inference.prompting import SYSTEM_PROMPT, build_user_prompt
from vqa_explanation.training.sft import supports_native_bf16
from vqa_explanation.utils.model_loading import (
    quantization_kwargs,
    resolve_dtype,
    validate_loading_profile,
)


class Qwen25VLRunner:
    """Qwen2.5-VL inference with an optional LoRA adapter."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        load_in_4bit: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        adapter_path: str | None = None,
        loading_profile: dict | None = None,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - exercised on GPU runtime
            raise RuntimeError(
                'Install the training dependencies with: pip install -e ".[train]"'
            ) from exc

        profile = validate_loading_profile(loading_profile if loading_profile is not None else {
            "dtype": "auto", "double_quant": False, "attention": "sdpa",
        })
        if profile["double_quant"] and not load_in_4bit:
            raise ValueError("double_quant requires load_in_4bit")
        dtype = resolve_dtype(torch, profile["dtype"], native_bf16=supports_native_bf16(torch.cuda))
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                **quantization_kwargs(dtype, double_quant=profile["double_quant"]))

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=quantization_config,
            attn_implementation=profile["attention"],
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.config.use_cache = True
        self.model.eval()
        self.torch = torch
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.load_in_4bit = load_in_4bit
        self.compute_dtype = str(dtype)
        self.loading_profile = ({**profile, "dtype": str(dtype).removeprefix("torch.")}
                                if loading_profile is not None else None)

    def protocol_metadata(self, max_new_tokens: int) -> dict:
        """Return the loaded prompt template and generation settings used for inference."""
        generation = self.model.generation_config.to_dict()
        generation.update(do_sample=False, max_new_tokens=max_new_tokens)
        prompt_spec = {
            "system": SYSTEM_PROMPT,
            "user_template": build_user_prompt("<QUESTION>"),
            "chat_template": self.processor.chat_template,
        }
        digest = hashlib.sha256(
            json.dumps(prompt_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metadata = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_new_tokens": max_new_tokens,
            "load_in_4bit": self.load_in_4bit,
            "prompt_sha256": digest,
            "generation_config": generation,
            "compute_dtype": self.compute_dtype,
        }
        if getattr(self, "loading_profile", None) is not None:
            metadata["model_loading"] = dict(self.loading_profile)
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None:
                spec = {"class": type(image_processor).__name__,
                        "config": image_processor.to_dict()}
                metadata["image_processor_sha256"] = hashlib.sha256(
                    json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return metadata

    def generate(self, image_path: Path, question: str, *, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_user_prompt(question)},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode(), guard_model_outputs(self.model, self.torch):
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        return self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
