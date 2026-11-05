"""Data and assistant-only supervision for image-conditioned SFT."""

from math import prod
from pathlib import Path

from PIL import Image

from vqa_explanation.data.formatting import ANSWER_TAG, format_target
from vqa_explanation.data.schema import VQASample
from vqa_explanation.inference.prompting import SYSTEM_PROMPT, build_user_prompt


def select_training_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    import random

    selected = []
    seen = set()
    samples = []
    for raw in records:
        sample = VQASample.from_dict(raw)
        sample.validate_ids()
        if raw["split"] != sample.split:
            raise ValueError(f"Noncanonical split label: {raw['split']}")
        samples.append((raw, sample))
    heldout_images = {sample.image_id for _, sample in samples if sample.split != "train"}
    heldout_paths = {sample.image_path for _, sample in samples if sample.split != "train"}
    for raw, sample in samples:
        if sample.split != "train":
            continue
        if sample.question_id in seen:
            raise ValueError(f"Duplicate training question: {sample.question_id}")
        if sample.image_id in heldout_images or sample.image_path in heldout_paths:
            raise ValueError(f"Train/held-out image overlap: {sample.image_path}")
        seen.add(sample.question_id)
        selected.append(raw)
    random.Random(seed).shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No training samples")
    return selected


def messages_for(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_user_prompt(record["question"])},
            ],
        },
    ]


def target_for(record: dict) -> str:
    sample = VQASample.from_dict(record)
    return format_target(sample.answer, sample.explanations[0])


def assistant_labels(full: list[int], prompt: list[int], image_ids: set[int]) -> list[int]:
    """Validate the template boundary for assistant-only supervision."""
    if not prompt or full[: len(prompt)] != prompt:
        raise ValueError("Prompt is not an exact token prefix of the full conversation")
    if len(full) <= len(prompt):
        raise ValueError("No assistant tokens to supervise")
    labels = [-100] * len(prompt) + full[len(prompt) :]
    labels = [
        -100 if token in image_ids else label for token, label in zip(full, labels, strict=True)
    ]
    if all(label == -100 for label in labels):
        raise ValueError("All assistant labels are masked")
    return labels


def answer_token_mask(offsets, labels, start, end):
    """Select supervised tokens overlapping the answer's character span."""
    if (type(start) is not int or type(end) is not int or not 0 <= start < end
            or len(offsets) != len(labels)):
        raise ValueError("Invalid answer span or token offsets")
    mask = []
    for offset, label in zip(offsets, labels, strict=True):
        if (len(offset) != 2 or any(type(n) is not int for n in offset)
                or not 0 <= offset[0] <= offset[1]):
            raise ValueError("Invalid token offset")
        mask.append(label != -100 and offset[1] > start and offset[0] < end)
    if not any(mask):
        raise ValueError("Answer has no supervised tokens")
    return mask


class VQACollator:
    """Single-image supervision; reuse the image grid when encoding the prompt."""

    def __init__(
        self,
        processor,
        image_root: Path,
        max_length: int = 1536,
        *,
        reuse_image_grid: bool = True,
        return_tensors: str = "pt",
        include_answer_mask: bool = False,
    ):
        if return_tensors not in {"pt", "np"}:
            raise ValueError("return_tensors must be pt or np")
        self.processor = processor
        self.image_root = image_root.resolve()
        self.max_length = max_length
        self.reuse_image_grid = reuse_image_grid
        self.return_tensors = return_tensors
        self.include_answer_mask = include_answer_mask
        vocab = processor.tokenizer.get_vocab()
        self.image_ids = {
            vocab[token]
            for token in (
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            )
            if token in vocab
        }

    def _expand_image_tokens(self, text, batch):
        # Qwen2.5-VL 4.57.6 expands each image placeholder using this grid.
        # Calling the processor without images retains its tokenizer defaults.
        token = self.processor.image_token
        grid = batch["image_grid_thw"].tolist()
        merge_size = self.processor.image_processor.merge_size
        if text.count(token) != 1 or len(grid) != 1:
            raise ValueError("Prompt must contain exactly one image and one image grid")
        if len(grid[0]) != 3 or any(type(n) is not int or n <= 0 for n in grid[0]):
            raise ValueError("Invalid image grid")
        if type(merge_size) is not int or merge_size <= 0:
            raise ValueError("Invalid image merge size")
        patches = prod(grid[0])
        if patches % (merge_size**2):
            raise ValueError("Image grid is not divisible by merge size")
        return text.replace(token, token * (patches // merge_size**2), 1)

    def _prompt_ids(self, prompt, image, batch):
        if not self.reuse_image_grid:
            return self.processor(
                text=[prompt], images=[image], return_tensors=self.return_tensors,
            )["input_ids"][0].tolist()
        return self.processor(
            text=[self._expand_image_tokens(prompt, batch)],
            return_tensors=self.return_tensors,
        )["input_ids"][0].tolist()

    def __call__(self, records):
        if len(records) != 1:
            raise ValueError("Use batch size 1 and gradient accumulation")
        record = records[0]
        path = (self.image_root / record["image_path"]).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError("Image path escapes image root")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        messages = messages_for(record)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # One reference per question avoids overweighting multi-reference questions.
        target = target_for(record)
        full = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.processor(text=[full], images=[image], return_tensors=self.return_tensors)
        full_ids = batch["input_ids"][0].tolist()
        prompt_ids = self._prompt_ids(prompt, image, batch)
        if len(full_ids) > self.max_length:
            raise ValueError(
                f"{record['question_id']}: {len(full_ids)} tokens exceeds {self.max_length}; "
                "reduce image resolution or raise max_length; do not truncate image tokens"
            )
        labels = assistant_labels(full_ids, prompt_ids, self.image_ids)
        answer_mask = None
        if self.include_answer_mask:
            if not full.startswith(prompt + target):
                raise ValueError("Answer supervision requires an exact text template boundary")
            try:
                encoded = self.processor(
                    text=[self._expand_image_tokens(full, batch)],
                    return_tensors=self.return_tensors, return_offsets_mapping=True,
                )
            except (NotImplementedError, TypeError) as error:
                raise ValueError("Answer supervision requires tokenizer offsets") from error
            if encoded["input_ids"][0].tolist() != full_ids:
                raise ValueError("Answer offsets differ from the image-conditioned token sequence")
            start = len(self._expand_image_tokens(prompt, batch)) + len(ANSWER_TAG) + 1
            end = start + len(VQASample.from_dict(record).answer.strip())
            answer_mask = answer_token_mask(
                encoded["offset_mapping"][0].tolist(), labels, start, end)
        if self.return_tensors == "pt":
            import torch

            batch["labels"] = torch.tensor([labels], dtype=torch.long)
            if answer_mask is not None:
                batch["answer_mask"] = torch.tensor([answer_mask], dtype=torch.bool)
        else:
            import numpy as np

            batch["labels"] = np.asarray([labels], dtype=np.int64)
            if answer_mask is not None:
                batch["answer_mask"] = np.asarray([answer_mask], dtype=np.bool_)
        return batch
