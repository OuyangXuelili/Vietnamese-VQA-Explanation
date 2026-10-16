"""Verified replay and bounded single-request inference."""

import io
import json
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps, UnidentifiedImageError

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.inference.export import validate_saved_predictions
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.training.preference_data import adapter_fingerprint
from vqa_explanation.utils.run_provenance import sha256_file

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_QUESTION_LENGTH = 500


class DemoError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise DemoError("Hãy nhập câu hỏi.")
    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        raise DemoError("Câu hỏi tối đa 500 ký tự.")
    return question


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise DemoError("Ảnh phải có dữ liệu và không quá 6 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if (image.format not in {"JPEG", "PNG"}
                        or image.width * image.height > MAX_IMAGE_PIXELS):
                    raise DemoError("Chỉ nhận JPEG/PNG, tối đa 16 triệu điểm ảnh.")
                if getattr(image, "n_frames", 1) != 1:
                    raise DemoError("Không nhận ảnh động.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as error:
        raise DemoError("Ảnh hỏng hoặc không đọc được.") from error


@dataclass(frozen=True)
class DemoResult:
    mode: str
    experiment_id: str
    answer: str
    explanation: str
    raw_output: str
    format_valid: bool
    latency_seconds: float
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


class ReplayService:
    """Read exact validation outputs, never infer an uploaded image."""

    mode = "replay"

    def __init__(self, predictions: Path, manifest: Path, annotations: Path,
                 image_root: Path, selection_path: Path, *, count: int = 12):
        if type(count) is not int or not 1 <= count <= 30:
            raise ValueError("Replay example count must be between 1 and 30")
        self.image_root = image_root.resolve()
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected = {"validation_predictions": predictions, "validation_manifest": manifest,
                    "validation_annotations": annotations}
        if (selection.get("status") != "selected"
                or selection.get("selection_split") != "validation"
                or any(selection.get("evidence", {}).get(key, {}).get("sha256") != sha256_file(path)
                       for key, path in expected.items())):
            raise ValueError("Replay sources differ from the frozen M1 selection")
        records = list(read_jsonl(annotations))
        saved = list(read_jsonl(predictions))
        inputs = self.manifest.get("inputs", {})
        if (inputs.get("experiment_id") != "M1_SFT" or inputs.get("split") != "validation"
                or inputs.get("annotations_sha256") != sha256_file(annotations)
                or inputs.get("model_id") != selection.get("model_id")
                or inputs.get("model_revision") != selection.get("model_revision")
                or inputs.get("adapter_files") != selection.get("adapter", {}).get("files")
                or any(row.get("split") != "validation" for row in records)):
            raise ValueError("Replay requires the M1 validation export and its annotations")
        completed = validate_saved_predictions(saved, records, self.manifest)
        if not records or len(completed) != len(records):
            raise ValueError("Replay requires a complete verified prediction export")
        self.examples = []
        seen = set()
        for row in sorted(saved, key=lambda row: row["question_id"]):
            if row["image_id"] in seen:
                continue
            path = (self.image_root / row["image_path"]).resolve()
            if not path.is_relative_to(self.image_root) or not path.is_file():
                raise ValueError("Replay image is missing or escapes the image root")
            if sha256_file(path) != row["image_sha256"]:
                raise ValueError("Replay image checksum mismatch")
            decode_image(path.read_bytes())
            seen.add(row["image_id"])
            self.examples.append(row)
            if len(self.examples) == count:
                break
        if len(self.examples) != count:
            raise ValueError("Not enough distinct validation images")
        self.by_id = {row["question_id"]: row for row in self.examples}

    def info(self) -> dict:
        inputs = self.manifest["inputs"]
        return {"mode": self.mode, "experiment_id": "M1_SFT",
                "model_id": inputs["model_id"], "model_revision": inputs["model_revision"],
                "message": "Kết quả đã tính trước trên validation, không suy luận ảnh mới.",
                "examples": [{"question_id": row["question_id"], "question": row["question"]}
                             for row in self.examples]}

    def example(self, qid: str) -> dict:
        if qid not in self.by_id:
            raise DemoError("Không có ví dụ này.", 404)
        row = self.by_id[qid]
        result = DemoResult("replay", "M1_SFT", row["prediction_answer"],
                            row["prediction_explanation"], row["raw_output"], row["format_valid"],
                            row["latency_seconds"], "Kết quả đã lưu; thời gian là lần chạy gốc.")
        return {"question_id": qid, "question": row["question"], **result.to_dict()}

    def image(self, qid: str) -> bytes:
        self.example(qid)
        row = self.by_id[qid]
        path = self.image_root / row["image_path"]
        data = path.read_bytes()
        # Recheck mutable local files before serving them.
        if sha256_file(path) != row["image_sha256"]:
            raise DemoError("Ảnh ví dụ đã đổi; hãy khởi động lại và kiểm dữ liệu.", 409)
        image = decode_image(data)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


class LiveService:
    mode = "live"

    def __init__(self, runner, identity: dict, *, max_new_tokens: int = 128):
        self.runner, self.identity = runner, identity
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"mode": self.mode, **self.identity, "examples": [],
                "message": "Suy luận trực tiếp bằng M1. Câu trả lời vẫn có thể sai."}

    def predict(self, data: bytes, question: str) -> DemoResult:
        if not self.lock.acquire(blocking=False):
            raise DemoError("GPU đang xử lý một câu hỏi khác. Hãy thử lại sau.", 409)
        try:
            question = validate_question(question)
            image = decode_image(data)
            start = time.perf_counter()
            with TemporaryDirectory(prefix="vqa-demo-") as directory:
                path = Path(directory) / "image.png"
                image.save(path, format="PNG")
                raw = self.runner.generate(path, question, max_new_tokens=self.max_new_tokens)
            parsed = parse_prediction(raw)
            return DemoResult("live", self.identity["experiment_id"], parsed.answer,
                              parsed.explanation, raw, parsed.format_valid,
                              round(time.perf_counter() - start, 3),
                              "Mô hình có thể sai; giải thích không phải kiểm chứng độc lập.")
        finally:
            self.lock.release()


def load_live_service(adapter: Path, selection_path: Path, validation_manifest: Path):
    """Validate M1 identity and require a GPU before loading the backbone."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(validation_manifest.read_text(encoding="utf-8"))
    if (selection.get("status") != "selected"
            or selection.get("selection_split") != "validation"
            or selection.get("adapter") != adapter_fingerprint(adapter)
            or selection.get("evidence", {}).get("validation_manifest", {}).get("sha256")
            != sha256_file(validation_manifest)):
        raise ValueError("Adapter/validation manifest differs from the frozen M1 selection")
    inputs, protocol = manifest["inputs"], manifest["protocol"]
    if (inputs.get("experiment_id") != "M1_SFT" or inputs.get("split") != "validation"
            or inputs.get("adapter_files") != selection["adapter"]["files"]
            or inputs.get("model_id") != selection["model_id"]
            or inputs.get("model_revision") != selection["model_revision"]):
        raise ValueError("Selected M1 identity does not match validation")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Live demo requires a CUDA GPU; use replay on this machine")
    from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner

    runner = Qwen25VLRunner(
        selection["model_id"], selection["model_revision"], adapter_path=str(adapter),
        min_pixels=protocol["min_pixels"], max_pixels=protocol["max_pixels"],
        load_in_4bit=protocol["load_in_4bit"],
    )
    if any(str(device) in {"cpu", "disk"} for device in runner.model.hf_device_map.values()):
        raise RuntimeError("GPU memory is insufficient; CPU/disk offload is disabled for the demo")
    actual = runner.protocol_metadata(protocol["max_new_tokens"])
    for key in ("min_pixels", "max_pixels", "max_new_tokens", "load_in_4bit", "prompt_sha256",
                "generation_config"):
        if actual[key] != protocol[key]:
            raise ValueError(f"Live inference protocol changed: {key}")
    return LiveService(runner, {"experiment_id": "M1_SFT", "model_id": selection["model_id"],
                               "model_revision": selection["model_revision"]},
                       max_new_tokens=protocol["max_new_tokens"])
