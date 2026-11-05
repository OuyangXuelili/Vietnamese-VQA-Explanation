"""GPU acceptance checks for the existing demo API, without a public server."""

import base64
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.demo.service import ReplayService, decode_image, load_live_service
from vqa_explanation.training.preference_data import adapter_fingerprint
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def load_demo_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    result = {"examples": config["examples"]}
    for key in ("predictions", "manifest", "annotations", "selection"):
        expanded = os.path.expandvars(config[key])
        if "$" in expanded or "%" in expanded:
            raise ValueError("Unresolved variable in demo path")
        value = Path(expanded)
        result[key] = value if value.is_absolute() else project_root() / value
    return result


def inspect_demo(predictions, manifest, annotations, image_root, selection, adapter,
                 *, replay_count=12):
    """Verify evidence and fix three validation cases outside the replay gallery."""
    paths = {key: Path(value).resolve() for key, value in {
        "predictions": predictions, "manifest": manifest, "annotations": annotations,
        "image_root": image_root, "selection": selection, "adapter": adapter,
    }.items()}
    replay = ReplayService(*(paths[key] for key in (
        "predictions", "manifest", "annotations", "image_root", "selection")), count=replay_count)
    frozen = json.loads(paths["selection"].read_text(encoding="utf-8"))
    fingerprint = adapter_fingerprint(paths["adapter"])
    if fingerprint != frozen["adapter"]:
        raise ValueError("Adapter differs from the frozen M1 selection")
    excluded = {row["image_id"] for row in replay.examples}
    cases = []
    for row in sorted(read_jsonl(paths["predictions"]), key=lambda row: row["question_id"]):
        if row["image_id"] in excluded:
            continue
        path = (paths["image_root"] / row["image_path"]).resolve()
        if not path.is_relative_to(paths["image_root"]) or not path.is_file():
            raise ValueError("Demo image is missing or escapes the image root")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Demo image checksum mismatch")
        decode_image(path.read_bytes())
        cases.append({key: row[key] for key in (
            "question_id", "image_id", "image_path", "image_sha256", "question")})
        excluded.add(row["image_id"])
        if len(cases) == 3:
            break
    if len(cases) != 3:
        raise ValueError("Need three distinct validation images outside the replay gallery")
    plan = {
        "schema": "demo_live_check_v1", "scope": "validation_only", "paths": {
            key: str(path) for key, path in paths.items()},
        "input_sha256": {key: sha256_file(paths[key]) for key in (
            "predictions", "manifest", "annotations", "selection")},
        "adapter": fingerprint, "identity": replay.info(), "replay_count": replay_count,
        "cases": cases,
    }
    plan["sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return plan


def check_output_path(output, plan):
    output = Path(output).resolve()
    root = project_root()
    protected = [Path(value) for value in plan["paths"].values()]
    protected += [root / name for name in ("src", "scripts", "configs", "tests", "notebooks",
                                           "app", "data", "runs")]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Output overlaps protected inputs or source")
    if output.exists():
        raise FileExistsError(f"Use a new demo check directory: {output}")
    return output


def _request_cases(service, plan, *, repeat=False, save_case):
    from fastapi.testclient import TestClient

    from vqa_explanation.demo.web import create_app

    results = []
    # This exercises HTTP handlers in-process; it does not expose a listening port.
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        for route in ("/", "/assets/demo.js", "/assets/demo.css", "/api/info"):
            if client.get(route).status_code != 200:
                raise RuntimeError(f"Demo route failed: {route}")
        if client.get("/api/info").json() != service.info() or service.mode != "live":
            raise RuntimeError("Demo API is not using the live service")
        for case in plan["cases"][:1] if repeat else plan["cases"]:
            path = Path(plan["paths"]["image_root"]) / case["image_path"]
            if sha256_file(path) != case["image_sha256"]:
                raise ValueError("Demo image changed after inspection")
            payload = {"question": case["question"],
                       "image_base64": base64.b64encode(path.read_bytes()).decode()}
            if not repeat:
                for invalid in ({**payload, "question": " "},
                                {**payload, "image_base64": "bad!"}):
                    if client.post("/api/predict", json=invalid).status_code != 400:
                        raise RuntimeError("Invalid demo input was not rejected")
                with service.lock:
                    if client.post("/api/predict", json=payload).status_code != 409:
                        raise RuntimeError("Busy demo request was not rejected")
            start = time.perf_counter()
            response = client.post("/api/predict", json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Live request failed: HTTP {response.status_code}")
            result = response.json()
            if result.get("mode") != "live" or result.get("experiment_id") != "M1_SFT":
                raise RuntimeError("Unexpected live result identity")
            results.append({**case, "result": result,
                            "request_seconds": round(time.perf_counter() - start, 3)})
            save_case({"stage": "reload" if repeat else "initial", **results[-1]})
    return results


def _load_and_check(plan, *, repeat=False, save_case):
    paths = plan["paths"]
    service = load_live_service(Path(paths["adapter"]), Path(paths["selection"]),
                                Path(paths["manifest"]))
    try:
        return _request_cases(service, plan, repeat=repeat, save_case=save_case)
    finally:
        del service
        gc.collect()
        import torch

        torch.cuda.empty_cache()


def run_demo_check(plan, output, *, expected_sha256):
    """Run real CUDA inference; a technical pass is not a model quality score."""
    paths = plan["paths"]
    current = inspect_demo(**paths, replay_count=plan["replay_count"])
    if current != plan or plan["sha256"] != expected_sha256:
        raise ValueError("Demo inputs changed; inspect again")
    output = check_output_path(output, plan)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Live demo check requires CUDA; no model was loaded")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "demo_live_check_v1", "status": "failed", "plan": plan,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "model_quality_reviewed": False,
              "browser_ui_reviewed": False, "cases": [], "reload": []}
    try:
        torch.cuda.reset_peak_memory_stats()
        with (output / "results.jsonl").open("x", encoding="utf-8") as saved:
            def save_case(row):
                saved.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved.flush()

            report["cases"] = _load_and_check(plan, save_case=save_case)
            # Release the first model before loading it again; never keep two on the GPU.
            report["reload"] = _load_and_check(plan, repeat=True, save_case=save_case)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["reload_raw_output_matches"] = (
            report["cases"][0]["result"]["raw_output"]
            == report["reload"][0]["result"]["raw_output"])
        valid = all(row["result"].get("format_valid") and row["result"].get("answer")
                    and row["result"].get("explanation")
                    for row in report["cases"] + report["reload"])
        if not valid:
            raise RuntimeError("Live inference returned empty or invalid-format output")
        report["status"] = "technical_passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (output / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
