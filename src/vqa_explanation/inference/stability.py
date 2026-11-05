"""Compare loading profiles on fixed numerical regression cases."""

import gc
import time
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.inference.generation_safety import NumericalInferenceError
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.model_loading import validate_loading_profile
from vqa_explanation.utils.run_provenance import sha256_file, source_fingerprints


def prepare_stability(environment, config, model_config):
    env = load_environment(environment)
    cfg = yaml.safe_load(Path(config).read_text("utf-8"))
    model = yaml.safe_load(Path(model_config).read_text("utf-8"))["model"]
    for key in ("min_pixels", "max_pixels", "max_new_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(f"Invalid stability setting: {key}")
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    rows = {r["question_id"]: r for r in read_jsonl(env["annotations"])}
    cases, names = [], set()
    for case in cfg["cases"]:
        if (not isinstance(case.get("name"), str) or not case["name"].strip()
                or case["name"] in names or case.get("role") not in {"control", "regression"}):
            raise ValueError("Use distinct named control/regression cases")
        names.add(case["name"])
        row = rows[str(case["question_id"])]
        path = (env["image_root"] / row["image_path"]).resolve()
        if not path.is_relative_to(env["image_root"].resolve()) or not path.is_file():
            raise ValueError("Regression image is missing or outside the image root")
        cases.append({**case, **{k: row[k] for k in ("question_id", "image_path", "question")},
                      "image_sha256": sha256_file(path)})
    if not cases or not any(c["role"] == "regression" for c in cases):
        raise ValueError("Provide a numerical regression case and controls")
    if not any(c["role"] == "control" for c in cases):
        raise ValueError("Provide controls alongside the regression case")
    profiles = {}
    for name, relative in cfg["profiles"].items():
        path = project_root() / relative
        profiles[name] = {"settings": validate_loading_profile(
            yaml.safe_load(path.read_text("utf-8"))),
                          "source_sha256": sha256_file(path)}
    if not profiles:
        raise ValueError("Provide at least one loading profile")
    plan = {"schema": "numerical_stability_plan_v1", "model": model, "cases": cases,
            "profiles": profiles,
            **{k: cfg[k] for k in ("min_pixels", "max_pixels", "max_new_tokens")},
            "annotations_sha256": sha256_file(env["annotations"]),
            "config_sha256": sha256_file(Path(config)),
            "model_config_sha256": sha256_file(Path(model_config)),
            "source_files": source_fingerprints(project_root()),
            "quality_evaluated": False, "benchmark_changed": False}
    return env, plan


def probe_cases(runner, cases, image_root, max_new_tokens):
    results = []
    for case in cases:
        path = Path(image_root) / case["image_path"]
        if sha256_file(path) != case["image_sha256"]:
            raise ValueError("Regression image changed after inspection")
        started = time.perf_counter()
        try:
            raw = runner.generate(path, case["question"], max_new_tokens=max_new_tokens)
            row = {"status": "finite", "raw_output": raw}
        except NumericalInferenceError as error:
            row = {"status": "nonfinite", "error": str(error)}
        except Exception as error:
            row = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        results.append({**case, **row, "seconds": round(time.perf_counter() - started, 3)})
    return results


def run_stability(env, plan, output, *, adapter=None):
    import torch

    from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner
    from vqa_explanation.training.preference_data import adapter_fingerprint

    if not torch.cuda.is_available():
        raise RuntimeError("Numerical stability execution requires a CUDA GPU")
    output = Path(output)
    if (output.exists() or output.resolve().is_relative_to(env["image_root"].resolve())
            or adapter and output.resolve().is_relative_to(Path(adapter).resolve())):
        raise ValueError("Use a new diagnostic output outside the image directory")
    parent = adapter_fingerprint(Path(adapter)) if adapter else None
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    report = {"schema": "numerical_stability_report_v1", "status": "running",
              "gpu": torch.cuda.get_device_name(0), "adapter": parent, "profiles": {},
              "quality_evaluated": False, "benchmark_changed": False,
              "root_cause_fixed": False,
              "limitation": "Short probes do not establish full-generation stability or quality."}
    for name, profile in plan["profiles"].items():
        runner = None
        torch.cuda.reset_peak_memory_stats()
        try:
            runner = Qwen25VLRunner(
                plan["model"]["id"], plan["model"]["revision"], load_in_4bit=True,
                min_pixels=plan["min_pixels"], max_pixels=plan["max_pixels"],
                loading_profile=profile["settings"], adapter_path=str(adapter) if adapter else None)
            device_map = {k: str(v) for k, v in runner.model.hf_device_map.items()}
            if any(device in {"cpu", "disk"} for device in device_map.values()):
                raise RuntimeError("Numerical profile offloaded to CPU/disk")
            rows = probe_cases(runner, plan["cases"], env["image_root"], plan["max_new_tokens"])
            report["profiles"][name] = {
                "protocol": runner.protocol_metadata(plan["max_new_tokens"]), "cases": rows,
                "device_map": device_map,
                "all_probes_finite": all(r["status"] == "finite" for r in rows),
                "controls_finite": all(r["status"] == "finite" for r in rows
                                       if r["role"] == "control"),
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        except Exception as error:
            report["profiles"][name] = {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error)}
        finally:
            del runner
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError as error:
                report["profiles"][name]["cleanup_error"] = str(error)
        write_json(output / "report.json", report)
    report["status"] = ("diagnostic_complete" if any("cases" in r
                                                  for r in report["profiles"].values())
                        else "failed")
    write_json(output / "report.json", report)
    return report
