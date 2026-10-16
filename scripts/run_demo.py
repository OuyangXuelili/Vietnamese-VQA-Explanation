"""Serve saved validation examples locally, or M1 inference on a CUDA runtime."""

import argparse
import os
from pathlib import Path

import yaml

from vqa_explanation.demo.service import ReplayService, load_live_service
from vqa_explanation.utils.environment import load_environment, project_root


def resolve_path(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded or "%" in expanded:
        raise ValueError("Unresolved variable in demo path")
    path = Path(expanded)
    return path if path.is_absolute() else project_root() / path


def launch_settings(host: str, port: int) -> dict:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    user, password = os.getenv("VQA_DEMO_USER"), os.getenv("VQA_DEMO_PASSWORD")
    if bool(user) != bool(password):
        raise ValueError("Set both VQA_DEMO_USER and VQA_DEMO_PASSWORD")
    if host not in {"127.0.0.1", "localhost", "::1"} and not user:
        raise ValueError("A non-loopback server requires demo authentication and HTTPS forwarding")
    return {"auth": (user, password) if user else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--config", type=Path, default=resolve_path("configs/demo/m1.yaml"))
    parser.add_argument("--environment", type=Path,
                        default=resolve_path("configs/environments/local_full.yaml"))
    parser.add_argument("--adapter", type=Path, help="M1 checkpoint directory in the GPU runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    settings = launch_settings(args.host, args.port)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    predictions = resolve_path(config["predictions"])
    manifest = resolve_path(config["manifest"])
    if args.mode == "replay":
        environment = load_environment(args.environment)
        service = ReplayService(predictions, manifest, resolve_path(config["annotations"]),
                                environment["image_root"], resolve_path(config["selection"]),
                                count=config["examples"])
    else:
        if args.adapter is None:
            parser.error("Live mode requires --adapter; never train/load on CPU implicitly")
        service = load_live_service(args.adapter, resolve_path(config["selection"]), manifest)
    import uvicorn

    from vqa_explanation.demo.web import create_app

    print(f"Demo mode: {service.mode}; http://{args.host}:{args.port}; no training", flush=True)
    uvicorn.run(create_app(service, **settings), host=args.host, port=args.port, workers=1,
                access_log=False)


if __name__ == "__main__":
    main()
