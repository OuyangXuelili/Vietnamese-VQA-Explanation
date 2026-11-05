"""Private Colab proxy policy and lifecycle of the notebook-owned web process."""

import ipaddress
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


def proxy_policy(origin, port=None, frame_origin=None):
    parsed = urlsplit(origin)
    hostname = parsed.hostname or ""
    legacy = re.fullmatch(r"([a-z0-9-]+)-(\d+)-colab\.googleusercontent\.com", hostname)
    current = re.fullmatch(
        r"(\d+)-([a-z0-9]+(?:-[a-z0-9]+)*)-([a-z])\.([a-z][a-z0-9-]*)"
        r"\.prod\.r\.colab\.dev",
        hostname,
    )
    if (not (legacy or current) or parsed.scheme != "https" or parsed.netloc != hostname
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Use the exact HTTPS origin returned by Colab proxyPort")
    actual_port = int(legacy[2] if legacy else current[1])
    if not 1 <= actual_port <= 65535 or (port is not None and actual_port != port):
        raise ValueError("Colab proxy port differs from the web server port")
    if not frame_origin and not legacy:
        raise ValueError("Provide the exact Colab output-frame origin")
    frame = urlsplit(frame_origin or f"https://{legacy[1]}-0-colab.googleusercontent.com")
    if (frame.scheme != "https" or frame.netloc != frame.hostname
            or not re.fullmatch(r"[a-z0-9-]+-0-colab\.googleusercontent\.com",
                                frame.hostname or "")
            or frame.path not in {"", "/"} or frame.query or frame.fragment):
        raise ValueError("Use the exact Colab output-frame origin")
    internal = (f"{current[2]}.{current[4]}-{current[3]}"
                ".c.codatalab-user-runtimes.internal:8007") if current else None
    return {"origin": f"https://{parsed.hostname}", "host": parsed.hostname, "port": actual_port,
            "internal_authority": internal,
            "frame_sources": ("'self' https://colab.research.google.com "
                              f"https://{frame.hostname}")}


def proxy_host_allowed(host, policy, peer):
    """Accept the public proxy or this runtime's exact internal forwarding authority."""
    try:
        authority = urlsplit("//" + host)
        if (authority.username is not None or authority.password is not None
                or authority.path or authority.query or authority.fragment):
            return False
        if authority.hostname == policy["host"]:
            return authority.port in {None, 443, policy["port"]}
        return (authority.netloc == policy.get("internal_authority")
                and ipaddress.ip_address(peer).is_loopback)
    except (TypeError, ValueError):
        return False


def stop_web(process):
    """Stop only the child process returned by start_web."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def start_web(command, *, cwd, output_dir, port, expected_identity, source_release,
              timeout_seconds=240):
    if not isinstance(command, list) or not command or timeout_seconds <= 0:
        raise ValueError("Provide an argument list and a positive startup timeout")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid web port")
    if set(expected_identity) != {
        "experiment_id", "model_id", "model_revision", "adapter_sha256",
    } or not all(expected_identity.values()):
        raise ValueError("Provide the complete frozen M1 identity")
    cwd, output_dir = Path(cwd).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(cwd) or cwd.is_relative_to(output_dir):
        raise ValueError("Web logs must not overlap the source checkout")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise RuntimeError("Web port is already in use; do not start a second model")
    except OSError:
        pass
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "demo_web_session_v1", "status": "failed", "port": port,
               "source_release": source_release, "expected_identity": expected_identity,
               "started_at": datetime.now(timezone.utc).isoformat()}
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    process = None
    try:
        with (output_dir / "server.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen([str(part) for part in command], cwd=cwd, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
        receipt["pid"] = process.pid
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Web process stopped; read server.log")
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/info", timeout=2) as response:
                    info = json.loads(response.read(65536))
            except (URLError, OSError, ValueError):
                time.sleep(1)
                continue
            if info.get("mode") != "live" or any(
                info.get(key) != value for key, value in expected_identity.items()
            ):
                raise RuntimeError("Web backend differs from the frozen M1 identity")
            receipt.update(status="ready", identity=info)
            return process, receipt
        raise TimeoutError("Model startup timed out; read server.log before retrying")
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        if process is not None:
            stop_web(process)
        raise
    finally:
        receipt["checked_at"] = datetime.now(timezone.utc).isoformat()
        with (output_dir / "receipt.json").open("x", encoding="utf-8") as saved:
            json.dump(receipt, saved, ensure_ascii=False, indent=2)
