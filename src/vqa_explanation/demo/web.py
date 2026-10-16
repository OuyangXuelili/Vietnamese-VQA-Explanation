"""Private HTTP interface for the demo, not for annotation or scoring."""

import asyncio
import base64
import binascii
import json
import logging
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from vqa_explanation.demo.service import MAX_IMAGE_BYTES, DemoError
from vqa_explanation.utils.environment import project_root

MAX_REQUEST_BYTES = 9 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def create_app(service, *, auth: tuple[str, str] | None = None, timeout_seconds: float = 90,
               static_dir: Path | None = None) -> FastAPI:
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")
    if auth is not None and (len(auth) != 2 or not all(auth)):
        raise ValueError("Auth needs a username and password")
    static = static_dir or project_root() / "app/static"
    app = FastAPI(title="Vietnamese VQA", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if auth:
            try:
                scheme, token = request.headers.get("authorization", "").split(" ", 1)
                user, password = base64.b64decode(token, validate=True).decode().split(":", 1)
                permitted = scheme.lower() == "basic"
                permitted &= secrets.compare_digest(user.encode(), auth[0].encode())
                permitted &= secrets.compare_digest(password.encode(), auth[1].encode())
            except (ValueError, UnicodeError, binascii.Error):
                permitted = False
            if not permitted:
                return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="VQA"'})
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
        # Prevent unauthenticated localhost access through DNS rebinding.
        if not auth and request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return Response(status_code=403)
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; "
                                       "script-src 'self'; style-src 'self'; "
                                       "frame-ancestors 'none'",
        })
        return response

    @app.exception_handler(DemoError)
    async def demo_error(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"demo.js", "demo.css"}:
            raise DemoError("Không có tệp này.", 404)
        return FileResponse(static / name)

    @app.get("/api/info")
    def info():
        return service.info()

    @app.get("/api/examples/{qid}")
    def example(qid: str):
        if service.mode != "replay":
            raise DemoError("Không dùng kết quả đã lưu ở chế độ live.", 404)
        return service.example(qid)

    @app.get("/api/examples/{qid}/image")
    def example_image(qid: str):
        if service.mode != "replay":
            raise DemoError("Không có ảnh ví dụ.", 404)
        return Response(service.image(qid), media_type="image/png")

    @app.post("/api/predict")
    async def predict(request: Request):
        if service.mode != "live":
            raise DemoError("Bản xem lại không trả lời ảnh mới. Cần backend GPU.", 409)
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise DemoError("Require application/json", 415)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                raise DemoError("Dữ liệu yêu cầu quá lớn.", 413)
        try:
            payload = json.loads(body)
            if (not isinstance(payload, dict) or set(payload) != {"question", "image_base64"}
                    or not isinstance(payload["image_base64"], str)):
                raise ValueError("Invalid fields")
            data = base64.b64decode(payload["image_base64"], validate=True)
        except (ValueError, UnicodeError, binascii.Error) as error:
            raise DemoError("Dữ liệu ảnh hoặc câu hỏi không hợp lệ.") from error
        if len(data) > MAX_IMAGE_BYTES:
            raise DemoError("Ảnh không quá 6 MB.", 413)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(service.predict, data, payload["question"]), timeout_seconds)
        except asyncio.TimeoutError as error:
            # The worker retains its lock until generation actually finishes.
            raise DemoError("Quá thời gian chờ. GPU có thể vẫn đang xử lý; không gửi liên tục.",
                            504) from error
        except DemoError:
            raise
        except Exception as error:
            LOGGER.warning("Demo inference failed: %s", type(error).__name__)
            raise DemoError("Backend chưa trả lời được. Kiểm phiên GPU rồi thử lại.",
                            503) from error
        return result.to_dict()

    return app
