"""Serve the Slovak ID detector through the Hugging Face Spaces queue."""

import base64
from binascii import Error as Base64Error
import os
from pathlib import Path

import spaces
import gradio as gr
from fastapi.exceptions import RequestValidationError
from gradio.route_utils import CustomCORSMiddleware
from gradio.routes import App
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from nrsw.extraction import ImageAlignmentError, InvalidImageError
from nrsw.registry import LayoutRegistry
from nrsw.server import IMAGE_FORMATS, MAX_UPLOAD_BYTES


LAYOUT_ID = "IDSpace_SVK_R29"
MAX_BASE64_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3)
MAX_REQUEST_BYTES = MAX_BASE64_BYTES + 1024
QUEUE_PREFIX = "/gradio_api/call/"
CALLS = {QUEUE_PREFIX + name for name in ("status", "predict")}
PAGES_ORIGIN = "https://riverhor.github.io"


class PagesApp(App):
    """Keep only the explicit CORS policy when Gradio adds its default middleware."""

    def add_middleware(self, middleware_class, *args, **kwargs):
        if middleware_class is not CustomCORSMiddleware:
            super().add_middleware(middleware_class, *args, **kwargs)


class UploadLimits:
    """Restrict payloads to the JSON queue and disable Gradio's file upload routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path, method = scope["path"], scope["method"]
        if method == "POST":
            if path not in CALLS:
                await JSONResponse({"error": "unavailable"}, status_code=404)(scope, receive, send)
                return
            lengths = [value for name, value in scope["headers"] if name == b"content-length"]
            if len(lengths) != 1 or not lengths[0].isdigit():
                await JSONResponse({"error": "invalid_image"}, status_code=400)(scope, receive, send)
                return
            limit = 1024 if path.endswith("/status") else MAX_REQUEST_BYTES
            if len(lengths[0]) > len(str(limit)) or int(lengths[0]) > limit:
                await JSONResponse({"error": "too_large"}, status_code=413)(scope, receive, send)
                return

        async def send_no_store(message):
            if message["type"] == "http.response.start":
                headers = [(key, value) for key, value in message["headers"] if key != b"cache-control"]
                message = {**message, "headers": [*headers, (b"cache-control", b"no-store")]}
            await send(message)

        await self.app(scope, receive, send_no_store if path.startswith(QUEUE_PREFIX) else send)


def build_demo(registry):
    fprs = registry.available_fprs(LAYOUT_ID)
    if not fprs:
        raise ValueError("The upload layout needs a calibrated threshold")

    def status() -> dict:
        """Return the layout, upload formats and false-positive rate targets."""
        return {"layout": LAYOUT_ID, "max_bytes": MAX_UPLOAD_BYTES,
                "supported_types": list(IMAGE_FORMATS), "fprs": fprs}

    @spaces.GPU(duration=30)
    def predict(encoded_image: str, mime_type: str, fpr: float) -> dict:
        """Score one JPEG or PNG supplied as bare base64 and return local evidence."""
        image_format = IMAGE_FORMATS.get(mime_type)
        if image_format is None:
            return {"error": "unsupported_type"}
        if fpr not in fprs:
            return {"error": "invalid_fpr"}
        if not encoded_image:
            return {"error": "invalid_image"}
        if len(encoded_image) > MAX_BASE64_BYTES:
            return {"error": "too_large"}
        try:
            image = base64.b64decode(encoded_image, validate=True)
        except (Base64Error, ValueError):
            return {"error": "invalid_image"}
        if len(image) > MAX_UPLOAD_BYTES:
            return {"error": "too_large"}
        try:
            return registry.explain_upload(LAYOUT_ID, image, image_format=image_format, fpr=fpr)
        except InvalidImageError:
            return {"error": "invalid_image"}
        except ImageAlignmentError:
            return {"error": "alignment_failed"}
        except (ArithmeticError, ImportError, OSError, RuntimeError, ValueError):
            return {"error": "server_error"}

    with gr.Blocks(analytics_enabled=False, title="NRSW document detection") as demo:
        gr.Markdown("# NRSW document detection\n[Open the public demo](https://riverhor.github.io/nrsw/).")
        status_button = gr.Button(visible=False)
        predict_button = gr.Button(visible=False)
        image_input = gr.Textbox(label="Image as base64", visible=False)
        format_input = gr.Textbox(label="Image media type", visible=False)
        fpr_input = gr.Number(label="Target false-positive rate", visible=False)
        status_button.click(status, outputs=gr.JSON(visible=False), api_name="status")
        predict_button.click(predict, inputs=[image_input, format_input, fpr_input],
                             outputs=gr.JSON(visible=False), api_name="predict", concurrency_limit=1)
    return demo.queue(max_size=6, api_open=False)


def launch(demo, **options):
    async def invalid_request(_request, _error):
        return JSONResponse({"error": "invalid_image"}, status_code=422)

    service = PagesApp(middleware=[
        Middleware(CORSMiddleware, allow_origins=[PAGES_ORIGIN], allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"]),
        Middleware(UploadLimits),
    ], exception_handlers={RequestValidationError: invalid_request})
    # The Space's ZeroGPU runtime installs its startup hook on Blocks.launch.
    return demo.launch(_app=service, share=False, show_error=False, enable_monitoring=False,
                       ssr_mode=False, pwa=False, run_history=False, **options)


def main():
    layout_path = Path(os.environ.get("NRSW_LAYOUTS", Path(__file__).with_name("detector") / "layouts.json"))
    with LayoutRegistry.from_file(layout_path, device="cuda", precision="fp16", allow_download=True,
                                  allow_zerogpu_torch=True) as registry:
        # ZeroGPU packs these tensors at startup and restores them in the GPU worker.
        registry.prepare(LAYOUT_ID)
        launch(build_demo(registry), server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
