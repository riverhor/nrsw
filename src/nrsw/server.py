"""Serve the demo and run configured image predictions in memory."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from threading import BoundedSemaphore, Lock
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from .demo import WEB, browser_data, demo_html, specimen_browser_data
from .extraction import ImageAlignmentError, InvalidImageError


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_SECONDS = 20
IMAGE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG"}


def serve(data, port=8000, *, registry=None, layout_id=None, allowed_origin=None):
    if registry is not None:
        fprs = registry.available_fprs(layout_id)
        if not fprs:
            raise ValueError("The upload layout needs at least one calibrated threshold")
    if allowed_origin is not None:
        origin = urlsplit(allowed_origin)
        if (origin.scheme != "https" or not origin.hostname or origin.username is not None
                or origin.password is not None or origin.path or origin.query or origin.fragment):
            raise ValueError("Use one exact HTTPS origin for --allow-origin")
    incoming = BoundedSemaphore(4)
    inference = Lock()
    types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
             ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
    assets = {f"/{p.name}": (types[p.suffix], p.read_bytes())
              for p in WEB.iterdir() if p.is_file() and p.suffix in types and not p.name.startswith("test")}
    specimen_types = {".json": "application/json; charset=utf-8", ".png": "image/png",
                      ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    specimens = WEB / "specimens"
    if specimens.is_dir():
        assets.update({f"/specimens/{p.name}": (specimen_types[p.suffix], p.read_bytes())
                       for p in specimens.iterdir() if p.is_file() and p.suffix in specimen_types})
    if registry is not None:
        page = demo_html(assets["/index.html"][1].decode("utf-8"), ".")
        assets["/index.html"] = (types[".html"], page.encode("utf-8"))
    assets["/"] = assets["/index.html"]
    assets["/examples.json"] = ("application/json; charset=utf-8",
                                json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    assets["/examples.js"] = (types[".js"], browser_data(data).encode("utf-8"))
    assets["/specimens/examples.js"] = (types[".js"], specimen_browser_data().encode("utf-8"))

    class DemoHandler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(MAX_UPLOAD_SECONDS)
            super().setup()

        def _accepted_origin(self):
            origin = self.headers.get("Origin")
            port = self.server.server_port
            return origin is None or origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}", allowed_origin}

        def _headers(self, status, content_type, length, *, api=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if api and allowed_origin is not None:
                self.send_header("Vary", "Origin")
                if self.headers.get("Origin") == allowed_origin:
                    self.send_header("Access-Control-Allow-Origin", allowed_origin)
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _api(self, status, body, *, include_body=True):
            payload = json.dumps(body, allow_nan=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(payload), api=True)
            if include_body:
                self.wfile.write(payload)

        def _send(self, include_body):
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                if not self._accepted_origin():
                    self._api(403, {"error": "unavailable"}, include_body=include_body)
                elif registry is None:
                    self._api(503, {"error": "unavailable"}, include_body=include_body)
                else:
                    self._api(200, {"layout": layout_id, "max_bytes": MAX_UPLOAD_BYTES,
                                    "supported_types": list(IMAGE_FORMATS), "fprs": fprs}, include_body=include_body)
                return
            asset = assets.get(path)
            if asset is None:
                self.send_error(404)
                return
            content_type, payload = asset
            self._headers(200, content_type, len(payload))
            if include_body:
                self.wfile.write(payload)

        def do_GET(self):
            self._send(True)

        def do_HEAD(self):
            self._send(False)

        def do_OPTIONS(self):
            if self.path.split("?", 1)[0] != "/api/predict" or not self._accepted_origin():
                self._api(403, {"error": "unavailable"})
                return
            self._headers(204, "application/json; charset=utf-8", 0, api=True)

        def _read_image(self, length):
            deadline = monotonic() + MAX_UPLOAD_SECONDS
            image = BytesIO()
            while length:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.request.settimeout(remaining)
                block = self.rfile.read1(min(length, 64 * 1024))
                if not block:
                    raise InvalidImageError("Upload ended early")
                image.write(block)
                length -= len(block)
            self.request.settimeout(MAX_UPLOAD_SECONDS)
            return image.getvalue()

        def do_POST(self):
            path, _, query = self.path.partition("?")
            if path != "/api/predict":
                return self._api(404, {"error": "unavailable"})
            if not self._accepted_origin():
                return self._api(403, {"error": "unavailable"})
            if registry is None:
                return self._api(503, {"error": "unavailable"})
            values = parse_qs(query, keep_blank_values=True).get("fpr", ["0.01"])
            try:
                fpr, = values
                fpr = float(fpr)
                if fpr not in fprs:
                    raise ValueError("unsupported target")
            except ValueError:
                return self._api(400, {"error": "invalid_fpr"})
            image_format = IMAGE_FORMATS.get(self.headers.get_content_type())
            if image_format is None:
                return self._api(415, {"error": "unsupported_type"})
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal() or self.headers.get("Transfer-Encoding"):
                return self._api(400, {"error": "invalid_image"})
            if len(lengths[0]) > len(str(MAX_UPLOAD_BYTES)) or int(lengths[0]) > MAX_UPLOAD_BYTES:
                return self._api(413, {"error": "too_large"})
            length = int(lengths[0])
            if not length:
                return self._api(400, {"error": "invalid_image"})
            if not incoming.acquire(blocking=False):
                return self._api(429, {"error": "busy"})
            try:
                image = self._read_image(length)
                if not inference.acquire(blocking=False):
                    return self._api(429, {"error": "busy"})
                try:
                    result = registry.explain_upload(layout_id, image, image_format=image_format, fpr=fpr)
                finally:
                    inference.release()
            except (InvalidImageError, TimeoutError):
                return self._api(400, {"error": "invalid_image"})
            except ImageAlignmentError:
                return self._api(422, {"error": "alignment_failed"})
            except (ArithmeticError, ImportError, OSError, RuntimeError, ValueError):
                return self._api(500, {"error": "server_error"})
            finally:
                incoming.release()
            self._api(200, result)

        def log_message(self, *args):
            return

    with ThreadingHTTPServer(("127.0.0.1", port), DemoHandler) as server:
        print(f"NRSW demo: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
