"""Serve the demo and its read-only results on the loopback interface."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

from .demo import WEB, browser_data, specimen_browser_data


def serve(data, port=8000):
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
    assets["/"] = assets["/index.html"]
    assets["/examples.json"] = ("application/json; charset=utf-8",
                                json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    assets["/examples.js"] = (types[".js"], browser_data(data).encode("utf-8"))
    assets["/specimens/examples.js"] = (types[".js"], specimen_browser_data().encode("utf-8"))

    class DemoHandler(BaseHTTPRequestHandler):
        def _send(self, include_body):
            asset = assets.get(self.path.split("?", 1)[0])
            if asset is None:
                self.send_error(404)
                return
            content_type, payload = asset
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            if include_body:
                self.wfile.write(payload)

        def do_GET(self):
            self._send(True)

        def do_HEAD(self):
            self._send(False)

        def log_message(self, *args):
            return

    with ThreadingHTTPServer(("127.0.0.1", port), DemoHandler) as server:
        print(f"NRSW demo: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
