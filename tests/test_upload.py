from contextlib import closing
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from queue import Queue
from threading import Thread

from nrsw import server
from nrsw.extraction import ImageAlignmentError, InvalidImageError


def test_upload_api_handles_an_image_without_writing_it_to_disk(monkeypatch):
    class Registry:
        def available_fprs(self, layout):
            assert layout == "svk"
            return [0.01, 0.05]

        def explain_upload(self, layout, image, *, image_format, fpr):
            assert layout == "svk"
            assert image_format == "PNG"
            assert fpr == 0.05
            if image == b"invalid":
                raise InvalidImageError("invalid image")
            if image == b"unaligned":
                raise ImageAlignmentError("wrong layout")
            assert image == b"image data"
            return {"dataset": "custom", "selected_fpr": 0.05, "examples": [{"score": 2, "flagged": True}]}

    started = Queue()

    def bind(*args):
        instance = ThreadingHTTPServer(*args)
        started.put(instance)
        return instance

    origin = "https://riverhor.github.io"
    monkeypatch.setattr(server, "ThreadingHTTPServer", bind)
    thread = Thread(target=server.serve, args=({}, 0),
                    kwargs={"registry": Registry(), "layout_id": "svk", "allowed_origin": origin}, daemon=True)
    thread.start()
    instance = started.get(timeout=5)

    def request(method, path, *, body=None, **headers):
        with closing(HTTPConnection("127.0.0.1", instance.server_port, timeout=5)) as connection:
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            payload = response.read()
            return response.status, response.headers, json.loads(payload) if payload else None

    try:
        status, headers, data = request("GET", "/api/status", Origin=origin)
        assert status == 200
        assert headers["Access-Control-Allow-Origin"] == origin
        assert headers["Cache-Control"] == "no-store"
        assert data == {"layout": "svk", "max_bytes": 8_388_608,
                        "supported_types": ["image/jpeg", "image/png"], "fprs": [0.01, 0.05]}
        assert request("OPTIONS", "/api/predict", Origin=origin)[0] == 204
        assert request("OPTIONS", "/api/predict", Origin="https://example.org")[0] == 403
        path = "/api/predict?fpr=0.05"
        request_headers = {"Origin": origin, "Content-Type": "image/png"}
        status, _, result = request("POST", path, body=b"image data", **request_headers)
        assert status == 200 and result["examples"][0]["flagged"]
        cases = (
            (path, b"invalid", request_headers, 400, "invalid_image"),
            (path, b"unaligned", request_headers, 422, "alignment_failed"),
            ("/api/predict?fpr=0.99", b"image data", request_headers, 400, "invalid_fpr"),
            (path, b"image data", {"Content-Type": "application/pdf"}, 415, "unsupported_type"),
            (path, b"image data", {**request_headers, "Content-Length": "8388609"}, 413, "too_large"),
        )
        for path, body, headers, expected_status, error in cases:
            status, _, response = request("POST", path, body=body, **headers)
            assert status == expected_status and response == {"error": error}
    finally:
        instance.shutdown()
        thread.join(timeout=5)
