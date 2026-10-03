#!/usr/bin/env python3
"""Unified OpenAI-compatible gateway in front of vLLM and NInfer-HIP.

Both engines run at the same time on the same host (one GPU each by default):

    vLLM (gfx906 branch)  : http://127.0.0.1:8000   -- throughput / batching
    ninfer-serve          : http://127.0.0.1:8080   -- low-latency single stream
    gateway               : http://0.0.0.0:9000     -- single client-facing port

Routing (env-tunable):
  * model name starting with "ninfer/"  -> NInfer-HIP (prefix stripped upstream)
  * NINFER_GATEWAY_DEFAULT=ninfer       -> NInfer-HIP for everything
  * otherwise                           -> vLLM
  * NINFER_GATEWAY_MIRROR=1             -> also send every request to the other
                                           backend in the background (A/B soak)

SSE streaming is proxied byte-for-byte, so both backends' stream formats pass
through unchanged. Standard library only — no pip dependencies.
"""

import json
import os
import sys
import threading
import urllib.error
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
NINFER_HOST = os.environ.get("NINFER_HOST", "127.0.0.1")
NINFER_PORT = int(os.environ.get("NINFER_PORT", "8080"))
DEFAULT_BACKEND = os.environ.get("NINFER_GATEWAY_DEFAULT", "vllm")
MIRROR = os.environ.get("NINFER_GATEWAY_MIRROR", "0") == "1"
GATEWAY_PORT = int(os.environ.get("NINFER_GATEWAY_PORT", "9000"))

BACKENDS = {
    "vllm": (VLLM_HOST, VLLM_PORT),
    "ninfer": (NINFER_HOST, NINFER_PORT),
}


def choose_backend(model: str) -> str:
    if model.startswith("ninfer/"):
        return "ninfer"
    if model.startswith("vllm/"):
        return "vllm"
    return "ninfer" if DEFAULT_BACKEND == "ninfer" else "vllm"


def strip_prefix(body: bytes, backend: str) -> bytes:
    """Removes the routing prefix so the backend sees the real model name."""
    try:
        payload = json.loads(body or b"{}")
    except Exception:
        return body
    model = payload.get("model")
    if isinstance(model, str):
        if model.startswith("ninfer/"):
            payload["model"] = model[len("ninfer/"):]
        elif model.startswith("vllm/"):
            payload["model"] = model[len("vllm/"):]
    return json.dumps(payload).encode()


def forward(method: str, path: str, body: bytes, headers: dict, backend: str):
    host, port = BACKENDS[backend]
    conn = HTTPConnection(host, port, timeout=3600)
    try:
        conn.putrequest(method, path, skip_accept_encoding=True)
        for k, v in headers.items():
            if k.lower() in ("host", "content-length", "connection",
                             "transfer-encoding"):
                continue
            conn.putheader(k, v)
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        if body:
            conn.send(body)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp
    except (OSError, urllib.error.URLError) as exc:  # backend down
        return 502, {"Content-Type": "application/json"}, None, str(exc)
    finally:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ninfer-gateway"

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, status: int, headers: dict, chunks):
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() in ("transfer-encoding", "connection",
                             "content-length"):
                continue
            self.send_header(k, v)
        if chunks is None:
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in chunks:
                if not chunk:
                    continue
                self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _mirror(self, backend: str, method: str, path: str, body: bytes,
                headers: dict):
        other = "ninfer" if backend == "vllm" else "vllm"

        def run():
            try:
                host, port = BACKENDS[other]
                conn = HTTPConnection(host, port, timeout=600)
                conn.request(method, path, body=body, headers=headers)
                conn.getresponse().read()
                conn.close()
            except Exception as exc:
                print(f"[mirror-> {other}] {exc}", file=sys.stderr)

        threading.Thread(target=run, daemon=True).start()

    def _proxy(self, method: str):
        body = self._read_body()
        model = ""
        try:
            model = json.loads(body or b"{}").get("model", "") or ""
        except Exception:
            model = ""
        backend = choose_backend(model)
        payload = strip_prefix(body, backend)
        headers = {k: v for k, v in self.headers.items()}

        if MIRROR and method == "POST" and path.endswith("/chat/completions"):
            self._mirror(backend, method, path, payload, headers)

        status, resp_headers, resp, err = None, {}, None, None
        try:
            out = forward(method, path, payload, headers, backend)
            if len(out) == 4:
                status, resp_headers, resp, err = out
            else:
                status, resp_headers, resp = out
        except Exception as exc:
            status, err = 502, str(exc)

        if resp is None:
            msg = json.dumps({"error": err or "backend unavailable",
                              "backend": backend}).encode()
            self.send_response(status or 502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        def stream():
            try:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    yield chunk
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

        self._send(status, resp_headers, stream())

    def do_GET(self):
        if self.path in ("/health", "/gateway/health"):
            msg = json.dumps({
                "gateway": "ok",
                "default_backend": DEFAULT_BACKEND,
                "vllm": f"http://{VLLM_HOST}:{VLLM_PORT}",
                "ninfer": f"http://{NINFER_HOST}:{NINFER_PORT}",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def log_message(self, fmt, *args):
        print(f"[gateway] {fmt % args}", file=sys.stderr)


def main() -> int:
    server = ThreadingHTTPServer(("0.0.0.0", GATEWAY_PORT), Handler)
    print(f"gateway on :{GATEWAY_PORT}  default={DEFAULT_BACKEND} "
          f"vllm=:{VLLM_PORT} ninfer=:{NINFER_PORT}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
