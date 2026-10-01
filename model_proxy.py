#!/usr/bin/env python3
"""Fast local reverse proxy for Codex Responses API model rewriting.

Design notes:
- One lightweight thread per connection; request/response streaming uses small
  chunks so SSE first-token latency stays low.
- Only JSON API paths are parsed. Other requests pass through byte-for-byte.
- Authorization and cookies are never logged.
- Logs are structured JSON lines with rotation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import logging.handlers
import socket
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 15731
UPSTREAM = "https://api.deepseek.com"
MODEL_MAP = {"gpt-5.6-luna": "deepseek-flash"}

# Small enough for prompt SSE deltas, large enough to avoid syscall overhead.
STREAM_CHUNK = 16 * 1024
# Parse only bodies that can plausibly be JSON API requests.
MAX_REWRITE_BODY = 8 * 1024 * 1024
SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key", "api-key", "x-auth-token"}
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

@dataclass(frozen=True)
class RequestStats:
    started: float

    @classmethod
    def start(cls) -> "RequestStats":
        return cls(time.monotonic())

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

def json_log(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

def parse_content_encoding(value: str | None) -> str | None:
    if not value:
        return None
    encoding = value.split(",", 1)[0].strip().lower()
    return encoding or None

def decode_body(body: bytes, encoding: str | None) -> tuple[bytes, str | None]:
    if encoding == "gzip":
        return gzip.decompress(body), "gzip"
    if encoding in {"deflate", "br", "zstd"}:
        # Do not attempt ad-hoc decompression for less common formats.
        return body, encoding
    return body, None

def encode_body(body: bytes, encoding: str | None) -> bytes:
    return gzip.compress(body) if encoding == "gzip" else body

def rewrite_body(body: bytes, model_map: dict[str, str]) -> tuple[bytes, str | None, str | None]:
    """Return (body, original_model, rewritten_model)."""
    try:
        obj = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, None, None
    if not isinstance(obj, dict):
        return body, None, None

    old = obj.get("model")
    if not isinstance(old, str):
        return body, None, None
    new = model_map.get(old)
    if new is None or new == old:
        return body, old, None

    obj["model"] = new
    # Compact separators are valid JSON and avoid unnecessary upload bytes.
    rewritten = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return rewritten, old, new

def forward_headers(headers) -> dict[str, str]:
    """Headers safe to forward: all except hop-by-hop fields."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "codex-model-proxy/2"
    sys_version = ""

    @property
    def upstream(self) -> str:
        return self.server.upstream  # type: ignore[attr-defined]

    @property
    def model_map(self) -> dict[str, str]:
        return self.server.model_map  # type: ignore[attr-defined]

    @property
    def proxy_logger(self) -> logging.Logger:
        return self.server.proxy_logger  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        # Replaced by structured request completion logs.
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/healthz", "/health"}:
            self.send_health()
        else:
            self.proxy("GET")

    def do_POST(self) -> None:
        self.proxy("POST")

    def do_PUT(self) -> None:
        self.proxy("PUT")

    def do_PATCH(self) -> None:
        self.proxy("PATCH")

    def do_DELETE(self) -> None:
        self.proxy("DELETE")

    def do_OPTIONS(self) -> None:
        self.proxy("OPTIONS")

    def do_HEAD(self) -> None:
        self.proxy("HEAD")

    def send_health(self) -> None:
        payload = json.dumps({
            "status": "ok",
            "upstream": self.upstream,
            "model_map": self.model_map,
        }, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def read_request_body(self) -> tuple[bytes, bool]:
        transfer = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" in transfer:
            # Read chunked request payloads without loading an unbounded stream.
            chunks: list[bytes] = []
            total = 0
            while True:
                size_line = self.rfile.readline(1024).strip()
                try:
                    size = int(size_line.split(b";", 1)[0], 16)
                except ValueError:
                    return b"", False
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(1024)
                        if trailer in {b"\r\n", b"\n", b""}:
                            break
                    break
                chunk = self.rfile.read(size)
                total += len(chunk)
                if total > MAX_REWRITE_BODY:
                    return b"", False
                chunks.append(chunk)
                self.rfile.read(2)
            return b"".join(chunks), True

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return b"", True
        if length > MAX_REWRITE_BODY:
            return b"", False
        return self.rfile.read(length), True

    def build_target(self) -> str:
        upstream_base = self.upstream.rstrip("/")
        request_path = self.path.lstrip("/")
        # Both Codex and the upstream conventionally include /v1. Normalize it
        # away if duplicated.
        if upstream_base.endswith("/v1") and request_path == "v1":
            return upstream_base
        if upstream_base.endswith("/v1") and request_path.startswith("v1/"):
            return upstream_base + "/" + request_path[len("v1/"):]
        return upstream_base + "/" + request_path

    def proxy(self, method: str) -> None:
        stats = RequestStats.start()
        old_model = new_model = None
        status = 500
        bytes_out = 0
        target = self.build_target()

        try:
            body, ok = self.read_request_body()
            if not ok:
                self.send_json_error(413, "request body too large or malformed chunked encoding")
                return

            encoding = parse_content_encoding(self.headers.get("Content-Encoding"))
            if body and encoding:
                body, _used = decode_body(body, encoding)

            if body:
                body, old_model, new_model = rewrite_body(body, self.model_map)
                if encoding:
                    body = encode_body(body, encoding)

            headers = forward_headers(self.headers)
            if body:
                headers["Content-Length"] = str(len(body))
            # Preserve Codex's Accept-Encoding exactly.

            req = Request(target, data=body if body else None, headers=headers, method=method)
            with urlopen(req, timeout=self.server.timeout) as upstream:  # type: ignore[attr-defined]
                status = upstream.status
                response_headers = [(k, v) for k, v in upstream.headers.items()
                                    if k.lower() not in HOP_BY_HOP]
                self.send_response(status)
                for k, v in response_headers:
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()

                if method == "HEAD":
                    return
                while True:
                    chunk = upstream.read(STREAM_CHUNK)
                    if not chunk:
                        break
                    bytes_out += len(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except HTTPError as exc:
            status = exc.code
            try:
                error_body = exc.read()
                error_headers = [(k, v) for k, v in exc.headers.items()
                                 if k.lower() not in HOP_BY_HOP]
            except Exception:
                error_body, error_headers = b"", []
            self.send_upstream_error(method, status, error_headers, error_body)

        except (URLError, socket.timeout, OSError) as exc:
            status = 502
            self.send_json_error(502, f"upstream connection failed: {exc}")

        except Exception as exc:
            status = 500
            self.proxy_logger.exception("proxy_internal_error")
            self.send_json_error(500, f"proxy failure: {exc}")

        finally:
            json_log(
                self.proxy_logger,
                logging.INFO if status < 500 else logging.ERROR,
                "request",
                method=method,
                path=self.path,
                status=status,
                elapsed_ms=stats.elapsed_ms(),
                bytes_out=bytes_out,
                model=old_model,
                model_rewritten_to=new_model,
            )

    def send_upstream_error(self, method: str, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        try:
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json_error(self, status: int, message: str) -> None:
        try:
            payload = json.dumps({"error": {"message": message}},
                                 separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, addr: tuple[str, int], upstream: str,
                 model_map: dict[str, str], timeout: float, logger: logging.Logger):
        super().__init__(addr, ProxyHandler)
        self.upstream = upstream
        self.model_map = model_map
        self.timeout = timeout
        self.proxy_logger = logger

def parse_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"invalid mapping {value!r}; expected from=to")
        src, dst = value.split("=", 1)
        if not src or not dst:
            raise argparse.ArgumentTypeError(f"invalid mapping {value!r}; both sides are required")
        result[src] = dst
    return result

def setup_logger(path: str, level: str, max_bytes: int, backups: int) -> logging.Logger:
    logger = logging.getLogger("codex_model_proxy")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if path:
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups
        )
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default=f"{LISTEN_HOST}:{LISTEN_PORT}")
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--map", action="append", default=[],
                        help="model rewrite, repeatable (example: gpt-5.6-luna=deepseek-flash)")
    parser.add_argument("--timeout", type=float, default=900,
                        help="upstream timeout in seconds (default: 900)")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--log-backups", type=int, default=5)
    args = parser.parse_args()

    host, port_text = args.listen.rsplit(":", 1)
    port = int(port_text)
    mappings = parse_map(args.map) if args.map else dict(MODEL_MAP)
    logger = setup_logger(args.log_file, args.log_level, args.log_max_bytes, args.log_backups)

    server = ProxyServer((host, port), args.upstream, mappings, args.timeout, logger)
    json_log(logger, logging.INFO, "startup", listen=f"{host}:{port}",
             upstream=args.upstream, model_map=mappings)
    print(f"codex-model-proxy listening on http://{host}:{port}", flush=True)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        json_log(logger, logging.INFO, "shutdown")
    return 0

if __name__ == "__main__":
    sys.exit(main())
