#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.115,<1", "httpx>=0.27,<1", "PyYAML>=6,<7", "uvicorn>=0.30,<1"]
# ///
"""Small OpenAI/Anthropic-compatible gateway. Run: uv run uvicorn model_proxy:app --host 127.0.0.1 --port 15731."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import secrets
import zlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


CONFIG_PATH = Path(os.getenv("MODEL_MAPPER_CONFIG", "model_proxy_config.yaml"))
# config.yaml
"""
default_upstream: openai

upstreams:
  openai:
    base_url: https://api.openai.com/v1
    api_key: sk-xxxx
  deepseek:
    base_url: https://api.deepseek.com/v1
    api_key: sk-xxxx

models:
  my_dpsk:
    upstream: deepseek
    model: deepseek-flash

# /v1/models returns virtual models immediately. Upstream models are refreshed
# in the background and retained for this many seconds.
model_cache:
  enabled: true
  ttl_seconds: 1800
  request_timeout_seconds: 10

# Optional authentication for clients of this proxy.
auth:
  api_key: sk-xxxx
"""


class LogFormatter(logging.Formatter):
    """Pad the level like uvicorn does so both kinds of line start at the same column."""

    def format(self, record: logging.LogRecord) -> str:
        record.levelprefix = f"{record.levelname}:".ljust(10)
        return super().format(record)


logger = logging.getLogger("model_proxy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(LogFormatter("%(levelprefix)s%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(os.getenv("MODEL_MAPPER_LOG_LEVEL", "INFO").upper())
    # Do not print twice once uvicorn takes over the root logger.
    logger.propagate = False


def decompress_body(content: bytes, encoding: str | None) -> bytes:
    """Best-effort undo of Content-Encoding, for logging only.

    Bodies are streamed through untouched, so what we hold is still compressed;
    logging those bytes raw makes journald store "[NNB blob data]" instead of text
    and hides upstream errors. Returns `content` itself when it cannot decode.
    """
    if not encoding:
        return content
    try:
        if encoding.lower() in ("gzip", "x-gzip"):
            return gzip.decompress(content)
        if encoding.lower() == "deflate":
            try:
                return zlib.decompress(content)
            except zlib.error:
                # Servers disagree on whether deflate means zlib or a raw stream.
                return zlib.decompress(content, -zlib.MAX_WBITS)
    except (OSError, EOFError, zlib.error):
        return content
    return content


def body_snippet(content: bytes, limit: int = 300, encoding: str | None = None) -> str:
    raw = decompress_body(content, encoding)
    text = raw[:limit].decode("utf-8", "replace")
    # Control bytes (NUL, newlines) are what make journalctl print a blob marker.
    text = " ".join("".join(char if char.isprintable() else " " for char in text).split())
    if len(raw) > limit:
        text += "..."
    if raw is content and encoding:
        text = f"[{encoding} encoded] {text}"
    return text


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise RuntimeError("Configuration root must be a mapping")

    upstreams = config.get("upstreams")
    default_upstream = config.get("default_upstream")
    models = config.get("models") or {}
    auth = config.get("auth") or {}
    cache = config.get("model_cache") or {}
    if not isinstance(upstreams, dict) or not upstreams:
        raise RuntimeError("upstreams must define at least one upstream")
    if default_upstream not in upstreams:
        raise RuntimeError(f"default_upstream does not exist: {default_upstream}")
    if not isinstance(models, dict):
        raise RuntimeError("models must be a mapping")
    if not isinstance(auth, dict) or ("api_key" in auth and not isinstance(auth["api_key"], str)):
        raise RuntimeError("auth must be a mapping with an optional string api_key")
    if not isinstance(cache, dict):
        raise RuntimeError("model_cache must be a mapping")
    ttl = cache.get("ttl_seconds", 1800)
    timeout = cache.get("request_timeout_seconds", 10)
    if not isinstance(ttl, (int, float)) or ttl <= 0:
        raise RuntimeError("model_cache.ttl_seconds must be positive")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise RuntimeError("model_cache.request_timeout_seconds must be positive")

    watch = config.get("config_watch") or {}
    if not isinstance(watch, dict):
        raise RuntimeError("config_watch must be a mapping")
    for key, default in (("interval_seconds", 1), ("settle_seconds", 0.5)):
        value = watch.get(key, default)
        if not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"config_watch.{key} must be positive")

    for name, upstream in upstreams.items():
        if not isinstance(upstream, dict) or not upstream.get("base_url"):
            raise RuntimeError(f"Upstream {name!r} must define base_url")
        if "api_key" not in upstream:
            raise RuntimeError(f"Upstream {name!r} must define api_key")
    for exposed_model, mapping in models.items():
        if not isinstance(mapping, dict):
            raise RuntimeError(f"Model mapping {exposed_model!r} must be a mapping")
        if mapping.get("upstream") not in upstreams:
            raise RuntimeError(f"Model mapping {exposed_model!r} references unknown upstream")
        if not mapping.get("model"):
            raise RuntimeError(f"Model mapping {exposed_model!r} must define model")
    return config


config = load_config(CONFIG_PATH)
upstream_models: dict[str, list[dict[str, Any]]] = {}
model_cache_ready = asyncio.Event()
model_cache_task: asyncio.Task[None] | None = None
config_watch_task: asyncio.Task[None] | None = None


def cache_enabled(cfg: dict[str, Any] | None = None) -> bool:
    active = config if cfg is None else cfg
    return (active.get("model_cache") or {}).get("enabled", True) is not False


def model_object(model_id: str, owned_by: str) -> dict[str, str]:
    return {"id": model_id, "object": "model", "owned_by": owned_by}


def unique_models() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for model_id in (config.get("models") or {}):
        if model_id not in seen:
            result.append(model_object(model_id, "model-mapper"))
            seen.add(model_id)
    for models in upstream_models.values():
        for item in models:
            model_id = item["id"]
            if model_id not in seen:
                result.append(item)
                seen.add(model_id)
    return result


async def refresh_upstream_models() -> None:
    cache = config.get("model_cache") or {}
    timeout = float(cache.get("request_timeout_seconds", 10))
    async with httpx.AsyncClient(timeout=timeout) as client:
        for upstream_name, upstream in config["upstreams"].items():
            try:
                response = await client.get(
                    f"{upstream['base_url'].rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {upstream['api_key']}"},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                upstream_models[upstream_name] = [
                    model_object(item["id"], upstream_name)
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                logger.info(
                    "cached /v1/models %s: %s",
                    upstream_name,
                    ", ".join(item["id"] for item in upstream_models[upstream_name]) or "(empty)",
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                # Keep the previous cache when an upstream is temporarily unavailable.
                logger.warning("upstream %s /v1/models refresh failed: %s", upstream_name, exc)
                continue


async def model_cache_loop() -> None:
    while True:
        await refresh_upstream_models()
        model_cache_ready.set()
        # Re-read each round so a reload can retune the refresh interval.
        interval = float((config.get("model_cache") or {}).get("ttl_seconds", 1800))
        await asyncio.sleep(interval)


async def sync_model_cache_task() -> None:
    """Start or stop the /v1/models refresher so it matches the current config."""
    global model_cache_task
    if cache_enabled():
        if model_cache_task is None or model_cache_task.done():
            model_cache_task = asyncio.create_task(model_cache_loop())
        return
    if model_cache_task is not None and not model_cache_task.done():
        model_cache_task.cancel()
        await asyncio.gather(model_cache_task, return_exceptions=True)
    model_cache_task = None
    model_cache_ready.set()


def config_stamp() -> tuple[int, int] | None:
    try:
        stat = CONFIG_PATH.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


async def config_watch_loop() -> None:
    """Reload the YAML in place, so upstreams/models/auth edits need no restart."""
    global config
    stamp = config_stamp()
    while True:
        watch = config.get("config_watch") or {}
        interval = float(watch.get("interval_seconds", 1))
        settle = float(watch.get("settle_seconds", 0.5))
        await asyncio.sleep(interval)
        current = config_stamp()
        if current is None or current == stamp:
            continue
        stamp = current
        # Editors truncate and rewrite in place, so wait for the file to settle.
        await asyncio.sleep(settle)
        settled = config_stamp()
        if settled != current:
            stamp = settled
            continue
        try:
            reloaded = load_config(CONFIG_PATH)
        except (OSError, RuntimeError, yaml.YAMLError) as exc:
            logger.error("config reload rejected, keeping the running config: %s", exc)
            continue
        config = reloaded
        logger.info("config reloaded from %s", CONFIG_PATH)
        await sync_model_cache_task()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global config_watch_task
    await sync_model_cache_task()
    if (config.get("config_watch") or {}).get("enabled", True) is not False:
        config_watch_task = asyncio.create_task(config_watch_loop())
    yield
    for task in (config_watch_task, model_cache_task):
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


app = FastAPI(title="Model Mapper", lifespan=lifespan)


async def require_auth(request: Request) -> None:
    expected = (config.get("auth") or {}).get("api_key")
    if not expected:
        return
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        # Anthropic clients (Claude Code, the Anthropic SDKs) authenticate with x-api-key.
        supplied = request.headers.get("x-api-key", "")
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing API key", {"WWW-Authenticate": "Bearer"})


def model_owner(model_id: str, cfg: dict[str, Any]) -> str | None:
    """Upstream whose /v1/models listing contains this id, preferring default_upstream."""
    default = cfg["default_upstream"]
    order = [default, *(name for name in cfg["upstreams"] if name != default)]
    for name in order:
        if any(item["id"] == model_id for item in upstream_models.get(name, [])):
            return name
    return None


def resolve_model(requested: str, cfg: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve a requested model into (upstream name, model to send, how it was routed)."""
    mapping = (cfg.get("models") or {}).get(requested)
    if mapping:
        return mapping["upstream"], mapping["model"], "mapping"
    # /v1/models advertises every upstream's own models, so an id served by a single
    # upstream must be routed there instead of straight to default_upstream. Only ids we
    # have never seen (cache still empty, disabled, or an upstream that is down) fall back.
    owner = model_owner(requested, cfg)
    if owner:
        return owner, requested, "owner"
    return cfg["default_upstream"], requested, "default"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def models() -> dict[str, Any]:
    return {"object": "list", "data": unique_models()}


ANTHROPIC_DEFAULT_VERSION = "2023-06-01"


def upstream_request_headers(request: Request, upstream: dict[str, Any], style: str) -> dict[str, str]:
    """Headers for the upstream call.

    Both styles send Authorization, which every OpenAI-compatible upstream accepts, so
    an aggregator that speaks both protocols keeps working. The anthropic style adds the
    Messages API headers the client sent: x-api-key is the documented credential and
    anthropic-version pins the protocol revision between client and upstream.
    """
    headers = {
        "Authorization": f"Bearer {upstream['api_key']}",
        "Content-Type": "application/json",
        "Accept-Encoding": request.headers.get("accept-encoding", "identity"),
    }
    if style == "anthropic":
        headers["x-api-key"] = upstream["api_key"]
        headers["anthropic-version"] = request.headers.get("anthropic-version", ANTHROPIC_DEFAULT_VERSION)
        beta = request.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
    return headers


async def proxy_endpoint(request: Request, endpoint: str, style: str = "openai"):
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Request body must be JSON") from exc
    requested = body.get("model")
    if not isinstance(requested, str) or not requested:
        raise HTTPException(400, "Request must include a model")
    # Snapshot: a hot reload can swap the module global while this request awaits.
    cfg = config
    # Do not race the initial /v1/models refresh: before the cache is populated,
    # owner-routed models would otherwise silently fall back to default_upstream.
    explicit_mapping = (cfg.get("models") or {}).get(requested)
    if cache_enabled(cfg) and explicit_mapping is None:
        await model_cache_ready.wait()
    upstream_name, target, route = resolve_model(requested, cfg)
    if route == "default":
        # A model the default upstream does not serve comes back as an opaque 4xx from
        # upstream; say so instead of leaving the fallback silent.
        logger.warning(
            "model %s is not mapped and no upstream advertises it; falling back to %s",
            requested,
            upstream_name,
        )
    upstream = cfg["upstreams"][upstream_name]
    payload = dict(body, model=target)
    streaming = bool(payload.get("stream"))
    label = f"{request.method} {request.url.path} model={requested} -> {upstream_name}/{target}"
    logger.info("%s stream=%s route=%s", label, streaming, route)
    client = httpx.AsyncClient(timeout=upstream.get("timeout", 120.0))
    try:
        response = await client.send(
            client.build_request(
                "POST",
                f"{upstream['base_url'].rstrip('/')}/{endpoint}",
                headers=upstream_request_headers(request, upstream, style),
                json=payload,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        logger.error("%s upstream request failed: %s", label, exc)
        raise HTTPException(502, f"Upstream request failed: {exc}") from exc

    headers = {}
    encoding = response.headers.get("content-encoding")
    if encoding:
        headers["Content-Encoding"] = encoding
    media_type = response.headers.get("content-type", "application/json")
    status = response.status_code

    if not payload.get("stream"):
        content = b"".join([chunk async for chunk in response.aiter_raw()])
        await response.aclose()
        await client.aclose()
        if status >= 400:
            logger.warning("%s status=%s body=%s", label, status, body_snippet(content, encoding=encoding))
        return Response(content=content, status_code=status, media_type=media_type, headers=headers)

    if status >= 400:
        # The upstream error body is streamed through untouched, so only the status is known.
        logger.warning("%s status=%s (upstream error, body not logged for streams)", label, status)

    async def body_stream() -> AsyncIterator[bytes]:
        sent = 0
        try:
            async for chunk in response.aiter_raw():
                sent += len(chunk)
                yield chunk
        except httpx.HTTPError as exc:
            logger.error("%s upstream stream failed after bytes=%d: %s", label, sent, exc)
            raise
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(body_stream(), status_code=status, media_type=media_type, headers=headers)


@app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
async def chat_completions(request: Request):
    return await proxy_endpoint(request, "chat/completions")


@app.post("/v1/responses", dependencies=[Depends(require_auth)])
async def responses(request: Request):
    return await proxy_endpoint(request, "responses")


@app.post("/v1/messages", dependencies=[Depends(require_auth)])
async def messages(request: Request):
    """Anthropic Messages API passthrough, for Claude Code and the Anthropic SDKs."""
    return await proxy_endpoint(request, "messages", style="anthropic")


@app.post("/v1/messages/count_tokens", dependencies=[Depends(require_auth)])
async def messages_count_tokens(request: Request):
    return await proxy_endpoint(request, "messages/count_tokens", style="anthropic")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("MODEL_MAPPER_HOST", "127.0.0.1"),
        port=int(os.getenv("MODEL_MAPPER_PORT", "15731")),
    )
