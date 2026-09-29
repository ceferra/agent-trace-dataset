#!/usr/bin/env python3
"""Edge adapter between an OpenAI-style client (OpenClaw/Pi) and Azure OpenAI.

Azure OpenAI differs from the stock OpenAI / vLLM wire protocol in three ways
that a client configured for a single ``baseUrl`` cannot express:

  * Routing is by **deployment name in the URL path**, not by ``model`` in the
    body:  ``{endpoint}/openai/deployments/<deployment>/chat/completions``.
  * It needs an ``api-version`` query parameter.
  * It authenticates with an ``api-key`` header, not ``Authorization: Bearer``.

This proxy listens on ``--listen-port`` and rewrites incoming requests so the
client can keep talking plain OpenAI to ``http://127.0.0.1:<port>/v1``:

    POST /v1/chat/completions   body model="<deployment>"
      -> POST {endpoint}/openai/deployments/<deployment>/chat/completions
              ?api-version=<ver>
         header: api-key: $AZURE_OPENAI_KEY

Reasoning deployments (o-series and gpt-5 family, detected by name) require
``max_completion_tokens`` instead of ``max_tokens`` and reject the sampler
fields (temperature/top_p/penalties), so those are adjusted per request.

No model weights and no tool bridge: Azure GPT/o-series support native tool
calling, so requests pass through otherwise unchanged. The key is supplied only
via the ``AZURE_OPENAI_KEY`` environment variable and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Upstream statuses that are transient and worth retrying with backoff.
_RETRY_STATUS = (429, 500, 502, 503, 504)

# Sampler / decoding fields rejected by Azure reasoning deployments.
_REASONING_DROP = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
)


def is_reasoning(model: str, explicit: set[str]) -> bool:
    """Return True for deployments that use the reasoning request shape.

    Matches an explicit override set first, then o-series (``o1``/``o3``/``o4``...)
    and the gpt-5 family, all of which require max_completion_tokens and reject
    sampler params on Azure.
    """
    if not model:
        return False
    if model in explicit:
        return True
    m = model.lower()
    return bool(re.match(r"^(o\d|gpt-5)", m))


def adapt_reasoning_body(data: dict) -> None:
    """In place: rename max_tokens and strip sampler fields for reasoning models."""
    if "max_tokens" in data and "max_completion_tokens" not in data:
        data["max_completion_tokens"] = data.pop("max_tokens")
    else:
        data.pop("max_tokens", None)
    for field in _REASONING_DROP:
        data.pop(field, None)


def make_handler(endpoint: str, api_version: str, api_key: str,
                 deployments: list[str], reasoning: set[str], timeout: int,
                 max_retries: int = 6, retry_base: float = 2.0,
                 retry_cap: float = 60.0):
    endpoint = endpoint.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # ---- helpers ----
        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _error(self, code: int, payload: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, code: int, obj: dict):
            self._error(code, json.dumps(obj).encode("utf-8"))

        def _azure_url(self, deployment: str, kind: str) -> str:
            return (f"{endpoint}/openai/deployments/{deployment}/{kind}"
                    f"?api-version={api_version}")

        def _forward(self, url: str, body: bytes, stream_client: bool):
            req = urllib.request.Request(url, data=body or None, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("api-key", api_key)
            resp = urllib.request.urlopen(req, timeout=timeout)
            ctype = resp.headers.get("Content-Type", "application/json")
            if "text/event-stream" in ctype:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except Exception:
                    pass
            else:
                self._error(resp.status, resp.read(), ctype)

        # ---- chat / completions ----
        def _chat(self, raw: bytes, kind: str):
            try:
                data = json.loads(raw)
                assert isinstance(data, dict)
            except Exception:
                return self._send_json(400, {"error": "invalid JSON body"})
            deployment = data.get("model")
            if not deployment:
                return self._send_json(400, {"error": "missing 'model' (Azure deployment name)"})
            if is_reasoning(deployment, reasoning):
                adapt_reasoning_body(data)
            stream_client = bool(data.get("stream"))
            body = json.dumps(data).encode("utf-8")
            url = self._azure_url(deployment, kind)
            for attempt in range(max_retries + 1):
                try:
                    self._forward(url, body, stream_client)
                    return
                except urllib.error.HTTPError as exc:
                    payload = exc.read()
                    retryable = exc.code in _RETRY_STATUS and attempt < max_retries
                    print(f"[azure] upstream {exc.code} on {kind} ({deployment}) "
                          f"attempt {attempt + 1}/{max_retries + 1}"
                          f"{' -> retrying' if retryable else ''}: "
                          f"{payload[:300]!r}", file=sys.stderr, flush=True)
                    if not retryable:
                        return self._error(
                            exc.code, payload,
                            exc.headers.get("Content-Type", "application/json"))
                    time.sleep(self._backoff_delay(attempt, exc.headers))
                except Exception as exc:
                    if attempt < max_retries:
                        print(f"[azure] transport error on {kind} ({deployment}) "
                              f"attempt {attempt + 1}/{max_retries + 1} -> retrying: "
                              f"{exc!r}", file=sys.stderr, flush=True)
                        time.sleep(self._backoff_delay(attempt, None))
                        continue
                    return self._send_json(502, {"error": str(exc)})

        def _backoff_delay(self, attempt: int, headers) -> float:
            """Exponential backoff, honoring an upstream Retry-After header."""
            if headers is not None:
                ra = headers.get("Retry-After")
                if ra:
                    try:
                        return min(float(ra), retry_cap)
                    except ValueError:
                        pass
            return min(retry_base * (2 ** attempt), retry_cap)

        def _models(self):
            now = 0
            self._send_json(200, {
                "object": "list",
                "data": [{"id": d, "object": "model", "created": now,
                          "owned_by": "azure-openai"} for d in deployments],
            })

        def do_POST(self):  # noqa: N802
            body = self._read_body()
            if "chat/completions" in self.path:
                return self._chat(body, "chat/completions")
            if self.path.endswith("/completions"):
                return self._chat(body, "completions")
            self._send_json(404, {"error": f"unsupported path {self.path}"})

        def do_GET(self):  # noqa: N802
            if self.path.endswith("/models"):
                return self._models()
            self._send_json(404, {"error": f"unsupported path {self.path}"})

        def log_message(self, *args):  # silence access logs
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                        help="e.g. https://<resource>.openai.azure.com (or $AZURE_OPENAI_ENDPOINT)")
    parser.add_argument("--api-version", default=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
                        help="Azure api-version (or $AZURE_OPENAI_API_VERSION)")
    parser.add_argument("--deployments", default=os.environ.get("AZURE_DEPLOYMENTS", ""),
                        help="comma list of deployment names, for synthesized GET /v1/models")
    parser.add_argument("--reasoning-models", default=os.environ.get("AZURE_REASONING_MODELS", ""),
                        help="comma list of deployment names to force the reasoning request shape")
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("AZURE_TIMEOUT", "1800")))
    parser.add_argument("--max-retries", type=int,
                        default=int(os.environ.get("AZURE_MAX_RETRIES", "6")),
                        help="retries for 429/5xx upstream responses")
    parser.add_argument("--retry-base", type=float,
                        default=float(os.environ.get("AZURE_RETRY_BASE", "2.0")),
                        help="base seconds for exponential backoff")
    parser.add_argument("--retry-cap", type=float,
                        default=float(os.environ.get("AZURE_RETRY_CAP", "60.0")),
                        help="max backoff seconds per attempt")
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_KEY", "")
    if not args.endpoint:
        raise SystemExit("endpoint required; pass --endpoint or set AZURE_OPENAI_ENDPOINT")
    if not args.api_version:
        raise SystemExit("api-version required; pass --api-version or set AZURE_OPENAI_API_VERSION")
    if not api_key:
        raise SystemExit("AZURE_OPENAI_KEY environment variable is required (do not hardcode the key)")

    deployments = [d.strip() for d in args.deployments.split(",") if d.strip()]
    reasoning = {d.strip() for d in args.reasoning_models.split(",") if d.strip()}

    server = ThreadingHTTPServer(
        (args.host, args.listen_port),
        make_handler(args.endpoint, args.api_version, api_key,
                     deployments, reasoning, args.timeout,
                     max_retries=args.max_retries, retry_base=args.retry_base,
                     retry_cap=args.retry_cap),
    )
    print(
        f"azure-proxy listening on {args.host}:{args.listen_port} -> "
        f"{args.endpoint.rstrip('/')} (api-version={args.api_version}, "
        f"deployments={deployments or '[]'}, "
        f"reasoning_override={sorted(reasoning) or '[]'}, "
        f"max_retries={args.max_retries}, retry_base={args.retry_base}s, "
        f"retry_cap={args.retry_cap}s)",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
