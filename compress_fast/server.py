"""OpenAI-compatible endpoint for Hermes auxiliary.compression.

Compaction summary requests are answered by rules (summarizer.summarize). Everything else, and
every compaction prompt the summarizer does not recognise, goes unchanged to the upstream
OpenAI-compatible model server, including streaming responses.

Modes (COMPRESS_FAST_MODE):
  hybrid  rules for the facts, the upstream model writes the four reasoning sections from a small
          digest (COMPRESS_FAST_HYBRID_TIMEOUT, default 90 s; sections unfinished by then stay rules
          only) (default)
  serve   answer recognised compaction prompts by rules alone, no model call
  shadow  forward everything to upstream, but also build the rule summary and store prompt,
          rule summary and model summary side by side for comparison
"""
from __future__ import annotations

import http.client
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__, diag, hybrid, summarizer

UPSTREAM = os.environ.get("COMPRESS_FAST_UPSTREAM", "http://127.0.0.1:8080")
LISTEN_HOST = os.environ.get("COMPRESS_FAST_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("COMPRESS_FAST_PORT", "8091"))
MODE = os.environ.get("COMPRESS_FAST_MODE", "hybrid")
# hybrid: model for the reasoning call when the request names none (Hermes always sends one)
MODEL = os.environ.get("COMPRESS_FAST_MODEL", "")
SHADOW_DIR = Path(os.environ.get("COMPRESS_FAST_SHADOW_DIR", str(Path.home() / ".hermes" / "compress-fast-shadow")))
UPSTREAM_TIMEOUT = float(os.environ.get("COMPRESS_FAST_UPSTREAM_TIMEOUT", "1800"))
# hybrid: the model writes the reasoning sections; past this many seconds the sections it has finished are
# used and the rest stays rules only
HYBRID_TIMEOUT = float(os.environ.get("COMPRESS_FAST_HYBRID_TIMEOUT", "90"))
# hybrid: token cap for that answer; a section cut off by it is dropped
HYBRID_MAX_TOKENS = int(os.environ.get("COMPRESS_FAST_HYBRID_MAX_TOKENS", str(hybrid.MAX_TOKENS)))


def _json_object_env(name: str) -> dict:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid JSON ({exc}): {raw}") from None
    if not isinstance(value, dict) or {"model", "messages", "stream"} & set(value):
        raise ValueError(f"{name} must be a JSON object without model, messages or stream: {raw}")
    return value


# hybrid: extra request fields for that call, e.g. {"top_k": 20}. Left out again when the server rejects the
# request.
HYBRID_EXTRA_BODY = _json_object_env("COMPRESS_FAST_HYBRID_EXTRA_BODY")

HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-authorization", "proxy-authenticate", "host", "content-length"}


def _summary_prompt(body: dict) -> str | None:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str) or not content.startswith(summarizer.PREAMBLE):
        return None
    return content


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"compress-fast/{__version__}"

    def log_message(self, fmt, *args):  # route http.server logging into our log
        diag.log.debug("http %s", fmt % args)

    def do_GET(self):
        if self.path.rstrip("/") == "/compress-fast/stats":
            self._send_json(200, {"version": __version__, "mode": MODE, "upstream": UPSTREAM, **diag.stats()})
            return
        self._forward(b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.rstrip("/").endswith("/chat/completions"):
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            prompt = _summary_prompt(body) if isinstance(body, dict) else None
            if prompt is not None:
                diag.count("request.summary_prompt")
                if MODE == "shadow":
                    self._shadow(prompt, raw)
                    return
                start = time.perf_counter()
                with diag.timer("request.summarize"):
                    if MODE == "hybrid":
                        model = body.get("model") or MODEL
                        summary, info = hybrid.summarize_hybrid(prompt, UPSTREAM, model, HYBRID_TIMEOUT,
                                                                auth=self.headers.get("Authorization"),
                                                                max_tokens=HYBRID_MAX_TOKENS,
                                                                extra_body=HYBRID_EXTRA_BODY)
                        diag.log.info("hybrid info: %s", info)
                    else:
                        summary = summarizer.summarize(prompt)
                ms = (time.perf_counter() - start) * 1000
                if summary:
                    diag.count("request.served_by_rules")
                    diag.log.info("served by rules: prompt=%d chars summary=%d chars in %.1f ms",
                                  len(prompt), len(summary), ms)
                    self._send_completion(body, summary, prompt)
                    return
                diag.count("request.summary_forwarded")
                diag.log.info("summary prompt not handled by rules (%d chars) -> upstream", len(prompt))
        diag.count("request.forwarded")
        self._forward(raw)

    do_PUT = do_POST
    do_DELETE = do_GET

    # ---- responses -------------------------------------------------------------------------

    def _send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_completion(self, body: dict, text: str, prompt: str) -> None:
        cid = "chatcmpl-cf-" + uuid.uuid4().hex[:20]
        created = int(time.time())
        model = body.get("model") or "compress-fast"
        usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
                 "total_tokens": (len(prompt) + len(text)) // 4}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
            events = [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            for ev in events:
                self._chunk(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
            return
        self._send_json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": usage,
        })

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    # ---- forwarding ------------------------------------------------------------------------

    def _forward(self, raw: bytes, capture: list | None = None) -> None:
        up = urlsplit(UPSTREAM)
        start = time.perf_counter()
        if up.scheme == "https":
            conn = http.client.HTTPSConnection(up.hostname, up.port or 443, timeout=UPSTREAM_TIMEOUT)
        else:
            conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=UPSTREAM_TIMEOUT)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Content-Length"] = str(len(raw))
        try:
            with diag.timer("forward.connect"):
                conn.request(self.command, (up.path.rstrip("/") + self.path), body=raw or None, headers=headers)
                resp = conn.getresponse()
        except OSError as exc:
            diag.count("forward.upstream_error")
            diag.log.warning("upstream %s unreachable: %s", UPSTREAM, exc)
            self._send_json(502, {"error": {"message": f"compress-fast: upstream unreachable: {exc}",
                                            "type": "upstream_error"}})
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        total = 0
        try:
            while True:
                data = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not data:
                    break
                total += len(data)
                if capture is not None:
                    capture.append(data)
                self._chunk(data)
            self._chunk(b"")
        except (BrokenPipeError, ConnectionResetError):
            diag.count("forward.client_gone")
        finally:
            conn.close()
        ms = (time.perf_counter() - start) * 1000
        diag.count("forward.bytes", total)
        diag.log.info("forwarded %s %s -> %d, %d bytes in %.0f ms", self.command, self.path, resp.status, total, ms)

    def _shadow(self, prompt: str, raw: bytes) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        start = time.perf_counter()
        rules = summarizer.summarize(prompt)
        rules_ms = (time.perf_counter() - start) * 1000
        captured: list[bytes] = []
        model_start = time.perf_counter()
        self._forward(raw, capture=captured)
        model_ms = (time.perf_counter() - model_start) * 1000
        try:
            SHADOW_DIR.mkdir(parents=True, exist_ok=True)
            os.chmod(SHADOW_DIR, 0o700)
            base = SHADOW_DIR / stamp
            for suffix, text in ((".prompt.txt", prompt), (".rules.txt", rules or ""),
                                 (".model.raw", b"".join(captured).decode("utf-8", "replace"))):
                path = Path(str(base) + suffix)
                path.write_text(text)
                os.chmod(path, 0o600)
            diag.count("shadow.saved")
            diag.log.info("shadow %s: rules %s in %.1f ms, model %.0f ms", stamp,
                          "ok" if rules else "not handled", rules_ms, model_ms)
        except OSError:
            diag.count("shadow.save_error")
            diag.log.exception("shadow save failed")


def main() -> None:
    diag.setup()
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    server.daemon_threads = True
    diag.log.info("compress-fast %s listening on %s:%d, mode=%s, upstream=%s, hybrid timeout=%.0f s max_tokens=%d "
                  "extra fields=%s", __version__, LISTEN_HOST, LISTEN_PORT, MODE, UPSTREAM, HYBRID_TIMEOUT,
                  HYBRID_MAX_TOKENS, sorted(HYBRID_EXTRA_BODY) or "none")
    server.serve_forever()


if __name__ == "__main__":
    main()
