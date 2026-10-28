"""End-to-end tests: Hermes prompt files, a fake upstream, the server in a thread.

tests/fixtures/ holds prompts built by Hermes' own prompt builder from invented sessions
(tools/make_fixtures.py). Prompts from your own sessions (tools/make_prompts.py) can go into
testdata/ (git-ignored); they are tested too when present.

Run: python3 -m unittest discover -s tests -v   (from the project root)
"""
import http.client
import json
import os
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FRESH = FIXTURES / "session_fresh.txt"
NO_USER = FIXTURES / "session_no_user.txt"


def prompt_files() -> list[Path]:
    files = sorted(FIXTURES.glob("*.txt"))
    local = ROOT / "testdata"
    if local.is_dir():
        files += sorted(p for p in local.glob("*.txt") if not p.name.endswith(".out.txt"))
    return files
UP_PORT, CF_PORT = 18192, 18191
os.environ.update(COMPRESS_FAST_UPSTREAM=f"http://127.0.0.1:{UP_PORT}", COMPRESS_FAST_PORT=str(CF_PORT),
                  COMPRESS_FAST_MODE="serve")

from compress_fast import diag, server, summarizer  # noqa: E402

UPSTREAM_SEEN = []
AUTH_SEEN = []
HYBRID_BODIES = []
SLOW_UPSTREAM = False
STRICT_UPSTREAM = False  # rejects fields it does not know (chat_template_kwargs, extras) like a strict server
# how the fake answers the reasoning call: "stream" (SSE), "json" (ignores stream), "length" (token cap hit
# inside the last section), "stall" (stops sending in the middle of the third section)
HYBRID_REPLY = "stream"
HYBRID_TEXT = ("## Constraints & Preferences\n- FAKE rule from model\n\n## Active State\n- FAKE state\n\n"
               "## Key Decisions\n- FAKE decision — reason\n\n## Errors & Fixes\n- FAKE error one — cause — fix\n"
               "- FAKE error two — cause — fix")


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, payload: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        UPSTREAM_SEEN.append(("GET", self.path, b""))
        self._reply(json.dumps({"object": "list", "data": [{"id": "qwen", "context_length": 204800}]}).encode())

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        UPSTREAM_SEEN.append(("POST", self.path, raw))
        body = json.loads(raw)
        content = body["messages"][-1]["content"]
        if "<digest>" in content:  # hybrid reasoning call
            AUTH_SEEN.append(self.headers.get("Authorization"))
            HYBRID_BODIES.append(body)
            if STRICT_UPSTREAM and {"chat_template_kwargs", "top_k"} & set(body):
                payload = b'{"error": {"message": "unknown field"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if SLOW_UPSTREAM:
                time.sleep(3)
            if HYBRID_REPLY == "json" or not body.get("stream"):
                self._reply(json.dumps({"choices": [{"message": {"role": "assistant", "content": HYBRID_TEXT},
                                                     "finish_reason": "stop"}]}).encode())
                return
            self._stream_hybrid()
            return
        if body.get("stream"):
            sse = b'data: {"choices":[{"delta":{"content":"up"}}]}\n\ndata: [DONE]\n\n'
            self._reply(sse, "text/event-stream")
        else:
            self._reply(json.dumps({"choices": [{"message": {"role": "assistant", "content": "from upstream"}}]}).encode())

    def _stream_hybrid(self):
        text, finish = HYBRID_TEXT, "stop"
        if HYBRID_REPLY == "length":  # cut inside the second error bullet
            text, finish = HYBRID_TEXT[:HYBRID_TEXT.index("two — cause")], "length"
        stall_at = HYBRID_TEXT.index("FAKE decision") if HYBRID_REPLY == "stall" else len(text)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i in range(0, len(text), 20):
                if i >= stall_at:
                    time.sleep(3)
                    return
                event = {"choices": [{"index": 0, "delta": {"content": text[i:i + 20]}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            end = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": {"completion_tokens": 42}}
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
        except (BrokenPipeError, ConnectionResetError):  # the client stopped reading at its deadline
            pass


def _start(handler, port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(path, body):
    c = http.client.HTTPConnection("127.0.0.1", CF_PORT, timeout=10)
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    try:
        return r.status, r.read()
    finally:
        c.close()


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = _start(FakeUpstream, UP_PORT)
        cls.cf = _start(server.Handler, CF_PORT)
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.up.shutdown()
        cls.cf.shutdown()

    def setUp(self):
        UPSTREAM_SEEN.clear()

    def prompts(self):
        files = prompt_files()
        self.assertTrue(files, "no prompt files in tests/fixtures/")
        return files

    def test_summary_served_without_upstream(self):
        for f in self.prompts():
            prompt = f.read_text()
            t = time.perf_counter()
            status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]})
            ms = (time.perf_counter() - t) * 1000
            self.assertEqual(status, 200, f.name)
            text = json.loads(raw)["choices"][0]["message"]["content"]
            self.assertIn("## Historical Task Snapshot", text, f.name)
            self.assertIn("## Detailed Session Log (oldest first)", text, f.name)
            self.assertLess(ms, 1000, f.name)
        self.assertEqual(UPSTREAM_SEEN, [], "summary prompts must not reach upstream")

    def test_summary_streaming(self):
        prompt = self.prompts()[0].read_text()
        status, raw = _post("/v1/chat/completions",
                            {"model": "m", "stream": True, "messages": [{"role": "user", "content": prompt}]})
        self.assertEqual(status, 200)
        events = [ln[6:] for ln in raw.decode().split("\n") if ln.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        content = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
        self.assertIn("## Goal", content)

    def test_other_chat_forwarded(self):
        status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(status, 200)
        self.assertIn(b"from upstream", raw)
        self.assertEqual(len(UPSTREAM_SEEN), 1)

    def test_other_chat_stream_forwarded(self):
        status, raw = _post("/v1/chat/completions",
                            {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn(b"data: [DONE]", raw)

    def test_focus_topic_kept_as_hint(self):
        # manual /compress: Hermes appends a focus block even without a topic
        prompt = (self.prompts()[0].read_text()
                  + '\n\nFOCUS TOPIC: "Recent user focus:\n- walls first"\nThis compaction should PRIORITISE x')
        status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]})
        text = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertIn("- walls first", text)
        self.assertEqual(UPSTREAM_SEEN, [])

    def test_garbled_summary_prompt_forwarded(self):
        prompt = summarizer.PREAMBLE + " but nothing else"
        status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]})
        self.assertIn(b"from upstream", raw)

    def test_models_forwarded(self):
        c = http.client.HTTPConnection("127.0.0.1", CF_PORT, timeout=5)
        c.request("GET", "/v1/models")
        r = c.getresponse()
        self.assertIn(b"204800", r.read())
        c.close()

    def _hybrid(self, reply="stream", timeout=90.0):
        global HYBRID_REPLY
        prompt = FRESH.read_text()
        server.MODE, server.HYBRID_TIMEOUT, HYBRID_REPLY = "hybrid", timeout, reply
        HYBRID_BODIES.clear()
        try:
            t = time.perf_counter()
            status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]})
            secs = time.perf_counter() - t
        finally:
            server.MODE, server.HYBRID_TIMEOUT, HYBRID_REPLY = "serve", 90.0, "stream"
        self.assertEqual(status, 200)
        return json.loads(raw)["choices"][0]["message"]["content"], secs

    def test_hybrid_merges_model_sections(self):
        text, _ = self._hybrid()
        self.assertIn("FAKE decision — reason", text)
        self.assertIn("## Completed Actions", text)          # facts from rules stay
        self.assertIn("Recorded failures (rules):", text)    # rule text follows the model's text
        self.assertIn("written by the model from a digest", text)
        self.assertNotIn("no model call", text)
        self.assertEqual(len(UPSTREAM_SEEN), 1)               # one small reasoning call only
        self.assertTrue(HYBRID_BODIES[0]["stream"])
        self.assertEqual(HYBRID_BODIES[0]["max_tokens"], server.HYBRID_MAX_TOKENS)

    def test_hybrid_json_answer(self):
        text, _ = self._hybrid("json")
        self.assertIn("FAKE error two — cause — fix", text)

    def test_hybrid_token_cap_keeps_finished_lines(self):
        text, _ = self._hybrid("length")
        self.assertIn("FAKE decision — reason", text)
        self.assertIn("FAKE error one — cause — fix", text)   # finished line of the cut section
        self.assertNotIn("FAKE error two", text)              # the line being written at the cap
        self.assertIn("cut off here: token cap", text)
        self.assertIn("AssertionError: 0 != 2", text)         # the rule text follows

    def test_hybrid_deadline_keeps_finished_sections(self):
        text, secs = self._hybrid("stall", timeout=1.0)
        self.assertIn("FAKE state", text)                     # finished before the deadline
        self.assertNotIn("FAKE dec", text)                    # unfinished at the deadline
        self.assertIn("## Completed Actions", text)
        self.assertLess(secs, 2.5)

    def test_hybrid_extra_body_dropped_by_strict_retry(self):
        global STRICT_UPSTREAM
        server.HYBRID_EXTRA_BODY, STRICT_UPSTREAM = {"top_k": 20}, True
        try:
            text, _ = self._hybrid()
        finally:
            server.HYBRID_EXTRA_BODY, STRICT_UPSTREAM = {}, False
        self.assertIn("FAKE decision — reason", text)
        self.assertEqual(HYBRID_BODIES[0]["top_k"], 20)
        self.assertNotIn("top_k", HYBRID_BODIES[1])
        self.assertNotIn("chat_template_kwargs", HYBRID_BODIES[1])

    def test_extra_body_setting_checked(self):
        for raw, ok in (('{"top_k": 20}', True), ("{top_k: 20}", False),
                        ('{"stream": false}', False), ("[1]", False)):
            os.environ["COMPRESS_FAST_TEST_EXTRA"] = raw
            if ok:
                self.assertEqual(server._json_object_env("COMPRESS_FAST_TEST_EXTRA"), {"top_k": 20})
            else:
                with self.assertRaises(ValueError):
                    server._json_object_env("COMPRESS_FAST_TEST_EXTRA")
        del os.environ["COMPRESS_FAST_TEST_EXTRA"]

    def test_hybrid_strict_upstream_and_auth(self):
        global STRICT_UPSTREAM
        prompt = self.prompts()[0].read_text()
        server.MODE, STRICT_UPSTREAM = "hybrid", True
        AUTH_SEEN.clear()
        try:
            c = http.client.HTTPConnection("127.0.0.1", CF_PORT, timeout=10)
            c.request("POST", "/v1/chat/completions",
                      body=json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}]}),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"})
            text = json.loads(c.getresponse().read())["choices"][0]["message"]["content"]
            c.close()
        finally:
            server.MODE, STRICT_UPSTREAM = "serve", False
        self.assertIn("FAKE decision — reason", text)            # retried without template kwargs
        self.assertEqual(AUTH_SEEN, ["Bearer test-key"] * 2)      # key passed on to the reasoning call

    def test_hybrid_timeout_falls_back_to_rules(self):
        global SLOW_UPSTREAM
        prompt = self.prompts()[0].read_text()
        server.MODE, server.HYBRID_TIMEOUT, SLOW_UPSTREAM = "hybrid", 1.0, True
        try:
            t = time.perf_counter()
            status, raw = _post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]})
            secs = time.perf_counter() - t
        finally:
            server.MODE, server.HYBRID_TIMEOUT, SLOW_UPSTREAM = "serve", 90.0, False
        text = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertNotIn("FAKE", text)
        self.assertIn("## Completed Actions", text)
        self.assertLess(secs, 2.5)

    def test_stats(self):
        c = http.client.HTTPConnection("127.0.0.1", CF_PORT, timeout=5)
        c.request("GET", "/compress-fast/stats")
        data = json.loads(c.getresponse().read())
        c.close()
        self.assertIn("counters", data)


class SummaryContentTest(unittest.TestCase):
    def test_no_user_rules(self):
        self.assertIn(summarizer.NO_USER_MARK, NO_USER.read_text()[:2000])
        for f in prompt_files():
            prompt = f.read_text()
            s = summarizer.summarize(prompt)
            self.assertIsNotNone(s, f.name)
            if summarizer.NO_USER_MARK in prompt[:2000]:
                self.assertIn(summarizer.NO_USER_SENTINEL, s)
                self.assertNotRegex(s, r"(?i)\bUser\s+asked\s*:")

    def test_facts_kept(self):
        s = summarizer.summarize(FRESH.read_text())
        # user rules verbatim, the error and that the same command later passed, written files
        self.assertIn("Do not touch the README", s)
        self.assertIn("A missing path must not crash the script", s)
        self.assertIn("FileNotFoundError", s)
        self.assertIn("AssertionError: 0 != 2", s)
        self.assertIn("/home/user/projects/diskreport/disk_report.py", s)
        self.assertIn("/home/user/projects/diskreport/test_disk_report.py", s)
        # the agent's own reasoning notes survive as quotes
        self.assertIn("main() always returned 0", s)

    def test_memory_context_kept_verbatim(self):
        prompt = FRESH.read_text()
        cut = max(prompt.rfind(m) for m in summarizer.TEMPLATE_MARKS)
        note = "secrets file: /home/x/notes/secrets.txt; the user answered: use nvme-thin"
        mem = ("\n\nMEMORY PROVIDER CONTEXT:\nThe block contains one JSON string.\n<memory-provider-context>\n"
               + json.dumps(note) + "\n</memory-provider-context>")
        s = summarizer.summarize(prompt[:cut] + mem + prompt[cut:])
        self.assertIn(note, s)

    def test_summaries_fit_target(self):
        model = {h: "- model text " * 40 for h in ("## Constraints & Preferences", "## Active State",
                                                   "## Key Decisions", "## Errors & Fixes")}
        for f in prompt_files():
            parsed = summarizer.parse_prompt(f.read_text())
            budget = int(parsed.target_tokens * summarizer.CHARS_PER_TOKEN)
            self.assertLessEqual(len(summarizer.summarize_parsed(parsed)), budget, f.name)
            # the model's own text is never cut, nor is the short line of each action; with every other cap at its
            # floor the rest may stand slightly on top of the budget
            hybrid_rules_part = len(summarizer.summarize_parsed(parsed, model)) - sum(len(t) for t in model.values())
            self.assertLessEqual(hybrid_rules_part, budget * 1.02, f.name + " (hybrid)")

    def test_log_refers_to_action_numbers(self):
        s = summarizer.summarize(FRESH.read_text())
        log = s.split(summarizer.SESSION_LOG, 1)[1]
        self.assertRegex(log, r"- actions? #\d+")
        self.assertNotIn("- -> ", log)                        # calls are not repeated from Completed Actions
        self.assertIn("(quoted above)", log)                  # nor are user turns quoted in Constraints

    def test_completeness_notes(self):
        s = summarizer.summarize(FRESH.read_text())
        errors = summarizer._previous_sections(s)["## Errors & Fixes"]
        self.assertRegex(errors, r"Every exception reported as 'Name: message' in this window \(actions #1-#\d+\): "
                                 r"AssertionError, FileNotFoundError")
        self.assertRegex(s, r"All \d+ user turns of this window are quoted above in full")

    def test_completeness_note_only_when_true(self):
        prompt = FRESH.read_text()
        parsed = summarizer.parse_prompt(prompt)
        f = summarizer.extract(parsed, 1)
        self.assertEqual(summarizer._user_turns_note(parsed, f, "- User said: something else"), "")

    def test_each_file_change_listed(self):
        s = summarizer.summarize(sorted(FIXTURES.glob("*.txt"))[0].read_text())
        actions = summarizer._previous_sections(s)["## Completed Actions"]
        for line in actions.splitlines():
            if "[tool: patch]" in line or "[tool: write_file]" in line:
                self.assertNotRegex(line, r"\(x\d+, through #")

    def test_legacy_truncation_marker_noted(self):
        prompt = FRESH.read_text()
        at = prompt.index("\n\n[ASSISTANT]: ")
        marker = ("\n\n...[summary input truncated: omitted 12,345 chars from the middle to keep compression prompt "
                  "bounded]...")
        s = summarizer.summarize(prompt[:at] + marker + prompt[at:])
        self.assertIn("1 block(s) of records were elided by Hermes", s)

    def test_update_mode_continues_numbering(self):
        prompt = FRESH.read_text()
        first = summarizer.summarize(prompt)
        head_end = prompt.find(summarizer.FRESH_MARK)
        turns_and_rest = prompt[head_end + len(summarizer.FRESH_MARK):]
        upd = (prompt[:head_end] + "\nPREVIOUS SUMMARY:\n" + first + summarizer.NEW_MARK + turns_and_rest)
        upd = upd.replace("\n\nUse this exact structure:\n", "\n\nUpdate the summary using this exact structure.\n")
        second = summarizer.summarize(upd)
        self.assertIsNotNone(second)
        first_max = max(int(x) for x in re.findall(r"(?m)^(\d+)\. ", first))
        nums = [int(x) for x in re.findall(r"(?m)^(\d+)\. ", second)]
        self.assertGreater(max(nums), first_max)


if __name__ == "__main__":
    unittest.main()
