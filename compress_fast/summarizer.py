"""Turn a Hermes compaction prompt into a structured summary without a model call.

The input is the exact text Hermes sends to auxiliary.compression (one user message):
a preamble, optionally PREVIOUS SUMMARY, the serialized turns ([USER]/[ASSISTANT]/[TOOL RESULT id]),
optionally MEMORY PROVIDER CONTEXT, then the section template. Anything that does not match this
shape returns None, and the server forwards the request to the real model instead.

Ideas for the fact extraction (commands with outcomes, errors and where they were fixed, file
lists) follow hermes-vcc-engine (Apache-2.0, github.com/danbiocchi/hermes-vcc-engine); the code
here is written for the prompt-text interface, not for Hermes internals.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property

from . import diag

PREAMBLE = "You are a summarization agent creating a context checkpoint."
NO_USER_MARK = "This session contains no user-authored turns."
NO_USER_SENTINEL = "None. This session contains no user-authored turns."
FRESH_MARK = "\nTURNS TO SUMMARIZE:\n"
PREV_MARK = "\nPREVIOUS SUMMARY:\n"
NEW_MARK = "\n\nNEW TURNS TO INCORPORATE:\n"
MEMORY_MARK = "\n\nMEMORY PROVIDER CONTEXT:\n"
TEMPLATE_MARKS = ("\n\nUse this exact structure:\n", "\n\nUpdate the summary using this exact structure.")
FOCUS_MARK = "\n\nFOCUS TOPIC:"

HANDOFF_PREFIX = "[CONTEXT COMPACTION"
OOB_RE = re.compile(r"^\[OUT-OF-BAND USER MESSAGE[^\]]*\]\s*(.*?)\s*\[/OUT-OF-BAND USER MESSAGE\]$", re.S)

HISTORICAL = "## Historical Task Snapshot"
SESSION_LOG = "## Detailed Session Log (oldest first)"

RECORD_RE = re.compile(r"(?:^|\n\n)\[(USER|ASSISTANT|SYSTEM|TOOL RESULT [^\]\n]*)\]: ")
# The closing "\n]" can be cut away when Hermes elides the middle of a long message.
TOOL_CALLS_RE = re.compile(r"(?:^|\n)\[Tool calls:\n(.*?)(?:\n\]\s*)?$", re.S)
RESULT_PATH_RE = re.compile(r"^\[[a-z_]+\] (?:replace in|wrote to|read|created|edited|patched) (/\S+)")
DIFF_PATH_RE = re.compile(r'"diff": "--- a/(/[^"\\\s]+)')
CALL_LINE_RE = re.compile(r"^  ([A-Za-z0-9_.:\-]+)\((.*)$")
# lean tail mode samples oversized input ("records ... elided"); the legacy mode cuts the middle once
ELISION_RE = re.compile(r"\.\.\.\[(?:records [\d,]+-[\d,]+: [\d,]+ chars elided|summary input truncated: omitted [\d,]+ chars)"
                        r"[^\]]*\]\.\.\.")
SKILL_PRUNED_RE = re.compile(r"\[SKILL_PRUNED:[^\]]*\]")
TARGET_TOKENS_RE = re.compile(r"Target ~(\d+) tokens")
PATH_RE = re.compile(r"(?<![\w.])((?:~|/)[\w.\-+@]+(?:/[\w.\-+@]+)+)")
ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\)|\b[A-Z][A-Za-z]*(?:Error|Exception)\b[:\s]|\bERROR\b|\bFAILED\b|"
    r"\bfatal:|Blocked by|command not found|No such file or directory|Permission denied)")

WRITE_TOOLS = {"write_file", "patch", "edit_file", "str_replace", "create_file", "apply_patch", "replace_in_file",
               "edit", "multi_edit", "write"}
READ_TOOLS = {"read_file", "view_file", "skill_view", "cat", "open_file", "search_files", "grep", "list_files"}
TARGET_KEYS = ("path", "file_path", "filename", "file", "target", "url", "name", "query", "skill", "pattern")

# Measured with the Qwen3.8 tokenizer, 8 rule and hybrid summaries over 3000 tokens: 2.81-3.29 chars per token
# (paths, commands and code are dense). The lowest keeps every measured summary within Hermes' target.
CHARS_PER_TOKEN = 2.8

# Section caps as shares of the summary budget (chars). At Hermes' usual 7000-token target they are close to
# the fixed caps of 0.3.0. The session log takes what is left, at least its "log" share.
SECTION_SHARE = {
    "## Goal": 0.10, "## Constraints & Preferences": 0.36, "## Completed Actions": 0.40, "## Key Decisions": 0.24,
    # Critical Context: the fixed 12000 of 0.3.0 never applied (largest measured 4063 chars)
    "## Errors & Fixes": 0.16, "## Resolved Questions": 0.12, "## Relevant Files": 0.16, "## Critical Context": 0.25,
    "log": 0.08,            # minimum for the session log
    "last_message": 0.12,   # last substantial assistant message, verbatim in Critical Context
}
# When the summary is over its target, caps shrink tier by tier, each down to its floor factor: first what is
# repeated elsewhere (log, the verbatim last message, answers to questions), then the assistant's notes and the
# file lists (read-only files go first), then actions and errors (older actions fold per target), last the
# constraints, where a cut takes the entries of earlier summaries first (this window's user turns are at the end).
# The goal never shrinks.
SHRINK_TIERS = (
    (("log", "last_message", "## Resolved Questions"), 0.15),
    (("## Key Decisions", "## Relevant Files", "## Critical Context"), 0.3),
    (("## Completed Actions", "## Errors & Fixes"), 0.6),
    (("## Constraints & Preferences",), 0.5),
)
# Completed Actions may grow past its share to this much of the budget, so that every action keeps its short line
# (177 calls with long commands needed 56 % in one measured window)
ACTIONS_MAX_SHARE = 0.6
WD_SECTIONS = ("## Completed Actions", "## Active State", "## Blocked", "## Errors & Fixes")
# hybrid: label in front of the rule text that follows the model's own text in a section
MODEL_SECTION_LABELS = {
    "## Constraints & Preferences": "User turns (verbatim, rules):",
    "## Active State": "Recorded state (rules):",
    "## Key Decisions": "Assistant notes (rules):",
    "## Errors & Fixes": "Recorded failures (rules):",
}
ORIGIN_NOTE_RE = re.compile(r"(?m)^(?:This summary was built by rules|Facts in this summary were extracted by rules).*\n?")
USER_TURNS_NOTE_RE = re.compile(r"(?m)^- \(All \d+ user turns? of this window are quoted above in full[^\n]*\n?")


@dataclass
class Call:
    tool: str
    args_raw: str
    args: dict
    index: int = 0              # action number in this summary
    record: int = 0             # record position in the window
    outcome: str = "no result in window"
    ok: bool | None = None
    error: str = ""
    result_excerpt: str = ""
    abbrevs: tuple = ()         # (directory, name) pairs, longest first; shortened before truncating

    def _short(self, text: str) -> str:
        return _abbreviate(text, self.abbrevs)

    # cached: a summary build reads it tens of thousands of times. extract() may still fill args["path"]
    # from the result, so nothing may read it before extract() has finished.
    @cached_property
    def target(self) -> str:
        if self.tool in ("terminal", "shell", "bash", "run_command", "execute"):
            cmd = self._short(str(self.args.get("command") or self.args.get("cmd") or ""))
            return "`" + _one_line(cmd, 160) + "`" if cmd else _one_line(self.args_raw, 120)
        for key in TARGET_KEYS:
            if self.args.get(key):
                return _one_line(self._short(str(self.args[key])), 160)
        if self.args.get("code"):
            return "code: " + _one_line(str(self.args["code"]), 110)
        return _one_line(self.args_raw, 120)


@dataclass
class Parsed:
    mode: str                    # "fresh" | "update"
    no_user: bool
    previous: str
    records: list[tuple[str, str]]
    memory: str
    headings: list[str]
    target_tokens: int
    elided_notes: list[str] = field(default_factory=list)
    focus: str = ""


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_prompt(prompt: str) -> Parsed | None:
    """Split the Hermes prompt into its parts; None when it is not a compaction prompt we handle."""
    with diag.timer("parse_prompt"):
        if not prompt.startswith(PREAMBLE):
            diag.count("parse.not_summary_prompt")
            return None
        tpl_at = max(prompt.rfind(m) for m in TEMPLATE_MARKS)
        if tpl_at < 0:
            diag.count("parse.no_template")
            return None
        # Manual /compress always carries a focus block (Hermes derives "Recent user focus" from the
        # latest user turns when no topic is given); keep it as a hint, the user turns are kept anyway.
        focus = ""
        focus_at = prompt.find(FOCUS_MARK, tpl_at)
        if focus_at >= 0:
            fm = re.match(r'\s*"(.*?)"\nThis compaction should PRIORITISE', prompt[focus_at + len(FOCUS_MARK):], re.S)
            focus = fm.group(1).strip() if fm else ""
            diag.count("parse.focus_topic")
        head, template = prompt[:tpl_at], prompt[tpl_at:focus_at if focus_at >= 0 else len(prompt)]
        no_user = NO_USER_MARK in head[:2000]
        previous = ""
        prev_at = head.find(PREV_MARK)
        if prev_at >= 0:
            new_at = head.find(NEW_MARK, prev_at)
            if new_at < 0:
                diag.count("parse.update_without_new_turns")
                return None
            previous = head[prev_at + len(PREV_MARK):new_at]
            body = head[new_at + len(NEW_MARK):]
            mode = "update"
        else:
            fresh_at = head.find(FRESH_MARK)
            if fresh_at < 0:
                diag.count("parse.no_turns_marker")
                return None
            body = head[fresh_at + len(FRESH_MARK):]
            mode = "fresh"
        memory = ""
        mem_at = body.rfind(MEMORY_MARK)
        if mem_at >= 0 and "<memory-provider-context>" in body[mem_at:]:
            memory = _decode_memory(body[mem_at:])
            body = body[:mem_at]
        records = _split_records(body)
        if not records:
            diag.count("parse.no_records")
            return None
        headings = [line.strip() for line in template.splitlines() if line.startswith("## ")]
        if HISTORICAL not in headings:
            diag.count("parse.unknown_template")
            return None
        m = TARGET_TOKENS_RE.search(template)
        target = int(m.group(1)) if m else 6000
        elided = ELISION_RE.findall(body)
        diag.count(f"parse.ok.{mode}")
        return Parsed(mode, no_user, previous.strip(), records, memory, headings, target, elided, focus)


def _decode_memory(block: str) -> str:
    m = re.search(r"<memory-provider-context>\n(.*?)\n</memory-provider-context>", block, re.S)
    raw = m.group(1) if m else block.split("<memory-provider-context>", 1)[1].strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except ValueError:
        diag.count("parse.memory_not_json")
        return raw


def _split_records(body: str) -> list[tuple[str, str]]:
    marks = list(RECORD_RE.finditer(body))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append((m.group(1), body[m.end():end].strip("\n")))
    return out


def _parse_calls(content: str) -> tuple[str, list[Call]]:
    m = TOOL_CALLS_RE.search(content)
    if not m:
        return content.strip(), []
    text = content[: m.start()].strip()
    calls: list[Call] = []
    for line in m.group(1).split("\n"):
        cm = CALL_LINE_RE.match(line)
        if cm:
            raw = cm.group(2)
            raw = raw[:-1] if raw.endswith(")") else raw
            name, args = cm.group(1), _loads_args(raw)
            if name == "tool_call":
                # wrapper for MCP calls: {"calls":[{"arguments":{...},"name":"mcp__x"}]}
                inner = (args.get("calls") or [{}])[0] if isinstance(args.get("calls"), list) else {}
                nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                name = inner.get("name") or (nm.group(1) if nm else name)
                if isinstance(inner.get("arguments"), dict):
                    args = inner["arguments"]
            calls.append(Call(name, raw, args))
        elif calls:  # continuation of a multi-line argument string
            calls[-1].args_raw += "\n" + line
    return text, calls


def _loads_args(raw: str) -> dict:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except ValueError:
        # truncated JSON ("..."): pull the simple string fields we need
        out = {}
        for key in TARGET_KEYS + ("command", "cmd", "code"):
            km = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, raw)
            if km:
                try:
                    out[key] = json.loads('"' + km.group(1) + '"')
                except ValueError:
                    out[key] = km.group(1)
        return out


STUB_TERMINAL_RE = re.compile(r"^\[terminal\] ran `.*?` -> exit (-?\d+)(.*)$", re.S)
STUB_GENERIC_RE = re.compile(r"^\[[a-z_]+\] [^\n]*?\([\d,]+ (?:chars result|lines)\)(.*)$", re.S)
STUB_DEMOTED_RE = re.compile(r"^\[[a-z_ ]*output demoted at compaction")
UNTRUSTED_RE = re.compile(r"^<untrusted_tool_result[^>]*>\s*(?:The following content[^\n]*\n)?", re.S)


def classify_result(content: str) -> tuple[bool | None, str, str]:
    """(ok, short outcome, error excerpt) for one tool result."""
    text = UNTRUSTED_RE.sub("", content.strip()).strip()
    if text.endswith("</untrusted_tool_result>"):
        text = text[: -len("</untrusted_tool_result>")].strip()
    # Hermes prunes old results to one-line stubs before summarizing; read those first.
    m = STUB_TERMINAL_RE.match(text)
    if m:
        code = int(m.group(1))
        rest = _one_line(m.group(2).lstrip(", "), 120)
        return (code == 0), f"exit {code}" + (f", {rest}" if rest else ""), ("" if code == 0 else f"exit {code}")
    if STUB_DEMOTED_RE.match(text):
        return None, "output archived by Hermes (session_search)", ""
    m = STUB_GENERIC_RE.match(text)
    if m:
        tail = m.group(1).strip()
        if ERROR_LINE_RE.search(tail):
            excerpt = _error_excerpt(tail)
            return False, f"error: {excerpt}", excerpt
        return True, "ok", ""
    data = None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    if isinstance(data, dict):
        out = str(data.get("output") or data.get("result") or data.get("content") or "")
        err = str(data.get("error") or data.get("stderr") or "")
        code = data.get("exit_code", data.get("returncode"))
        if isinstance(code, int):
            if code == 0 and not err:
                return True, f"exit 0: {_output_gist(out) or 'no output'}", ""
            excerpt = _error_excerpt(err + "\n" + out)
            return False, f"exit {code}: {excerpt}", excerpt
        if data.get("success") is False or err:
            excerpt = _error_excerpt(err or out or text)
            return False, f"failed: {excerpt}", excerpt
        if data.get("success") is True:
            diff = str(data.get("diff") or "")
            if diff:
                plus = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
                minus = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
                return True, f"ok (+{plus}/-{minus} lines)" + _diff_gist(diff), ""
            return True, f"ok: {_output_gist(out or text)}" if (out or text) else "ok", ""
        text = out or text
    if ERROR_LINE_RE.search(text[:4000] + text[-2000:]):
        excerpt = _error_excerpt(text)
        return False, f"error: {excerpt}", excerpt
    return True, "ok: " + _output_gist(text), ""


DIFF_KEY_LINE_RE = re.compile(
    r"^\+\s*(def |class |async def |import |from \S+ import |#\s*\S|[A-Z_][A-Z0-9_]{2,}\s*=|.*add_argument\(|"
    r"function |const |let |export |## |- \*\*)")


def _diff_gist(diff: str) -> str:
    """Up to three added lines that name what changed (definitions, imports, constants, headings)."""
    keys = []
    for ln in diff.splitlines():
        if ln.startswith("+") and not ln.startswith("+++") and DIFF_KEY_LINE_RE.match(ln):
            keys.append(_one_line(ln[1:].strip(), 90))
        if len(keys) == 3:
            break
    return (": " + "; ".join(keys)) if keys else ""


def _output_gist(text: str, limit: int = 280) -> str:
    """First line plus the last two lines: results and totals usually sit at the end."""
    lines = [" ".join(ln.split()) for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) <= 3:
        return _one_line(" | ".join(lines), limit)
    return _one_line(f"{lines[0]} | ... | {lines[-2]} | {lines[-1]}", limit)


# not \b: in escaped output (JSON in JSON) the name follows a literal "\n", which \b does not see as a boundary
EXC_RE = re.compile(r"(?:(?<![A-Za-z0-9_])|(?<=\\n)|(?<=\\t))([A-Z][A-Za-z]*(?:Error|Exception)): ([^\n\"\\]{3,200})")
EXC_PAIR_RE = re.compile(r'exception_type\W+(\w+)\W+message\W+(.{3,200}?)\\*"')


def _error_excerpt(text: str) -> str:
    # Escaped tracebacks (JSON inside JSON from MCP tools) have no real newlines; the final
    # "SomethingError: message" is the line that matters.
    found = list(EXC_RE.finditer(text))
    if found:
        return _one_line(f"{found[-1].group(1)}: {found[-1].group(2)}", 220)
    pair = EXC_PAIR_RE.search(text)
    if pair:
        return _one_line(f"{pair.group(1)}: {pair.group(2)}", 220)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "no output"
    if any("Traceback (most recent call last)" in ln for ln in lines):
        for ln in reversed(lines):
            if re.match(r"[A-Za-z_.]*(Error|Exception|Exit|Interrupt)\b", ln):
                return _one_line(ln, 220)
        return _one_line(lines[-1], 220)
    for ln in lines:
        if ERROR_LINE_RE.search(ln):
            return _one_line(ln, 220)
    return _one_line(lines[-1], 220)


@dataclass
class Facts:
    users: list[str] = field(default_factory=list)
    assistant_notes: list[tuple[int, str]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    files_read: dict[str, int] = field(default_factory=dict)
    files_written: dict[str, int] = field(default_factory=dict)
    # str, (role, text) or Call; rendered by render_log once results, paths and the other sections are known
    log: list = field(default_factory=list)
    skill_markers: list[str] = field(default_factory=list)
    questions: list[tuple[str, str]] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)  # every "SomeError: message" name in the window


def render_log(f: Facts, quoted: str = "") -> list[str]:
    """Chronological log. Every call is listed in Completed Actions, so the calls between two messages
    become one line with their action numbers and outcome counts. A user or assistant message whose
    log text already stands in `quoted` (the other sections) keeps only its start here."""
    with diag.timer("render_log"):
        out: list[str] = []
        run: list[Call] = []
        shortened = 0
        for item in f.log + [None]:
            if isinstance(item, Call):
                run.append(item)
                continue
            if run:
                out.append(_log_run(run))
                run = []
            if item is None:
                break
            if isinstance(item, tuple):
                role, text = item
                limit = 400 if role == "user" else 300
                flat = " ".join(text.split())
                if quoted and len(flat) > 160 and flat[: limit - 3] in quoted:
                    out.append(f"- {role}: {_one_line(flat, 160)} (quoted above)")
                    shortened += 1
                else:
                    out.append(f"- {role}: {_one_line(text, limit)}")
            else:
                out.append(item)
        diag.count("render_log.messages_shortened", shortened)
        return out


def _log_run(calls: list[Call]) -> str:
    """One log line for consecutive calls: action numbers, tools, outcome counts, failed targets."""
    first, last = calls[0], calls[-1]
    if len(calls) == 1:
        state = "ok" if first.ok else "FAILED" if first.ok is False else "outcome unknown"
        return f"- action #{first.index} {first.tool} {_one_line(first.target, 100)}: {state}"
    tools: dict[str, int] = {}
    for c in calls:
        tools[c.tool] = tools.get(c.tool, 0) + 1
    kinds = ", ".join(f"{t} x{n}" if n > 1 else t for t, n in sorted(tools.items(), key=lambda kv: -kv[1])[:3])
    kinds += ", ..." if len(tools) > 3 else ""
    failed = [c for c in calls if c.ok is False]
    ok = sum(1 for c in calls if c.ok)
    parts = [f"{ok} ok"] if ok else []
    if failed:
        parts.append(f"{len(failed)} FAILED (" + ", ".join(f"#{c.index} {c.tool} {_one_line(c.target, 60)}"
                                                           for c in failed[:3]) + (", ..." if len(failed) > 3 else "") + ")")
    if len(calls) - ok - len(failed):
        parts.append(f"{len(calls) - ok - len(failed)} outcome unknown")
    return f"- actions #{first.index}-#{last.index} ({kinds}): {', '.join(parts)}"


def extract(parsed: Parsed, first_action: int, abbrevs: tuple = ()) -> Facts:
    with diag.timer("extract"):
        f = Facts()
        pending: list[Call] = []
        open_question: str | None = None
        for pos, (role, content) in enumerate(parsed.records):
            f.skill_markers += SKILL_PRUNED_RE.findall(content)
            if role == "USER":
                text = content.strip()
                if text.startswith(HANDOFF_PREFIX):
                    # an earlier compaction handoff replayed as a user row: background, not a request
                    diag.count("extract.handoff_rows_skipped")
                    f.log.append("- (earlier compaction summary row; see Critical Context of that summary)")
                    continue
                oob = OOB_RE.match(text)
                if oob:
                    text = oob.group(1).strip()
                if text in f.users:
                    diag.count("extract.duplicate_user_rows")
                    continue
                f.users.append(text)
                f.log.append(("user", text))
                open_question = text if "?" in text else None
            elif role == "ASSISTANT":
                text, calls = _parse_calls(content)
                if text:
                    f.assistant_notes.append((pos, text))
                    f.log.append(("assistant", text))
                    if open_question:
                        f.questions.append((open_question, text))
                        open_question = None
                for c in calls:
                    c.record = pos
                    c.index = first_action + len(f.calls)
                    c.abbrevs = abbrevs
                    f.calls.append(c)
                    f.log.append(c)
                    _note_files(f, c)
                pending = list(calls)
            elif role.startswith("TOOL RESULT"):
                ok, outcome, error = classify_result(content)
                call = pending.pop(0) if pending else None
                if call:
                    call.ok, call.outcome, call.error = ok, outcome, error
                    call.result_excerpt = _one_line(content, 200)
                    pm = RESULT_PATH_RE.match(content.strip()) or DIFF_PATH_RE.search(content[:600])
                    if pm and not any(isinstance(call.args.get(k), str) for k in ("path", "file_path")):
                        # truncated args lost the path; the result stub still names it
                        call.args["path"] = pm.group(1)
                        _note_files(f, call)
                else:
                    f.log.append(f"- <- tool: {_one_line(outcome, 220)}")
            else:
                f.log.append(f"- {role.lower()}: {_one_line(content, 200)}")
        f.exceptions = sorted({m.group(1) for _, content in parsed.records for m in EXC_RE.finditer(content)})
        diag.count("extract.calls", len(f.calls))
        diag.count("extract.records", len(parsed.records))
        return f


def _note_files(f: Facts, c: Call) -> None:
    paths = []
    for key in ("path", "file_path", "filename", "file"):
        if isinstance(c.args.get(key), str):
            paths.append(c.args[key])
    cmd = str(c.args.get("command") or c.args.get("cmd") or "")
    cd_dirs = set()
    if cmd:
        paths += PATH_RE.findall(cmd)
        cd_dirs = {d.rstrip("/") for d in re.findall(r"\bcd\s+([^\s;&|]+)", cmd)}
    for p in paths:
        if p.rstrip("/") in cd_dirs:  # a directory we moved into, not a file we used
            continue
        p = p.rstrip(".,;:)'\"")
        if len(p) < 4 or p.count("/") < 1 or p.startswith(("/dev/", "/proc/", "/sys/")):
            continue
        written = c.tool in WRITE_TOOLS or (cmd and re.search(r"(sed -i|>\s*" + re.escape(p) + r"|\bcp\b|\bmv\b|tee )", cmd))
        if written:
            f.files_written[p] = c.index
        else:
            f.files_read.setdefault(p, c.index)


def _previous_sections(previous: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in previous.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = line.strip(), []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def _last_action_number(text: str) -> int:
    nums = [int(n) for n in re.findall(r"(?m)^(\d+)\. ", text)]
    return max(nums) if nums else 0


def _merge(previous: str, new: str, limit: int) -> str:
    """Previous section text first, then new lines not already present; cut from the old end."""
    prev_lines = [ln for ln in previous.splitlines() if ln.strip() and not ln.startswith("None")]
    seen = set(prev_lines)
    new_lines = [ln for ln in new.splitlines() if ln.strip() and ln not in seen]
    lines = prev_lines + new_lines
    if sum(len(ln) + 1 for ln in lines) > limit:
        limit -= 70  # room for the "(older N line(s) dropped ...)" note
    out, size = [], 0
    for ln in reversed(lines):
        size += len(ln) + 1
        if size > limit:
            break
        out.append(ln)
    out.reverse()
    dropped = len(lines) - len(out)
    if dropped:
        out.insert(0, f"(older {dropped} line(s) dropped for size; recover via session_search)")
    return "\n".join(out) if out else "None."


def build_summary(parsed: Parsed, model_sections: dict[str, str] | None = None) -> str:
    """Rule summary fitted to Hermes' target size. With model_sections (hybrid) the model's text opens those
    sections and the rule text follows under a label; the caps then shrink to make room for it."""
    with diag.timer("build_summary"):
        prev = _expand_abbrevs(_previous_sections(parsed.previous)) if parsed.mode == "update" else {}
        first_action = _last_action_number(prev.get("## Completed Actions", "")) + 1
        f = extract(parsed, first_action)
        abbrevs = _abbreviations(f)
        if abbrevs:  # second pass so every target is shortened before it is truncated
            f = extract(parsed, first_action, abbrevs)
        budget_chars = int(parsed.target_tokens * CHARS_PER_TOKEN)
        model_sections = model_sections or {}
        summary = _render(parsed, prev, f, abbrevs, budget_chars, _shrink_factors(1.0), model_sections)
        if len(summary) > budget_chars:
            # the highest shrink level that fits: caps shrink only as far as needed
            diag.count("build.shrunk")
            level = 0.0
            summary = _render(parsed, prev, f, abbrevs, budget_chars, _shrink_factors(level), model_sections)
            if len(summary) <= budget_chars:
                lo, hi = 0.0, 1.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    candidate = _render(parsed, prev, f, abbrevs, budget_chars, _shrink_factors(mid), model_sections)
                    if len(candidate) <= budget_chars:
                        lo, summary, level = mid, candidate, mid
                    else:
                        hi = mid
            else:
                # verbatim parts (memory provider context, user turns, model text) are never cut
                diag.count("build.over_budget")
                diag.log.info("summary %d chars over its budget of %d (target %d tokens) with every cap at its floor",
                              len(summary), budget_chars, parsed.target_tokens)
            diag.log.debug("summary fitted at shrink level %.3f: %d of %d chars", level, len(summary), budget_chars)
        unknown = [h for h in parsed.headings if h not in RULE_HEADINGS and "SKILL" not in h.upper() and "Pruned" not in h]
        diag.count("build.unknown_heading", len(unknown))
        if parsed.no_user and re.search(r"\bUser\s+asked\s*:", summary, re.I):
            summary = re.sub(r"\bUser\s+asked\s*:", "Request:", summary, flags=re.I)
            diag.count("build.no_user_rewrite")
        diag.count("build.ok")
        diag.count("build.chars", len(summary))
        return summary


RULE_HEADINGS = (HISTORICAL, "## Goal", "## Constraints & Preferences", "## Completed Actions", "## Active State",
                 "## Blocked", "## Key Decisions", "## Errors & Fixes", "## Resolved Questions", "## Relevant Files",
                 "## Critical Context", SESSION_LOG)


def _abbreviate(text: str, abbrevs: tuple) -> str:
    """Directory -> name, longest directory first, only where the path ends or goes on with "/"."""
    for path, name in abbrevs:
        text = re.sub(re.escape(path) + r"(?![\w.\-+@])", lambda _m, n=name: n, text)
    return text


def _shrink_factors(level: float) -> dict[str, float]:
    """Cap factor per SECTION_SHARE key. Level 1.0: every cap at its share. Going down to 0.0, the tiers of
    SHRINK_TIERS shrink one after the other, each down to its floor factor."""
    factors = {}
    n = len(SHRINK_TIERS)
    for i, (keys, floor) in enumerate(SHRINK_TIERS):
        # tier i shrinks while the level falls from 1 - i/n to 1 - (i+1)/n
        t = min(1.0, max(0.0, (level - (1 - (i + 1) / n)) * n))
        for key in keys:
            factors[key] = floor + (1 - floor) * t
    return factors


def _render(parsed: Parsed, prev: dict[str, str], f: Facts, abbrevs: tuple, budget_chars: int,
            factors: dict[str, float], model: dict[str, str]) -> str:
    """One summary with the section caps at their shares times `factors`; the session log takes the rest."""
    with diag.timer("build_summary.render"):
        def cap(key: str) -> int:
            return int(budget_chars * SECTION_SHARE[key] * factors.get(key, 1.0))

        sec: dict[str, str] = {}
        if parsed.no_user:
            sec[HISTORICAL] = NO_USER_SENTINEL
        else:
            latest = f.users[-1] if f.users else prev.get(HISTORICAL, "")
            sec[HISTORICAL] = _one_line(latest, 1500) if latest else "None recorded in this window."

        goal_new = ""
        if f.users and not parsed.no_user:
            goal_new = _one_line(f.users[0], 1200)
        elif not f.users:
            # The user's request sits in the protected head; manual /compress hands it over as focus.
            focus_goal = re.sub(r"^Recent user focus:\s*", "", parsed.focus).strip("- \n")
            goal_new = (_one_line(focus_goal, 1200) if focus_goal and not parsed.no_user else
                        "Continue the work recorded below (no user message in this part of the session).")
        sec["## Goal"] = prev.get("## Goal") or goal_new or "None recorded."
        if prev.get("## Goal") and f.users and not parsed.no_user:
            # never shrunk: the goal is short, and cutting it drops its oldest line, the original task
            sec["## Goal"] = _merge(prev["## Goal"], "Latest instruction: " + _one_line(f.users[-1], 800),
                                    int(budget_chars * SECTION_SHARE["## Goal"]))

        # Rules often sit at the END of a long instruction ("ab jetzt gilt ..."): keep user turns nearly whole.
        constraints = "\n".join(f"- User said: {_one_line(u, 1500)}" for u in f.users[-12:]) if not parsed.no_user else ""
        # an earlier window's completeness note no longer holds once its quotes may have been cut
        prev_constraints = USER_TURNS_NOTE_RE.sub("", prev.get("## Constraints & Preferences", ""))
        sec["## Constraints & Preferences"] = _merge(prev_constraints, constraints, cap("## Constraints & Preferences"))
        sec["## Constraints & Preferences"] += _user_turns_note(parsed, f, sec["## Constraints & Preferences"])

        # every action keeps at least its short line (command, ok/FAILED), up to ACTIONS_MAX_SHARE of the budget;
        # what that needs beyond the share comes out of the other sections through the shrink tiers
        act_limit = max(cap("## Completed Actions"), min(_short_lines_size(f.calls), int(budget_chars * ACTIONS_MAX_SHARE)))
        sec["## Completed Actions"] = _merge(prev.get("## Completed Actions", ""), _fit_actions(f.calls, act_limit), act_limit)
        sec["## Active State"] = _active_state(f)
        sec["## Blocked"] = _blocked(f)
        # the agent's own notes between tool calls carry the reasons; keep them nearly whole.
        notes = "\n".join(f"- Assistant noted: {_one_line(t, 600)}" for _, t in f.assistant_notes[-10:])
        sec["## Key Decisions"] = _merge(prev.get("## Key Decisions", ""), notes, cap("## Key Decisions"))
        errors = _merge(prev.get("## Errors & Fixes", ""), _errors_fixes(f), cap("## Errors & Fixes"))
        # added after the cap: the note is short, and with it a reader can rule out errors that are not named
        sec["## Errors & Fixes"] = ("" if errors == "None." else errors + "\n") + _exceptions_note(parsed, f)
        if parsed.no_user:
            sec["## Resolved Questions"] = prev.get("## Resolved Questions") or "None."
        else:
            qa = "\n".join(f"- Q: {_one_line(q, 240)} -> A: {_one_line(a, 300)}" for q, a in f.questions[-6:])
            sec["## Resolved Questions"] = _merge(prev.get("## Resolved Questions", ""), qa, cap("## Resolved Questions"))
        sec["## Relevant Files"] = _merge(prev.get("## Relevant Files", ""), _files(f), cap("## Relevant Files"))
        critical = []
        if parsed.focus:
            critical.append("Compaction focus (from Hermes, verbatim):\n" + parsed.focus)
        if parsed.memory:
            critical.append("Memory provider context (verbatim):\n" + parsed.memory.strip())
        answers = [t for _, t in f.assistant_notes if len(t) >= 300]
        if answers:
            # the last substantial assistant message is usually the result handed to the user
            critical.append("Last substantial assistant message (verbatim, may be cut):\n" + answers[-1][:cap("last_message")])
        if parsed.elided_notes:
            critical.append(f"{len(parsed.elided_notes)} block(s) of records were elided by Hermes before summarizing; "
                            "recover them via session_search.")
        if model:
            critical.append("Facts in this summary were extracted by rules from the recorded messages; the opening text of "
                            + ", ".join(h[3:] for h in model) + " was written by the model from a digest of them.")
        else:
            critical.append("This summary was built by rules from the recorded messages (no model call); "
                            "reasons behind decisions survive only as quoted assistant notes.")
        # the note on how the summary was built describes this summary only
        prev_critical = ORIGIN_NOTE_RE.sub("", prev.get("## Critical Context", ""))
        prev_critical = re.sub(r"(?s)Memory provider context \(verbatim\):.*?(?=\n\S|\Z)", "", prev_critical) if parsed.memory else prev_critical
        sec["## Critical Context"] = (_merge(prev_critical, "\n".join(critical), cap("## Critical Context"))
                                      if prev_critical.strip() else "\n".join(critical))

        for heading, text in model.items():
            rule_text = sec.get(heading, "")
            if rule_text and (heading == "## Active State" or not rule_text.startswith("None")):
                text = f"{text}\n\n{MODEL_SECTION_LABELS[heading]}\n{rule_text}"
            sec[heading] = text.strip()

        if abbrevs:
            # long directories repeated on every line are pure noise; name them once
            # Relevant Files keeps full paths: it is the list the model and session_search look up.
            for key in WD_SECTIONS:
                sec[key] = _abbreviate(sec[key], abbrevs)
            defs = []
            for path, name in sorted(abbrevs, key=lambda a: a[1] != "$WD"):
                what = "working directory of this part of the session" if name == "$WD" else "used in commands and paths below"
                defs.append(f"- {name} = {path} ({what})\n")
            sec["## Active State"] = "".join(defs) + sec["## Active State"]

        log_new = _abbreviate("\n".join(render_log(f, "\n".join(sec.values()))), abbrevs)
        log_prev = prev.get(SESSION_LOG, "")
        markers = list(dict.fromkeys(f.skill_markers + SKILL_PRUNED_RE.findall(parsed.previous)))
        # headings, blank lines and the pruned-skill block count towards the target too
        overhead = sum(len(h) + 3 for h in parsed.headings) + sum(len(m) + 1 for m in markers)
        used = sum(len(v) for v in sec.values()) + overhead
        log_budget = max(cap("log"), budget_chars - used)
        slack = log_budget - len(_merge(log_prev, log_new, budget_chars * 10))
        actions = sec["## Completed Actions"]
        if slack > 0 and any(m in actions for m in ("Older actions ", "Earlier actions ", "line(s) dropped for size")):
            # the compact log left room: shorten fewer of the older actions
            act_limit += slack
            sec["## Completed Actions"] = _abbreviate(
                _merge(prev.get("## Completed Actions", ""), _fit_actions(f.calls, act_limit), act_limit), abbrevs)
            diag.count("build.actions_widened")
            used = sum(len(v) for v in sec.values()) + overhead
            log_budget = max(cap("log"), budget_chars - used)
        sec[SESSION_LOG] = _merge(log_prev, log_new, log_budget)

        out = []
        for heading in parsed.headings:
            if heading in sec:
                out.append(f"{heading}\n{sec[heading]}")
            elif "SKILL" in heading.upper() or "Pruned" in heading:
                if markers:
                    out.append(heading + "\n" + "\n".join(markers))
            else:
                out.append(f"{heading}\n{prev.get(heading) or 'None recorded.'}")
        return "\n\n".join(out).strip()


def _actions(calls: list[Call]) -> str:
    lines = []
    i = 0
    while i < len(calls):
        c = calls[i]
        j = i
        # repeats of a call fold into one line with the last outcome; every file change keeps its own line,
        # its outcome (diff size, changed definitions) differs from call to call
        while (c.tool not in WRITE_TOOLS and j + 1 < len(calls) and calls[j + 1].tool == c.tool
               and calls[j + 1].target == c.target and calls[j + 1].ok == c.ok):
            j += 1
        repeat = f" (x{j - i + 1}, through #{calls[j].index})" if j > i else ""
        last = calls[j]
        lines.append(f"{c.index}. {c.tool.upper()} {c.target} — {_one_line(last.outcome, 200)}{repeat} [tool: {c.tool}]")
        i = j + 1
    return "\n".join(lines)


def _compact_actions(calls: list[Call]) -> str:
    """One line per call (repeats of a call collapse): number, tool, target and the outcome without output."""
    lines = []
    i = 0
    while i < len(calls):
        c, j = calls[i], i
        while (c.tool not in WRITE_TOOLS and j + 1 < len(calls) and calls[j + 1].tool == c.tool
               and calls[j + 1].target == c.target and calls[j + 1].ok == c.ok):
            j += 1
        last = calls[j]
        state = ("ok" if c.ok else f"FAILED: {_one_line(last.error or last.outcome, 100)}" if c.ok is False
                 else "outcome unknown")
        repeat = f" (x{j - i + 1}, through #{last.index})" if j > i else ""
        lines.append(f"{c.index}. {c.tool} {c.target}: {state}{repeat}")
        i = j + 1
    return "\n".join(lines)


def _folded_actions(calls: list[Call]) -> str:
    """One line per target with ok/failed/unknown counts and the last outcome."""
    groups: dict[tuple[str, str], list[Call]] = {}
    for c in calls:
        groups.setdefault((c.tool, _one_line(c.target, 110)), []).append(c)
    lines = [f"Earlier actions {calls[0].index}-{calls[-1].index}, folded per target (ok/failed/unknown, last outcome):"]
    for (tool, target), cs in groups.items():
        ok = sum(1 for c in cs if c.ok)
        bad = sum(1 for c in cs if c.ok is False)
        last = cs[-1]
        lines.append(f"- {tool} {target}: {len(cs)}x ({ok}/{bad}/{len(cs) - ok - bad}), last #{last.index} "
                     f"{'ok' if last.ok else 'FAILED' if last.ok is False else 'unknown'}"
                     + (f": {_one_line(last.error, 120)}" if last.ok is False and last.error else ""))
    return "\n".join(lines)


def _short_lines_size(calls: list[Call]) -> int:
    """Size of the smallest list that still names every action: short lines, the newest 4 in full."""
    if len(calls) <= 4:
        return len(_actions(calls))
    return len(f"Older actions {calls[0].index}-{calls[-5].index}, outcome only:\n") + len(_compact_actions(calls[:-4])) \
        + 1 + len(_actions(calls[-4:]))


def _fit_actions(calls: list[Call], limit: int) -> str:
    """Every action stays answerable ('did X succeed') within `limit`, in three tiers: the newest with their
    output excerpt, older ones as one short line each (command and ok/FAILED), the oldest folded per target."""
    full = _actions(calls)
    if len(full) <= limit or len(calls) < 8:
        return full
    keep = len(calls)
    while keep > 4:
        keep = max(4, int(keep * 0.7))
        older, newer = calls[:-keep], calls[-keep:]
        text = f"Older actions {older[0].index}-{older[-1].index}, outcome only:\n{_compact_actions(older)}\n{_actions(newer)}"
        if len(text) <= limit:
            diag.count("build.actions_compact")
            return text
    # even one short line per action is too long: fold the oldest per target, the rest stays one line each
    older, newer = calls[:-4], calls[-4:]
    for share in (0.2, 0.4, 0.6, 0.8, 1.0):
        n = max(1, int(len(older) * share))
        folded, compact = older[:n], older[n:]
        text = _folded_actions(folded) + "\n"
        if compact:
            text += f"Older actions {compact[0].index}-{compact[-1].index}, outcome only:\n{_compact_actions(compact)}\n"
        text += _actions(newer)
        if len(text) <= limit:
            break
    diag.count("build.actions_folded")
    return text


def _later_success(f: Facts, failed: Call) -> Call | None:
    for c in f.calls:
        if c.index > failed.index and c.ok and c.tool == failed.tool and c.target == failed.target:
            return c
    return None


def _errors_fixes(f: Facts) -> str:
    lines = []
    for c in f.calls:
        if c.ok is False:
            fix = _later_success(f, c)
            tail = f"fixed: same call succeeded at action {fix.index}" if fix else "no identical successful retry in this window"
            lines.append(f"- action {c.index} {c.tool} {c.target}: {c.error or c.outcome} -> {tail}")
    return "\n".join(lines[-25:])


def _user_turns_note(parsed: Parsed, f: Facts, section: str) -> str:
    """A completeness line, only when every user turn of the window stands uncut in `section`."""
    if parsed.no_user or not f.users:
        return ""
    if all(" ".join(u.split()) in section for u in f.users):
        diag.count("build.user_turns_complete")
        n = len(f.users)
        return f"\n- (All {n} user turn{'s' if n > 1 else ''} of this window are quoted above in full, whitespace collapsed.)"
    diag.count("build.user_turns_cut")
    return ""


def _exceptions_note(parsed: Parsed, f: Facts) -> str:
    """Every exception name of the window: what is not named here did not occur in the records Hermes passed on."""
    span = f"actions #{f.calls[0].index}-#{f.calls[-1].index}" if f.calls else "no actions"
    scope = f", not counting the {len(parsed.elided_notes)} block(s) Hermes elided" if parsed.elided_notes else ""
    return (f"- Every exception reported as 'Name: message' in this window ({span}{scope}): "
            + (", ".join(f.exceptions) if f.exceptions else "none"))


def _blocked(f: Facts) -> str:
    open_fail = [c for c in f.calls if c.ok is False and not _later_success(f, c)]
    if not open_fail:
        return "None recorded in this window."
    last_ok_record = max((c.record for c in f.calls if c.ok), default=-1)
    lines = []
    for c in open_fail[-8:]:
        note = "" if c.record > last_ok_record else " (later actions succeeded; may be superseded)"
        lines.append(f"- action {c.index} {c.tool} {c.target}: {c.error or c.outcome}{note}")
    return "\n".join(lines)


def _active_state(f: Facts) -> str:
    lines = []
    cwd = None
    for c in f.calls:
        cmd = str(c.args.get("command") or "")
        m = re.findall(r"(?:^|&&|;)\s*cd\s+([^\s;&]+)", cmd)
        if m:
            cwd = m[-1]
    if cwd:
        lines.append(f"- Last working directory used in a command: {cwd}")
    if f.files_written:
        lines.append("- Files changed in this window (last action #):")
        for p, n in list(f.files_written.items())[-15:]:
            lines.append(f"  - {p} (#{n})")
    if f.calls:
        # the newest actions stand in full in Completed Actions (never folded); only the last one is repeated here
        last = f.calls[-1]
        first = f.calls[-6] if len(f.calls) >= 6 else f.calls[0]
        lines.append(f"- Most recent actions: #{first.index}-#{last.index} (see Completed Actions); last: "
                     f"#{last.index} {last.tool} {last.target}: {_one_line(last.outcome, 160)}")
    return "\n".join(lines) or "Unknown from this window; inspect current state."


def _common_dir(f: Facts) -> str:
    """Most used long directory (cd targets and parents of touched files), or ''."""
    counts: dict[str, int] = {}
    for c in f.calls:
        cmd = str(c.args.get("command") or "")
        for d in re.findall(r"\bcd\s+(/[^\s;&|]+)", cmd):
            counts[d.rstrip("/")] = counts.get(d.rstrip("/"), 0) + 1
    for p in list(f.files_written) + list(f.files_read):
        if p.startswith("/"):
            d = p.rsplit("/", 1)[0]
            counts[d] = counts.get(d, 0) + 1
    best = max(((n, d) for d, n in counts.items() if len(d) >= 25 and n >= 3), default=None)
    return best[1] if best else ""


MAX_DIR_ABBREVS = 3


def _abbreviations(f: Facts) -> tuple:
    """(directory, name) pairs, longest first: the working directory as $WD and up to MAX_DIR_ABBREVS other
    long directories that many calls name, as $D1..; the best saves its length on the most lines."""
    with diag.timer("abbreviations"):
        wd = _common_dir(f)
        counts: dict[str, int] = {}
        texts_all = []
        for c in f.calls:
            texts = [str(c.args.get("command") or c.args.get("cmd") or "")]
            texts += [c.args[k] for k in ("path", "file_path", "filename", "file") if isinstance(c.args.get(k), str)]
            texts_all += texts
            dirs = set()
            for p in (p for t in texts for p in PATH_RE.findall(t) if p.startswith("/")):
                parts = p.rstrip("/").split("/")
                dirs.update("/".join(parts[:i]) for i in range(3, len(parts) + 1))
            for d in dirs:
                counts[d] = counts.get(d, 0) + 1
        chosen: list[str] = []
        for d, n in sorted(counts.items(), key=lambda kv: -(len(kv[0]) - 3) * kv[1]):
            if len(chosen) == MAX_DIR_ABBREVS:
                break
            if len(d) < 25 or n < 4 or (wd and (d == wd or d.startswith(wd + "/"))):
                continue
            if any(d.startswith(x + "/") or x.startswith(d + "/") for x in chosen):
                continue
            chosen.append(d)
        joined = "\n".join(texts_all)
        names = [f"$D{i}" for i in range(1, 10) if f"$D{i}" not in joined]  # never a name the commands use
        pairs = ([(wd, "$WD")] if wd else []) + list(zip(chosen, names))
        diag.count("build.dir_abbrevs", len(pairs) - bool(wd))
        return tuple(sorted(pairs, key=lambda pn: -len(pn[0])))


def _expand_abbrevs(prev: dict[str, str]) -> dict[str, str]:
    """Back to full paths in an earlier summary: this summary may name other directories."""
    defs = re.findall(r"(?m)^- (\$WD|\$D\d) = (\S+)", prev.get("## Active State", ""))
    for name, path in defs:
        prev = {k: re.sub(re.escape(name) + r"(?![A-Za-z0-9_])", lambda _m, p=path: p, v) for k, v in prev.items()}
    return prev


def _files(f: Facts) -> str:
    written = [f"- {p} — modified (last at action {n})" for p, n in list(f.files_written.items())[-25:]]
    # "no write seen": no write tool and no shell write (>, tee, sed -i, cp, mv) on it in this window
    read = [f"- {p} — read/used, no write seen (first at action {n})" for p, n in list(f.files_read.items())[-25:]
            if p not in f.files_written]
    # modified files last: a section over its cap is cut from the top, so read-only files go first
    return "\n".join(read[:max(0, 40 - len(written))] + written)


def summarize(prompt: str) -> str | None:
    """Full pipeline; None means: forward to the model."""
    parsed = parse_prompt(prompt)
    if parsed is None:
        return None
    return summarize_parsed(parsed)


def summarize_parsed(parsed: Parsed, model_sections: dict[str, str] | None = None) -> str | None:
    try:
        return build_summary(parsed, model_sections)
    except Exception:  # never break a compaction; the model path takes over
        diag.count("build.exception")
        diag.log.exception("build_summary failed; forwarding to the model")
        return None
