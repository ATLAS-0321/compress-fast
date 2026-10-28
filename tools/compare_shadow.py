"""Compare a shadow pair: model's summary vs the rule summary for the same Hermes prompt.

Usage: python3 tools/compare_shadow.py [STAMP ...]   (default: every pair in the shadow dir)
Prints a plain-text report: section coverage, fact recall against the prompt, and identifiers
that only one side kept.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compress_fast import summarizer  # noqa: E402

SHADOW = Path.home() / ".hermes" / "compress-fast-shadow"
PATH_RE = re.compile(r"(/(?:home|media|tmp|etc|usr|mnt|opt)[\w.\-/]+\.\w+)")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|ms|s|m|cm|mm|px|GB|MB|GiB|MiB|tok/s|tris|Tris|Objekte|objects)\b")
TICK_RE = re.compile(r"`([^`\n]{3,80})`")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def model_text(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return raw
    parts = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            try:
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                parts.append(delta.get("content") or "")
            except (ValueError, KeyError, IndexError):
                continue
    return "".join(parts)


def sections(text: str) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = line.strip(), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def recall(items: set[str], text: str) -> tuple[int, int, list[str]]:
    missing = sorted(i for i in items if i not in text)
    return len(items) - len(missing), len(items), missing


def report(stamp: str) -> str:
    prompt = (SHADOW / f"{stamp}.prompt.txt").read_text()
    rules = (SHADOW / f"{stamp}.rules.txt").read_text()
    recomputed = False
    if not rules or "--recompute" in sys.argv:
        rules = summarizer.summarize(prompt) or ""
        recomputed = True
    model = THINK_RE.sub("", model_text((SHADOW / f"{stamp}.model.raw").read_text())).strip()
    parsed = summarizer.parse_prompt(prompt)
    lines = [f"=== {stamp} ===", f"prompt {len(prompt):,} chars | model {len(model):,} chars | rules {len(rules):,} chars"]
    if recomputed:
        lines.append("rules: recomputed from the stored prompt with the current code")
    if not rules:
        lines.append("rules: prompt NOT handled (would have gone to the model)")
    if not model:
        lines.append("model: no summary text captured")
        return "\n".join(lines)
    if parsed:
        facts = summarizer.extract(parsed, 1)
        turns = "\n".join(c for _, c in parsed.records)
        paths = {p.rstrip(".") for p in PATH_RE.findall(turns)}
        errors = {f"{m.group(1)}: {m.group(2)}"[:60] for m in summarizer.EXC_RE.finditer(turns)}
        users = {u[:80] for u in facts.users}
        lines.append(f"window: {len(parsed.records)} records, {len(facts.calls)} tool calls, mode {parsed.mode}")
        for label, items in (("file paths", paths), ("error lines", errors), ("user messages (first 80 chars)", users)):
            for side, text in (("model", model), ("rules", rules)):
                hit, total, _ = recall(items, text)
                lines.append(f"  {label:32} {side:6} {hit}/{total}")
        failed = [c for c in facts.calls if c.ok is False]
        lines.append(f"  failed tool calls in window: {len(failed)}")
    ms, rs = sections(model), sections(rules)
    lines.append("sections: " + ", ".join(f"{h[3:]}: S{len(ms.get(h, '')):,}/R{len(rs.get(h, '')):,}"
                                           for h in dict.fromkeys(list(ms) + list(rs))))
    only_model = sorted({x for x in TICK_RE.findall(model) + NUM_RE.findall(model)} - set(TICK_RE.findall(rules) + NUM_RE.findall(rules)))
    only_model = [x for x in only_model if x not in rules]
    lines.append(f"identifiers/values only in model's summary ({len(only_model)}):")
    lines += [f"  - {x}" for x in only_model[:40]]
    return "\n".join(lines)


def main() -> None:
    stamps = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted({p.name.split(".")[0] for p in SHADOW.glob("*.prompt.txt")})
    for s in stamps:
        print(report(s))
        print()


if __name__ == "__main__":
    main()
