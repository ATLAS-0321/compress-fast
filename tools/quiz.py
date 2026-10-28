"""Recall quiz for a shadow pair: can the model answer facts about the compacted turns
from the model's own summary vs from the rule summary (and hybrid / VCC)?

Questions are generated from the original turns in the stored prompt (ground truth) and asked to the
model with only one summary as context. Fact questions (call outcomes, files changed or only read,
errors seen, words of the user) are graded by exact yes/no match. Negative controls (a made-up file,
error or word) only check that a summary does not claim what did not happen: "no" and "unknown"
both pass, "yes" is a false claim. Counting "no" as better would reward a summary that sounds
complete without being able to know (a script can write a file the summary never saw).

Usage: python3 tools/quiz.py STAMP [--model-url http://127.0.0.1:8080/v1] [--model NAME] [--seed 7]
                                     [--hybrid] [--vcc-only]
       python3 tools/quiz.py --table      totals per variant over every stored <STAMP>.quiz.json
Writes <shadow>/<STAMP>.quiz.json and prints a short table.
"""
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from compress_fast import summarizer  # noqa: E402
from compare_shadow import SHADOW, THINK_RE, model_text  # noqa: E402

FAKE_ERRORS = ["ZeroDivisionError", "UnicodeDecodeError", "RecursionError", "KeyboardInterrupt"]
FAKE_FILES = ["/etc/hermes/override.yaml", "/opt/hermes/skills/devops/backup/scripts/run-backup",
              "/tmp/report_final_v2.md"]


SHOWN_TARGET_CHARS = 140


def _last_outcomes(facts, tool: str, shown: str) -> set:
    """Last outcome of every distinct call target that the shown (cut) target text fits."""
    last = {}
    for c in facts.calls:
        if c.tool == tool and c.target.startswith(shown):
            last[c.target] = c.ok
    return set(last.values())


def void_questions(parsed, qs: list[dict]) -> set[int]:
    """Call questions whose shown target fits several calls that ended differently: the right answer depends
    on which call is meant, so the question counts for no variant."""
    facts = summarizer.extract(parsed, 1)
    void = set()
    for i, q in enumerate(qs):
        m = re.match(r"Did the (\S+) call on (.*) succeed \(its last run\)\?$", q["q"], re.S)
        if q["kind"] == "call outcome" and m and len(_last_outcomes(facts, m.group(1), m.group(2))) > 1:
            void.add(i)
    return void


def build_questions(parsed, rnd: random.Random) -> list[dict]:
    facts = summarizer.extract(parsed, 1)
    qs: list[dict] = []
    known = [c for c in facts.calls if c.ok is not None]
    fails = [c for c in known if c.ok is False]
    oks = [c for c in known if c.ok]
    # only calls with a readable target (truncated JSON args make unanswerable questions) whose shown text
    # does not also fit a call that ended differently
    fails = [c for c in fails if not c.target.startswith("{")
             and len(_last_outcomes(facts, c.tool, c.target[:SHOWN_TARGET_CHARS])) == 1]
    oks = [c for c in oks if not c.target.startswith("{")
           and len(_last_outcomes(facts, c.tool, c.target[:SHOWN_TARGET_CHARS])) == 1]
    pick = rnd.sample(fails, min(4, len(fails))) + rnd.sample(oks, min(8 - min(4, len(fails)), len(oks)))
    last_ok = {}
    for c in known:  # the question asks about the last run of that exact call
        last_ok[(c.tool, c.target)] = c.ok
    seen = set()
    for c in pick:
        key = (c.tool, c.target)
        if key in seen:
            continue
        seen.add(key)
        qs.append({"kind": "call outcome", "q": f"Did the {c.tool} call on {c.target[:SHOWN_TARGET_CHARS]} succeed (its last run)?",
                   "a": "yes" if last_ok[key] else "no"})
    written = list(facts.files_written)
    # "not modified" is only safe for files touched by read tools alone and never named by any
    # other call (code or shell may write them without a write tool)
    other_args = " ".join(c.args_raw for c in facts.calls if c.tool not in summarizer.READ_TOOLS)
    read_only = sorted({str(c.args.get("path")) for c in facts.calls
                        if c.tool in summarizer.READ_TOOLS and c.args.get("path")
                        and str(c.args.get("path")) not in facts.files_written
                        and str(c.args.get("path")) not in other_args})
    for p in rnd.sample(written, min(3, len(written))):
        qs.append({"kind": "file changed", "q": f"Was the file {p} written or modified?", "a": "yes"})
    for p in rnd.sample(read_only, min(3, len(read_only))):
        qs.append({"kind": "file changed", "q": f"Was the file {p} written or modified (not only read)?", "a": "no"})
    qs.append({"kind": "control", "q": f"Was the file {rnd.choice(FAKE_FILES)} written or modified?", "a": "no"})
    turns = "\n".join(c for _, c in parsed.records)
    errs = sorted({m.group(1) for m in summarizer.EXC_RE.finditer(turns)})
    for e in errs[:3]:
        qs.append({"kind": "error seen", "q": f"Did a {e} occur?", "a": "yes"})
    fake = [e for e in FAKE_ERRORS if e not in turns]
    qs.append({"kind": "control", "q": f"Did a {rnd.choice(fake)} occur?", "a": "no"})
    for u in facts.users[-3:]:
        words = [w for w in re.findall(r"[A-Za-zÄÖÜäöüß][\wäöüß\-]{6,}", u) if not w.startswith("http")]
        if words:
            w = rnd.choice(words)
            qs.append({"kind": "user words", "q": f"Did the user's own instructions contain the word '{w}'?", "a": "yes"})
    qs.append({"kind": "control", "q": "Did the user's own instructions contain the word 'Kubernetes'?", "a": "no"})
    rnd.shuffle(qs)
    return qs


def ask(url: str, model: str, summary: str, qs: list[dict]) -> tuple[list[str], float, str]:
    numbered = "\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(qs))
    prompt = ("Below is a summary of earlier work in an agent session. Answer each question ONLY from this summary. "
              "Answer yes, no, or unknown when the summary does not say. Reply with exactly one line per question in the "
              f"form 'N: yes|no|unknown' and nothing else.\n\n<summary>\n{summary}\n</summary>\n\nQuestions:\n{numbered}")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": 4000, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    secs = time.perf_counter() - t
    text = THINK_RE.sub("", data["choices"][0]["message"]["content"] or "")
    answers = ["missing"] * len(qs)
    for m in re.finditer(r"(?m)^\s*(\d+)\s*[:.)]\s*(yes|no|unknown)", text, re.I):
        i = int(m.group(1)) - 1
        if 0 <= i < len(qs):
            answers[i] = m.group(2).lower()
    return answers, secs, text


def grade(answers: list[str], qs: list[dict], void: set[int] = frozenset()) -> dict:
    """Facts by exact match; controls only fail on a false "yes"; void questions do not count."""
    facts = [(a, q) for i, (a, q) in enumerate(zip(answers, qs)) if q["kind"] != "control" and i not in void]
    controls = [a for a, q in zip(answers, qs) if q["kind"] == "control"]
    return {"facts_correct": sum(a == q["a"] for a, q in facts), "facts_total": len(facts),
            "facts_wrong": sum(a in ("yes", "no") and a != q["a"] for a, q in facts),
            "controls_total": len(controls), "controls_false_yes": controls.count("yes"),
            "controls_no": controls.count("no"), "void": len(void)}


def table() -> None:
    totals: dict[str, dict] = {}
    for path in sorted(SHADOW.glob("*.quiz.json")):
        result = json.loads(path.read_text())
        parsed = summarizer.parse_prompt((SHADOW / path.name.replace(".quiz.json", ".prompt.txt")).read_text())
        void = void_questions(parsed, result["questions"])
        for name, run in result["runs"].items():
            g = grade(run["answers"], result["questions"], void)
            t = totals.setdefault(name, dict.fromkeys(g, 0))
            for key, value in g.items():
                t[key] += value
    print(f"{'variant':8} {'facts right':>12} {'facts wrong':>12} {'controls false yes':>19} {'controls no':>12}")
    for name, t in totals.items():
        print(f"{name:8} {t['facts_correct']:>7}/{t['facts_total']:<4} {t['facts_wrong']:>12} "
              f"{t['controls_false_yes']:>14}/{t['controls_total']:<4} {t['controls_no']:>12}")


def main() -> None:
    if "--table" in sys.argv:
        table()
        return
    stamp = sys.argv[1]
    url = sys.argv[sys.argv.index("--model-url") + 1] if "--model-url" in sys.argv else "http://127.0.0.1:8080/v1"
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
    prompt = (SHADOW / f"{stamp}.prompt.txt").read_text()
    parsed = summarizer.parse_prompt(prompt)
    if parsed is None:
        sys.exit("prompt not parseable")
    own = THINK_RE.sub("", model_text((SHADOW / f"{stamp}.model.raw").read_text())).strip()
    rules = summarizer.summarize(prompt) or ""
    qs = build_questions(parsed, random.Random(seed))
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else ""
    if not model:
        try:  # an idle/restarted engine may list no model until the first request loads it
            model = json.loads(urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10).read())["data"][0]["id"]
        except (IndexError, KeyError, OSError, ValueError):
            sys.exit("no model listed at --model-url; pass --model NAME")
    result = {"stamp": stamp, "questions": qs, "runs": {}}
    runs = [("none", "(no summary available)"), ("model", own), ("rules", rules)]
    out = SHADOW / f"{stamp}.quiz.json"
    if "--vcc-only" in sys.argv:
        # add hermes-vcc-engine's projection (<stamp>.vcc.txt, made by vcc_bridge.py) to an existing quiz,
        # asking the stored questions so the earlier runs stay comparable
        result = json.loads(out.read_text())
        qs = result["questions"]
        runs = [("vcc", (SHADOW / f"{stamp}.vcc.txt").read_text())]
    if "--hybrid" in sys.argv and "--vcc-only" not in sys.argv:
        from compress_fast import hybrid
        hyb, info = hybrid.summarize_hybrid(prompt, url.rsplit("/v1", 1)[0], model, timeout=300)
        result["hybrid_info"] = info
        print(f"{stamp} hybrid built: {info}")
        (SHADOW / f"{stamp}.hybrid.txt").write_text(hyb or "")
        (SHADOW / f"{stamp}.hybrid.txt").chmod(0o600)
        runs.append(("hybrid", hyb or ""))
    void = void_questions(parsed, qs)
    # "none" = no summary at all: how much the model gets right by guessing / general knowledge
    for name, summary in runs:
        answers, secs, raw = ask(url, model, summary, qs)
        right = sum(a == q["a"] for a, q in zip(answers, qs))
        unknown = sum(a == "unknown" for a in answers)
        g = grade(answers, qs, void)
        result["runs"][name] = {"answers": answers, "correct": right, "total": len(qs), "unknown": unknown, **g,
                                "seconds": round(secs, 1), "summary_chars": len(summary), "raw": raw}
        print(f"{stamp} {name:6} facts {g['facts_correct']}/{g['facts_total']} ({g['facts_wrong']} wrong), "
              f"controls {g['controls_false_yes']}/{g['controls_total']} false yes, {secs:.0f} s")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    out.chmod(0o600)


if __name__ == "__main__":
    main()
