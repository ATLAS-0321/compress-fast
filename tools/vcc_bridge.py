"""Run hermes-vcc-engine's projector (stdlib-only, reviewed) on a stored Hermes prompt for comparison.

The prompt's serialized records are turned back into OpenAI-format messages (the same pruned view
Hermes summarizes), then vcc.projector.project() builds its summary with the same character budget
the rule summary gets.

Usage: python3 tools/vcc_bridge.py <prompt.txt> <path to vcc/projector.py>
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compress_fast import summarizer  # noqa: E402


def load_projector(path: str):
    spec = importlib.util.spec_from_file_location("vcc_projector", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vcc_projector"] = mod
    spec.loader.exec_module(mod)
    return mod


def messages_from_prompt(parsed) -> list[dict]:
    msgs: list[dict] = []
    pending: list[str] = []
    n = 0
    for role, content in parsed.records:
        n += 1
        if role == "USER":
            msgs.append({"role": "user", "content": content, "_row_id": n})
        elif role == "ASSISTANT":
            text, calls = summarizer._parse_calls(content)
            msg = {"role": "assistant", "content": text, "_row_id": n}
            if calls:
                msg["tool_calls"] = []
                pending = []
                for i, c in enumerate(calls):
                    cid = f"call_{n}_{i}"
                    pending.append(cid)
                    msg["tool_calls"].append({"id": cid, "type": "function",
                                              "function": {"name": c.tool, "arguments": c.args_raw}})
            msgs.append(msg)
        elif role.startswith("TOOL RESULT"):
            cid = pending.pop(0) if pending else f"orphan_{n}"
            msgs.append({"role": "tool", "tool_call_id": cid, "content": content, "_row_id": n})
    return msgs


def vcc_summary(prompt: str, projector_path: str) -> str:
    parsed = summarizer.parse_prompt(prompt)
    if parsed is None:
        return ""
    proj = load_projector(projector_path)
    budget = int(parsed.target_tokens * summarizer.CHARS_PER_TOKEN)
    return proj.project(messages_from_prompt(parsed), budget_chars=budget, previous=parsed.previous or None,
                        focus=parsed.focus or None, memory_context=parsed.memory)


if __name__ == "__main__":
    print(vcc_summary(Path(sys.argv[1]).read_text(), sys.argv[2]))
