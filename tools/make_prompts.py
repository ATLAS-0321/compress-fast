"""Build real Hermes summary prompts from state.db for offline tests.

Test tool only: it imports Hermes internals (ContextCompressor) to reproduce the exact
prompt text Hermes sends to auxiliary.compression. The service itself never does this.

Usage (Hermes interpreter):
  python3 tools/make_prompts.py <session_id> <first_id> <last_id> <out.txt> [--previous <file>]
"""
import json
import sqlite3
import sys
from pathlib import Path

HERMES = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES))

from agent.context_compressor import ContextCompressor  # noqa: E402

DB = Path.home() / ".hermes" / "state.db"


def load_turns(session_id: str, first_id: int, last_id: int) -> list[dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "select role, content, tool_call_id, tool_calls from messages "
        "where session_id=? and id between ? and ? order by id",
        (session_id, first_id, last_id),
    ).fetchall()
    con.close()
    turns = []
    for role, content, tool_call_id, tool_calls in rows:
        msg = {"role": role, "content": content or ""}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if tool_calls:
            try:
                msg["tool_calls"] = json.loads(tool_calls)
            except ValueError:
                pass
        turns.append(msg)
    return turns


def build_prompt(turns: list[dict], previous: str = "", memory_context: str = "") -> str:
    cc = ContextCompressor.__new__(ContextCompressor)
    cc.tail_mode = "lean"
    cc._previous_summary = previous or None
    cc.summary_target_ratio = 0.20
    cc.context_length = 204800
    cc.threshold_tokens = 150118
    records = cc._serialize_records_for_summary(turns)
    content, _coverage = cc._sample_summary_records(records)
    has_user = any(t["role"] == "user" for t in turns)
    budget = 3000
    return cc._build_summary_prompt(content, budget, None, memory_context, has_user)


def main() -> None:
    session_id, first_id, last_id, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    previous = ""
    if "--previous" in sys.argv:
        previous = Path(sys.argv[sys.argv.index("--previous") + 1]).read_text()
    memory = ""
    if "--memory" in sys.argv:
        memory = Path(sys.argv[sys.argv.index("--memory") + 1]).read_text()
    turns = load_turns(session_id, first_id, last_id)
    prompt = build_prompt(turns, previous, memory)
    Path(out).write_text(prompt)
    print(f"{out}: {len(turns)} turns, {len(prompt)} chars")


if __name__ == "__main__":
    main()
