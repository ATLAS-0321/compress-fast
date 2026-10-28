"""Build the synthetic test fixtures in tests/fixtures/ with Hermes' own prompt builder.

The sessions below are invented (a small disk report script, a subagent run without user turns),
but every message uses the shapes Hermes stores: OpenAI tool_calls, JSON tool results of the
terminal/read_file/write_file/patch/search_files tools, out-of-band stub results. Building the
prompt with ContextCompressor gives the exact text Hermes sends to auxiliary.compression, so the
tests run on real prompt text without shipping anyone's real sessions.

Usage (Hermes interpreter, needs a Hermes checkout):
  <hermes venv>/bin/python tools/make_fixtures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_prompts import build_prompt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
WD = "/home/user/projects/diskreport"


class Session:
    def __init__(self) -> None:
        self.turns: list[dict] = []
        self.n = 0

    def user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def call(self, name: str, args: dict, result, note: str = "") -> None:
        self.n += 1
        cid = f"call_{self.n:04d}"
        self.turns.append({"role": "assistant", "content": note, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        content = result if isinstance(result, str) else json.dumps(result)
        self.turns.append({"role": "tool", "tool_call_id": cid, "content": content})

    def say(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})


def term(output: str, code: int = 0) -> dict:
    return {"output": output, "exit_code": code, "error": None}


def wrote(path: str, size: int) -> dict:
    return {"bytes_written": size, "dirs_created": False, "verified": True, "lint": {"status": "ok", "output": ""},
            "resolved_path": path, "files_modified": [path]}


def patched(path: str, minus: str, plus: str) -> dict:
    diff = f"--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,3 @@\n-{minus}\n+{plus}\n"
    return {"success": True, "diff": diff, "files_modified": [path], "lint": {"status": "ok", "output": ""},
            "resolved_path": path}


def disk_session() -> Session:
    s = Session()
    script, test, readme = f"{WD}/disk_report.py", f"{WD}/test_disk_report.py", f"{WD}/README.md"
    s.user("Write a small Python script disk_report.py in the current folder (standard library only). "
           "For /, /srv/data and /mnt/backup print total, used and free space in GiB as a table, and at the "
           "end the path with the least free space. A missing path must not crash the script, just mark it as "
           "missing. Keep the output in English.")
    s.call("terminal", {"command": f"cd {WD} && pwd && ls -la && df -h / /srv/data /mnt/backup"},
           term(f"{WD}\ntotal 8\ndrwxr-xr-x 2 user user 4096 .\n"
                "df: /mnt/backup: No such file or directory\n"
                "Filesystem Size Used Avail Use% Mounted on\n/dev/nvme0n1p2 916G 820G 87G 91% /\n"
                "/dev/sda1 1.8T 1.1T 700G 61% /srv/data", 1))
    s.call("read_file", {"path": readme},
           {"content": "# diskreport\n\nSmall tools for the backup host.\n", "total_lines": 3, "file_size": 48,
            "truncated": False, "is_binary": False, "is_image": False, "not_found": False},
           "Folder is empty apart from the README; /mnt/backup is not mounted, good test for the missing case.")
    s.call("write_file", {"path": script, "content": "import shutil\n..."}, wrote(script, 812))
    s.call("terminal", {"command": f"cd {WD} && python3 disk_report.py"},
           term("Traceback (most recent call last):\n  File \"disk_report.py\", line 14, in <module>\n"
                "    usage = shutil.disk_usage(path)\nFileNotFoundError: [Errno 2] No such file or directory: "
                "'/mnt/backup'", 1),
           "First run.")
    s.call("patch", {"path": script, "old_string": "usage = shutil.disk_usage(path)",
                     "new_string": "try:\n    usage = shutil.disk_usage(path)\nexcept OSError:\n    rows.append((path, None))"},
           patched(script, "usage = shutil.disk_usage(path)", "try: usage = shutil.disk_usage(path) except OSError"),
           "Missing path crashes with FileNotFoundError; catching OSError and marking the row as missing.")
    s.call("terminal", {"command": f"cd {WD} && python3 disk_report.py"},
           term("Path        Total (GiB)  Used (GiB)  Free (GiB)\n/           915.35       819.14      86.88\n"
                "/srv/data   1832.67      1132.67     700.00\n/mnt/backup missing\n\n"
                "Least free: / (86.88 GiB)"))
    s.say("The script works: / has the least free space with 86.88 GiB, /mnt/backup is reported as missing.")
    s.user("Good. Add a --json option and a --min-free GIB option that exits with code 2 when any path is "
           "below the limit. Write a unittest for both. Do not touch the README.")
    s.call("patch", {"path": script, "old_string": "def main():", "new_string": "def main(argv=None):"},
           patched(script, "def main():", "def main(argv=None):  # argparse: --json, --min-free"))
    s.call("write_file", {"path": test, "content": "import unittest\n..."}, wrote(test, 1404))
    s.call("terminal", {"command": f"cd {WD} && python3 -m unittest -q test_disk_report"},
           term("F.\n======================================================================\n"
                "FAIL: test_min_free_exit_code (test_disk_report.ReportTest)\n"
                "AssertionError: 0 != 2\n----------------------------------------------------------------------\n"
                "Ran 2 tests in 0.004s\n\nFAILED (failures=1)", 1))
    s.call("patch", {"path": script, "old_string": "return 0", "new_string": "return 2 if low else 0"},
           patched(script, "return 0", "return 2 if low else 0"),
           "main() always returned 0; the exit code has to come from the low-space check.")
    s.call("terminal", {"command": f"cd {WD} && python3 -m unittest -q test_disk_report"},
           term("..\n----------------------------------------------------------------------\nRan 2 tests in 0.003s\n\nOK"))
    s.call("search_files", {"pattern": "min_free", "path": WD},
           {"total_count": 3, "matches": [{"path": script, "line": 22}, {"path": script, "line": 41},
                                          {"path": test, "line": 18}]})
    s.call("terminal", {"command": f"cd {WD} && python3 disk_report.py --json --min-free 100; echo rc=$?"},
           term('[{"path": "/", "free_gib": 86.88}, {"path": "/srv/data", "free_gib": 700.0}, '
                '{"path": "/mnt/backup", "missing": true}]\nrc=2'))
    s.say("Done: --json and --min-free work, exit code 2 below the limit, both tests pass, README untouched.")
    return s


def subagent_session() -> Session:
    # delegated child run: Hermes marks it as having no user-authored turns
    s = Session()
    root = "/home/user/projects/webapp"
    s.call("search_files", {"pattern": "TODO", "path": root},
           {"total_count": 2, "matches": [{"path": f"{root}/app/views.py", "line": 88},
                                          {"path": f"{root}/app/models.py", "line": 12}]})
    s.call("read_file", {"path": f"{root}/app/views.py"},
           "[read_file] read /home/user/projects/webapp/app/views.py from line 1 (4,210 chars)")
    s.call("terminal", {"command": f"cd {root} && python3 -m pytest -q tests/test_views.py"},
           term("....F\nE   KeyError: 'user_id'\n1 failed, 4 passed in 0.41s", 1),
           "One view test fails on a missing session key.")
    s.call("read_file", {"path": f"{root}/tests/test_views.py"},
           {"content": "def test_profile(client):\n    ...", "total_lines": 60, "file_size": 1650, "truncated": False,
            "is_binary": False, "is_image": False, "not_found": False})
    s.say("Report: 2 TODOs (views.py:88, models.py:12). test_profile fails with KeyError 'user_id' because the "
          "test client has no logged-in session; no files changed.")
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sess in (("session_fresh", disk_session()), ("session_no_user", subagent_session())):
        prompt = build_prompt(sess.turns)
        (OUT / f"{name}.txt").write_text(prompt)
        print(f"{name}: {len(sess.turns)} turns, {len(prompt)} chars")


if __name__ == "__main__":
    main()
