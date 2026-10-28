# Changelog

## Unreleased

- README quick start: `api_key` for hosted APIs (Hermes otherwise sends a placeholder key), and
  how to run without systemd (`install.sh` needs `systemctl`).

## 0.4.0 - 2026-10-28

Summaries stay within Hermes' size target, the reasoning answer is capped, and the quiz scores
facts and negative controls apart.

### Hybrid

- The answer is streamed and capped (`COMPRESS_FAST_HYBRID_MAX_TOKENS`, default 1100, was 2200).
  At the cap or the timeout the finished sections are kept, and the finished lines of the section
  being written, marked as cut. Before, a timeout threw the whole answer away, and a runaway
  answer (2200 tokens, 46 s) ended in the middle of a section.
- The prompt asks for Constraints & Preferences last, so a cut hits the section the rules already
  carry verbatim.
- `COMPRESS_FAST_HYBRID_EXTRA_BODY`: extra JSON fields for the call, e.g.
  `{"top_k": 20}`; left out again when the server rejects them.
- Critical Context no longer says "no model call" in hybrid summaries.

### Size

- Summaries are fitted to Hermes' token target, in hybrid mode too (the model's sections used to
  come on top). Section caps are shares of the target. Over the target they shrink tier by tier:
  log, verbatim last message and answers first, then notes, file list and Critical Context, then
  actions and errors, last the constraints, oldest entries first. The goal and the model's own
  text are never cut. Headings count towards the size.
- `CHARS_PER_TOKEN` 3.6 -> 2.8. Measured with the model's tokenizer, large summaries have 2.7-3.3
  chars per token; 0.3.0 summaries reached 10.6k (rules) and 12.9k tokens (hybrid) for a 7k target.
- Every action stays answerable: the newest keep their output excerpt, older ones get one short
  line (command and ok/FAILED), only the oldest fold per target. Completed Actions may grow from
  40 % to 60 % of the budget for that. In a window with 177 calls the fitted summary had dropped
  its oldest actions.
- Up to three long directories that many calls name are defined once next to `$WD` (`$D1` ..). In
  that window: 32.5k -> 30.3k chars for the full action list, 27.1k -> 21.8k for the short form.
- Less repetition: the session log lists the calls between two messages as one line with action
  numbers and outcome counts, and keeps only the start of messages quoted in full elsewhere.
  Active State names the recent actions by number and repeats only the last one.

### Content

- Errors & Fixes lists every exception name of the window ("none" when there was none).
  Constraints says when every user turn of the window is quoted in full, only when that still
  holds after the cuts.
- File changes keep one line each in Completed Actions (no "xN" for patches to the same file).
- Relevant Files: "read/used, no write seen" for files without a write call or shell write;
  read-only files are cut before modified ones.

### Fixes

- Exception names after a literal `\n` in escaped tool output (JSON in JSON, e.g. MCP results) are
  found; error excerpts of such results improve too.
- The legacy Hermes truncation marker "summary input truncated" is noted like "records ... elided".
- `Call.target` is cached; it was recomputed ~31k times per build. Rule summaries take up to 47 ms
  on the largest windows although fitting renders a summary up to 9 times (0.3.0: 22 ms).

### Quiz and measurements

- `tools/quiz.py` counts facts and negative controls apart. A fact counts by exact answer; a
  control fails only on a false "yes" ("no" would reward a summary that only sounds complete).
  Call questions whose shown command prefix fits calls that ended differently are void: no longer
  generated, and not counted in stored quizzes for any variant. `--table` prints the totals.
- Measured again on the 8 pairs (71 fact questions, 24 controls): rules and hybrid 71/71 facts,
  none wrong, no false claims; model alone 59/71, VCC 63/71. Large windows average 6.8k (rules) and
  6.9k tokens (hybrid) for a 7k target, single windows up to 5 % and 14 % over (0.3.0: up to 52 %
  and 84 %). Hybrid 10-39 s; the largest window 47.8 s -> 38.7 s.

### Housekeeping

- CI tests on Python 3.10-3.13, issue templates (bug report, Hermes update compatibility), PR
  template, SECURITY.md, Dependabot for the CI actions, CODEOWNERS.
- README rewritten: shorter, results first; the results chart ranks the variants, best on top.

## 0.3.0 - 2026-10-28

First public release.

- Works with any OpenAI-compatible model server: no model name or port of one setup built in,
  HTTPS upstreams, `Authorization` passed on to the hybrid reasoning call, and a retry without
  `chat_template_kwargs` for servers that reject unknown fields.
- `hybrid` is now the default mode; `COMPRESS_FAST_MODEL` names the model when a request has none.
- `install.sh`: user service with a settings file (`~/.config/compress-fast/env`), prints the
  Hermes config block.
- Tests run on synthetic fixtures built with Hermes' own prompt builder (`tools/make_fixtures.py`);
  local prompts in `testdata/` are tested too when present. 15 tests, Python 3.10-3.13.
- `quiz.py --vcc-only` adds hermes-vcc-engine 0.4.3 (projector via `vcc_bridge.py`) to an
  existing quiz. Same 8 pairs: VCC 79/96 (2 confidently wrong, both tool calls whose result said
  "Error executing code" read as ok), rules 84/96, hybrid 85/96. VCC projection ~5 ms, no
  reasoning sections.
- README for other users with result chart (`docs/make_images.py` renders it and the social
  preview), MIT license.

## 0.2.0 - 2026-10-28

Measured on 8 shadow pairs (4 small test sessions, 4 large real windows of 100-166k chars),
recall quiz with 96 questions incl. negative controls: no summary 0/96, model alone 77/96
(2 confidently wrong), rules 84/96 (1), hybrid 85/96 (1). Time per compaction: model alone
42-229 s, rules 1-12 ms, hybrid 8-46 s.

- Hybrid mode: rule facts plus four reasoning sections (constraints, state, decisions,
  errors & fixes) written by the model from a digest of 1-28k chars instead of the whole window;
  thinking off; hard timeout 90 s, then rules alone.
- Manual `/compress` is handled: Hermes always adds a focus block, now kept as a hint and as the
  goal when the user turn sits in the protected head.
- Older actions are folded per target with ok/failed counts instead of being dropped.
- `$WD` names the working directory once; Relevant Files keep full paths.
- Tool outputs keep their end (totals, results); patches show +/- lines and the changed
  definitions; `/dev/*` and `cd` targets are no longer listed as files.
- User instructions kept up to 1500 chars (rules at the end of long messages survive).
- Tools: `compare_shadow.py`, `quiz.py` (with no-summary baseline and `--hybrid`),
  `vcc_bridge.py`.

## 0.1.0 - 2026-10-27

- OpenAI-compatible server: compaction prompts answered by rules, everything else forwarded to
  the model server (JSON and streaming).
- Summarizer for Hermes 0.22.5 prompt text: fresh and update mode, no-user sessions, memory
  provider context kept verbatim, Hermes result stubs (`[terminal] ran ... -> exit N`, demoted
  output), MCP `tool_call` wrappers, escaped tracebacks.
- Shadow mode that stores prompt, rule summary and model summary for comparison.
- Diagnostics: log file, counters and timers, `/compress-fast/stats`.
