# Changelog

## Unreleased

- Repository housekeeping: CI tests on Python 3.10-3.13, issue templates (bug report, Hermes update
  compatibility), PR template, SECURITY.md, Dependabot for the CI actions.

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
