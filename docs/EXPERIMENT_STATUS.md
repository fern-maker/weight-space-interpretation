# Experiment Status

_Last updated: 2026-11-05_

## Current phase

Stage 5 — initial 15-network experiment: offline part complete. Interpretation
backend switched to **headless Claude Code** (no API key available; METHODOLOGY
amendment 2026-11-05, ARCHITECTURE §10), implemented and tested offline.
**Blocked on the user's live isolation probe** (the agent sandbox cannot use
the user's Claude Code login).

## Completed

- Stage 1 planning docs (expansion criteria pre-registered, METHODOLOGY §8;
  pre-interpretation amendments: 2026-11-04 float64 permutation check,
  2026-11-05 Claude Code backend).
- Stage 2 organisation: `AGENTS.md`, configs, `.venv` (Python 3.12).
- Stage 3 implementation: all modules in `src/weightspace/`, seven scripts.
- Stage 4 verification: 94 pytest tests pass (incl. leakage, permutation,
  end-to-end pipeline with a scripted LLM, expansion logic, and the Claude
  Code backend against a fake `claude` CLI). Smoke stage (`smoke/`, 3
  networks, does not count) ran through training/export/checks.
- Stage 5 offline: 15/15 networks trained and validated, 15 permuted copies
  (verified, max |Δlogit| 1.8e-14 in float64), 5 random-init controls,
  35 anonymized exports, leak tests passed, bit-for-bit retraining check
  passed.
- Claude Code backend: `mcp_server.py`, `claude_code_backend.py`,
  `scripts/probe_claude_code.py`, `tests/test_claude_code_backend.py`.

## Implementation components

utils, config, tasks, models, training, checkpointing, weight_inspector,
weight_exporter, anonymization, permutation, expression, tools, tool_server,
mcp_server, prompts, llm_client, interpreter, claude_code_backend, statistics,
evaluation, baselines, experiment, status, reporting. Scripts: train_models,
inspect_weights, interpret_weights, run_experiment, expand_experiment,
generate_report, probe_claude_code.

## Training seeds completed / checkpoints validated

initial_15: circle/diagonal/square × seed indices 0–4, all `ok` (test acc
0.985–0.999). Registry: `artifacts/manifests/registry.jsonl`.

## Interpretation protocols completed

None. Neither backend has been exercised against a real model yet. One
pre-design CLI probe (2026-11-05, no model output) confirmed that the
lockdown flags give a session with no tools, no MCP servers, no skills, no
slash commands and only built-in plugins.

## Evaluation results available

None for interpretation. Training results in `docs/RESULTS.md` §6.

## Pending requirements

- Live isolation probe passes (user, normal terminal).
- Smoke stage interpretation via Claude Code (dev only, never counts).
- Protocol A (38 sessions) and Protocol B (40 sessions) for `initial_15`.
- Freeze, evaluate, report, expansion decision (criteria §8).
- Expansion to 30 / 60 only if criteria pass (same backend).

## Expansion eligibility

Not evaluated (needs Protocol B results).

## Next reproducible step

From a normal terminal in the repo, logged in to Claude Code:
1. `.venv/bin/python scripts/probe_claude_code.py` → must print `RESULT: PASS`.
2. `.venv/bin/python scripts/run_experiment.py --config configs/smoke.yaml`
   (live check on 7 throwaway units; does not count).
3. `.venv/bin/python scripts/run_experiment.py --config configs/default.yaml`.
   Training/units are verified and skipped; interpretation resumes per
   session; usage-limit stops are resumable by re-running.

## Known failures / blockers

- The agent sandbox cannot read the user's Claude Code login (keychain), so
  live sessions must be started by the user.
- `--input-format stream-json` message shape and `--allowedTools mcp__weights`
  permission matching are implemented from the CLI help and not yet seen
  working live; the probe checks both and fails safely.
- macOS `sandbox-exec` cannot nest inside the coding agent's sandbox; from a
  normal terminal it should wrap the MCP server. Recorded per session
  (`os_sandbox_active`).
- MPS not visible inside the agent sandbox; CPU is the configured default.
- Lint: pre-existing cosmetic ruff findings in older modules; new modules
  clean (ruff, mypy).

## Auto-generated stage records


<!-- AUTO smoke START -->
### Stage `smoke` (auto-generated 2026-11-04T21:37:16+00:00)

- Root: `smoke`; counts toward study: False
- Steps done: checks, train, units
- Validated networks: 3; units: 7
- Failed training runs: 0
- Leak tests passed: True; reproducible: True
- **Stopped:** no API credentials; interpretation not run (no results invented)
<!-- AUTO smoke END -->
<!-- AUTO initial_15 START -->
### Stage `initial_15` (auto-generated 2026-11-04T21:38:36+00:00)

- Root: `.`; counts toward study: True
- Steps done: checks, train, units
- Validated networks: 15; units: 35
- Failed training runs: 0
- Leak tests passed: True; reproducible: True
- **Stopped:** no API credentials; interpretation not run (no results invented)
<!-- AUTO initial_15 END -->
