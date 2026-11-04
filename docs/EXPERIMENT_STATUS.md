# Experiment Status

_Last updated: 2026-11-04_

## Current phase

Stage 5 — initial 15-network experiment: offline part complete; **blocked
before interpretation (no API credentials)**.

## Completed

- Stage 1 planning docs (expansion criteria pre-registered, METHODOLOGY §8;
  one pre-interpretation amendment: float64 permutation check).
- Stage 2 organisation: `AGENTS.md`, configs, `.venv` (Python 3.12).
- Stage 3 implementation: all modules in `src/weightspace/`, six scripts.
- Stage 4 verification: 56 pytest tests pass (incl. leakage, permutation,
  end-to-end pipeline with a scripted LLM, expansion logic). Smoke stage
  (`smoke/`, 3 networks, does not count) ran through training/export/checks.
- Stage 5 offline: 15/15 networks trained and validated, 15 permuted copies
  (verified, max |Δlogit| 1.8e-14 in float64), 5 random-init controls,
  35 anonymized exports, leak tests passed, bit-for-bit retraining check
  passed.

## Implementation components

utils, config, tasks, models, training, checkpointing, weight_inspector,
weight_exporter, anonymization, permutation, expression, tools, tool_server,
prompts, llm_client, interpreter, statistics, evaluation, baselines,
experiment, status, reporting. Scripts: train_models, inspect_weights,
interpret_weights, run_experiment, expand_experiment, generate_report.

## Training seeds completed / checkpoints validated

initial_15: circle/diagonal/square × seed indices 0–4, all `ok` (test acc
0.985–0.999). Registry: `artifacts/manifests/registry.jsonl`.

## Interpretation protocols completed

None. The live API path (`llm_client.py`) has not been exercised against the
real API yet; it is covered only by the scripted-LLM tests.

## Evaluation results available

None for interpretation. Training results in `docs/RESULTS.md` §6.

## Pending requirements

- Protocol A (38 sessions) and Protocol B (40 sessions) for `initial_15`.
- Freeze, evaluate, report, expansion decision (criteria §8).
- Expansion to 30 / 60 only if criteria pass.

## Expansion eligibility

Not evaluated (needs Protocol B results).

## Next reproducible step

1. Put `ANTHROPIC_API_KEY=...` in `.env` (git-ignored) or export it.
2. Run `.venv/bin/python scripts/run_experiment.py --config configs/default.yaml`.
   Training/units are verified and skipped; interpretation resumes per
   session. Estimated cost $20–60; hard cap $80 per stage
   (`configs/interpretation.yaml`).
   Optionally run the smoke config first to exercise the live API on 7 units.

## Known failures / blockers

- No Anthropic credentials in the environment (`ANTHROPIC_API_KEY` unset).
- macOS `sandbox-exec` cannot nest inside the coding agent's sandbox; the
  tool server falls back to process isolation and records
  `os_sandbox_active` per session. Running the pipeline from a normal
  terminal should enable the extra layer.
- MPS not visible inside the agent sandbox; CPU is the configured default.
- Lint: 6 cosmetic ruff findings (unused loop variable names, two long lines);
  mypy reports loose-typing warnings in dict-heavy code. No test failures.

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
