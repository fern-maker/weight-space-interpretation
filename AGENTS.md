# AGENTS.md — standing instructions (read first, every session)

## Session start (mandatory)

1. Read this file.
2. Read `docs/EXPERIMENT_STATUS.md` (current phase, next step).
3. Read `docs/ARCHITECTURE.md` and `docs/METHODOLOGY.md`.
4. Verify the artifacts the status file claims exist (manifests, hashes)
   before continuing. Never trust a previous conversation's claim that a
   stage finished.
5. Resume from the next incomplete step.

## Non-negotiable requirements

- Planning and architecture precede implementation. Architecture changes go
  into `docs/ARCHITECTURE.md` first.
- Target hardware: MacBook Pro, Apple M2 Pro, **16 GB unified memory**,
  macOS. CPU by default, MPS optional. Sequential training. No local LLM.
- Initial experiment: **exactly 3 tasks (circle, diagonal, square) × 5
  independent seeds = 15 trained networks**, all trained and evaluated.
  Smoke tests (stage `smoke`) never count toward this.
- Expansion: if the pre-registered criteria in `docs/METHODOLOGY.md` §8 pass,
  expand to 30, then consider 60 networks, using new independent seeds only.
  Never change criteria after seeing results. Never expand to chase a
  favourable result.
- Open-ended (Protocol A) and multiple-choice (Protocol B) interpretation
  are separate conditions, separate sessions, separate scores. Never merge.
- In strict weight-only conditions the LLM cannot access task identities,
  data, labels, metrics, checkpoints, manifests, or forward-pass results.
  Enforced by the isolated tool server and allowlist, not by prompt alone.
- Neuron-permutation robustness is mandatory for every trained network.
- Failed training runs are recorded, never silently dropped.
- Conclusions only from recorded evidence (`results/`, manifests). Never
  invent interpretation results; without API credentials, stop before the
  interpretation step and say so.
- Keep `docs/EXPERIMENT_STATUS.md` updated after each major phase.

## Practical notes

- Environment: `uv` venv at `.venv` (Python 3.12). Run with
  `.venv/bin/python`. Tests: `.venv/bin/pytest -q`.
- Private ground truth: `artifacts/private/` — never passed to prompts or the
  tool server.
- LLM: Anthropic API, model from `configs/interpretation.yaml`; key from
  `ANTHROPIC_API_KEY` env var (see `.env.example`). Never commit secrets.
