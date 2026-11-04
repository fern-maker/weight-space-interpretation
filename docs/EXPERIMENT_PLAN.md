# Experiment Plan

## 1. Research question

Can a reasoning-capable LLM infer the function a small neural network learned
using only its architecture and trained parameters — without data, outputs,
labels or task description?

Reference: Experiment 1 of "Sparks of Weight-Space Programming"
(rikiyat.github.io/blog/weight-space-programming). There, an LLM read the
weights of four PPO policies (500→32→32→5 MLP) and named the game for each;
n = 4, no tools, no repeated trials, no permutation controls. This project is
a smaller, more controlled analog: 2-D inputs, three geometric tasks, many
seeds, two separate protocols, permutation robustness and negative controls.

## 2. Hypotheses

- **H1 (identification).** In Protocol B (3-way multiple choice), accuracy on
  trained networks exceeds chance (1/3).
- **H2 (discovery).** In Protocol A (open-ended), the model's proposed rule
  agrees with the true task better than with the other two tasks, more often
  than expected by chance.
- **H3 (permutation invariance).** Predictions for a network and its
  neuron-permuted copy agree, and accuracy on permuted copies does not drop.
- **H4 (no confabulation).** On random-init networks the model reports lower
  confidence and does not systematically assign a learned function; with
  architecture only, its choices are not accurate beyond chance (there is
  nothing to be accurate about: architecture is identical across tasks).

The initial 15-network study is exploratory. H1–H4 are not tested with
corrected significance thresholds at this stage; see `METHODOLOGY.md` §9.

## 3. Conditions and units (initial stage `initial_15`)

| Condition | Units | Protocol A sessions | Protocol B sessions |
|---|---|---:|---:|
| Trained, original | 3 tasks × 5 seeds = 15 | 15 | 15 |
| Trained, neuron-permuted (k = 1 per network) | 15 | 15 | 15 |
| Random-init (untrained) control | 5 | 5 | 5 |
| Architecture-only control (no weights) | — | 3 repeats | 5 repeats |
| **Total** | 35 weight units | **38** | **40** |

Original, permuted and random units share one shuffled ID pool
(`network_001…network_035`) so neither ID nor order reveals kind or task.
Every session is a fresh conversation. Protocols A and B for a unit are run
in separate passes (all A sessions, then all B sessions), each in an
independently shuffled order.

Primary condition = **strict weight-only, full weights + analytic tools**.
Optional representation controls (raw dump without tools, statistics only)
are deferred; if run, they are separate conditions with their own records
and are labelled exploratory.

## 4. Stages

| Stage | Networks | Seeds per task | Status rule |
|---|---:|---|---|
| `smoke` | 3 (1 per task) | index 900 (reserved) | dev only, never counts, separate directories |
| `initial_15` | 15 | 0–4 | mandatory |
| `expand_30` | 30 | 0–9 (adds 5–9) | only if criteria in METHODOLOGY §8 pass |
| `expand_60` | 60 | 0–19 (adds 10–19) | only if 30-network criteria pass |

Expansion stages add independent seeds, new permutations for the new
networks, and new random-init controls (5 per stage, new seeds). Existing
predictions are reused unchanged when model ID, prompt version, tool set and
inference parameters are identical; otherwise expansion is refused and a new
full stage must be run.

## 5. Pipeline per stage

1. Validate config; record hardware/software environment.
2. Train networks sequentially (CPU), record every run in the registry.
3. Verify checkpoints (reload, recompute accuracy, SHA-256).
4. Build permuted variants; verify equivalence privately (max |Δlogit| ≤ 1e-5
   on 4,096 private points).
5. Build random-init controls.
6. Export + anonymize; leak-scan every export.
7. Protocol A pass. 8. Protocol B pass. 9. Architecture-only sessions.
10. Freeze predictions (SHA-256 manifest).
11. Private evaluation. 12. Figures and `docs/RESULTS.md`.
13. Update `docs/EXPERIMENT_STATUS.md`. 14. Expansion verdict.

## 6. LLM configuration

- Provider: Anthropic Claude API via the official `anthropic` Python SDK.
- Model: `claude-opus-5-5` (config `interpretation.model`); the served model
  from every response is recorded.
- Adaptive thinking (always on for this model), `display: summarized`,
  `effort: high`, streaming, `max_tokens` 32,000 per turn.
- Max 16 assistant turns per session; session must end with
  `submit_interpretation`. No sampling parameters (not accepted by the model).
- Refusal fallback to another model: disabled (would confound the model under
  test). A refusal = invalid response.
- Credentials: `ANTHROPIC_API_KEY` (or an SDK-supported credential source)
  from the environment; never stored in the repo.

## 7. Budget

Each weight unit export is ~4–5k tokens. With caching of the stable prefix
and ≤ 16 turns, a session is estimated at 40–150k input tokens and 5–20k
output tokens: ≈ $0.25–0.80 at $4/$20 per MTok. Initial stage: 78 sessions
≈ $20–60. Expansion to 30 adds ≈ 75 sessions; to 60 adds ≈ 150 more.
`interpretation.max_cost_usd` (default 80 per stage) aborts the pass when
exceeded; completed sessions are kept and the stage is resumable.

## 8. Compute

Training: 15 networks × ~200 epochs × 40 batches on CPU — expected under 5
minutes in total. Memory well under 1 GB. No GPU required; MPS optional.

## 9. Risks and failure modes

| Risk | Mitigation |
|---|---|
| Leakage via IDs/order/file names | Shuffled global pool, opaque IDs, allowlisted export keys, leak tests |
| Leakage via tools (forward pass) | No tool accepts input points; isolated tool process; tool calls logged |
| Model simulates forward pass mentally | Prompt prohibition; post-hoc transcript scan; reported as limitation |
| Class-prior cue in output bias (positive rate: circle ≈ 28%, square 36%, diagonal 50%) | Legitimate weight-derived evidence; reported as a possible shortcut; analysed by checking cited evidence |
| Position bias in MC candidates | Candidate order shuffled per session, recorded |
| Training failure (< 95%) | Recorded, never dropped; replacement seed drawn from reserved range and documented |
| Permutations not independent samples | Reported as paired robustness, never pooled into primary accuracy |
| API errors / rate limits | SDK retries + outer retry; resumable sessions |
| Cost overrun | Per-stage cap |
| Missing credentials | Pipeline stops before interpretation; nothing invented |
| Open-ended scoring subjectivity | Primary score is a symbolic rule checked numerically; free text only summarised |
| Changing criteria after seeing results | Criteria fixed in METHODOLOGY §8 before any interpretation run; frozen by hash in the stage manifest |
