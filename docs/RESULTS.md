# Results

Verified findings only. Sections marked **pending** have no data yet. No
interpretation result has been produced or estimated.

## 1. Research question

Can a reasoning-capable LLM infer the function a small neural network learned
from its architecture and trained parameters alone? See
`docs/EXPERIMENT_PLAN.md` §1–2.

## 2. Experimental design

3 tasks × 5 seeds = 15 trained networks, plus one neuron-permuted copy each
(15) and 5 untrained random-init controls: 35 units under one shuffled
opaque-ID pool. Protocol A (open-ended) and Protocol B (3-way multiple choice)
run as separate, isolated sessions; architecture-only controls run 3× (A) and
5× (B). Details: `docs/METHODOLOGY.md`.

## 3. Hardware and software

Target and execution machine: MacBook Pro, Apple M2 Pro, 16 GB. Recorded in
`artifacts/manifests/initial_15_manifest.json`: macOS 15.6.1 arm64, Python
3.12.14, torch 2.15.0, numpy 2.5.4, anthropic SDK 2.5.0. Training ran on
CPU (MPS was not visible inside the sandboxed shell; CPU is the configured
default anyway). Peak process memory during training: 332 MB.

## 4. Architecture

`Linear(2,16) → tanh → Linear(16,16) → tanh → Linear(16,1)`, 337 parameters,
identical for all units.

## 5. Datasets

Uniform on [-1,1]²; per network 10,000 train / 2,000 val / 5,000 test points,
each from its own derived seed. Mean observed positive rate in training sets:
circle 0.287 (expected 0.283), diagonal 0.503 (0.5), square 0.360 (0.36).

## 6. Training results (stage `initial_15`)

All 15 networks passed the ≥ 95% held-out test-accuracy criterion on the first
seed; no failed runs, no replacement seeds used.

| Task | Test accuracy, seeds 0–4 | Epochs | Total train time |
|---|---|---|---|
| circle | 0.9942, 0.9962, 0.9890, 0.9878, 0.9910 | 75, 120, 64, 39, 60 | 3.7 s |
| diagonal | 0.9960, 0.9958, 0.9962, 0.9968, 0.9988 | 14, 14, 12, 13, 15 | 0.7 s |
| square | 0.9912, 0.9848, 0.9910, 0.9920, 0.9920 | 213, 103, 109, 300, 160 | 9.0 s |

(square seed 3 reached the 300-epoch cap without meeting the early-stop rule,
but passed the test criterion.) Figures: `results/figures/initial_15/`
(`training_loss.png`, `test_accuracy.png`, `weight_heatmaps.png`,
`singular_values.png`).

Reproducibility check: retraining all 15 networks from the recorded seeds
reproduced every checkpoint bit-for-bit (`initial_15_checks.json`).

## 7. Weight extraction

35 exports in `artifacts/public/weight_exports/` (≈ 4.4k characters each,
allowlisted keys only, 6 significant digits, `parameters_sha256` included).
All exports pass the leak scan. Permuted copies verified equivalent in float64:
max |Δlogit| = 1.8e-14 over 4,096 points, identical classes (see
METHODOLOGY amendment of 2026-11-04).

## 8. Open-ended interpretation (Protocol A) — pending

## 9. Multiple-choice classification (Protocol B) — pending

## 10. Neuron-permutation robustness — pending

## 11. Baseline comparisons — pending

## 12. Statistical uncertainty — pending

## 13. Representative successes and failures — pending

## 14. Limitations (known in advance)

- n = 15 independent networks is exploratory; permuted copies are paired, not
  independent.
- The model may compute forward passes mentally; this cannot be blocked
  technically (prompt prohibition + post-hoc scan only).
- The output bias partly reflects class priors (28% / 50% / 36%), a
  legitimate but shallow cue.
- The macOS `sandbox-exec` layer could not nest inside the agent's own
  sandbox during development; the tool server then relies on process
  isolation (separate interpreter, empty env, temp CWD, no torch, no paths).
  Each session records whether the OS sandbox was active.

## 15. Expansion decision — pending (requires Protocol B results)

## 16. Future directions — pending
