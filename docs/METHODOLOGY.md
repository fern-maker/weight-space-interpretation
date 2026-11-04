# Methodology

Canonical experimental procedures. Sections 8 (expansion criteria) and 6–7
(scoring rules) are **pre-registered**: they were written before any
interpretation session was run and must not be changed after results are
observed. Any later change must be added as a dated amendment below the
original text, with the reason, and must not be applied retroactively to a
stage already evaluated.

## 1. Tasks

Input domain $(x_1,x_2)\in[-1,1]^2$. Label 1 = positive class.

| Task | Rule | Expected positive rate |
|---|---|---:|
| circle | $x_1^2+x_2^2\le 0.36$ | $\pi\cdot0.36/4\approx 28.3\%$ |
| diagonal | $x_2>x_1$ | 50% |
| square | $\lvert x_1\rvert\le0.6 \land \lvert x_2\rvert\le0.6$ | 36% |

Sampling: i.i.d. uniform on the square. Per network: train 10,000, validation
2,000, test 5,000 points, each split drawn from its own seed (§3). No
re-balancing (the class imbalance is part of the learned function and is
documented). Observed class balance is recorded for every split.

## 2. Network and training

- Architecture (identical for all units):
  `Linear(2,16) → Tanh → Linear(16,16) → Tanh → Linear(16,1)`; 337 params.
- Loss `BCEWithLogitsLoss`; Adam, lr = 1e-3; batch 256; up to 300 epochs;
  early stop when validation accuracy ≥ 0.99 for 10 consecutive epochs
  (stop rule uses validation only).
- Default PyTorch initialization under a fixed seed; float32; CPU;
  `torch.use_deterministic_algorithms(True)`; one thread.
- Validation criterion: **held-out test accuracy ≥ 95%**.
- Every run (success or failure) is appended to the registry with loss curve,
  accuracies, epochs, wall time and status. A failed network is never
  deleted. If a seed fails, the network is reported as failed, and a
  replacement seed index from the reserved range 500–599 (first unused, in
  order) is trained so the stage has 5 validated networks per task; the
  substitution is reported in RESULTS.

## 3. Seeds

`derive_seed(namespace, task, seed_index, purpose)` = first 8 bytes of
SHA-256 of the joined string, as an unsigned int mod 2³¹. Purposes:
`init`, `train_data`, `val_data`, `test_data`, `batch_order`, `permutation`.
Random-init controls use namespace `random_init` with their own indices and
a task-free name. Seed indices per stage: initial 0–4, expansion 5–9 then
10–19, smoke 900. The registry rejects any duplicate
`(task, seed_index, training_config_hash)`.

## 4. Weight export (LLM-facing)

Per unit, a JSON object with only these keys: `network_id`,
`format_version`, `architecture` (layer list with shapes), `activations`
(`tanh(z) = (e^z − e^{−z})/(e^z + e^{−z})`), `forward_equations`
(`h1 = tanh(W1 x + b1)`, `h2 = tanh(W2 h1 + b2)`, `z = W3 h2 + b3`),
`input_domain`, `output_rule` (`class 1 iff z > 0`, i.e. sigmoid(z) > 0.5),
`precision` (values rounded to 6 significant digits from float32),
`parameters` (`W1` 16×2, `b1` 16, `W2` 16×16, `b2` 16, `W3` 1×16, `b3` 1;
row i of `W_k` = incoming weights of unit i of layer k), and
`parameters_sha256`. Layer names are `layer1..layer3`; no task, seed, path,
date, metric or kind information.

## 5. Interpretation conditions

### 5.1 Strict weight-only (primary)

The model receives the export in the first user message and may call these
analytic tools (all computed from the export only):
`get_architecture`, `get_weight_matrix`, `get_bias_vector`, `get_neuron`,
`weight_statistics`, `singular_values`, `matrix_product` (weight matrices
only), `first_layer_hyperplanes`, and the terminal `submit_interpretation`.

Prohibited and technically unavailable: forward passes, evaluating chosen
inputs, boundary sampling, data, labels, metrics, parameter edits,
manifests, files, code execution. Tools run in an isolated subprocess holding
only the export. The prompt also asks the model not to simulate the full
network on chosen input points; this cannot be enforced, so transcripts are
scanned afterwards (§7.4).

### 5.2 Protocol A — open-ended

No candidate names or formulas. The model must submit:

```json
{
  "network_id": "network_007",
  "behavior_summary": "...",
  "boundary_geometry": "...",
  "symmetries": ["..."],
  "proposed_rule": "x1**2 + x2**2 <= 0.35",
  "rule_confidence": 0.0-1.0,
  "supporting_evidence": ["..."],
  "alternative_explanations": ["..."],
  "uncertainties": ["..."]
}
```

`proposed_rule` is a boolean expression over `x1`, `x2` describing the
**class-1 region**, using only numbers, `+ - * / **`, `abs`, `min`, `max`,
`sqrt`, comparisons, `and`, `or`, `not`. The format is a syntax, not a
candidate list; it does not hint at any task.

### 5.3 Protocol B — multiple choice

Fresh session. Same export and tools plus three candidates, each with a label
and formula: `circle` ($x_1^2+x_2^2\le0.36$), `diagonal` ($x_2>x_1$),
`square` ($|x_1|\le0.6\land|x_2|\le0.6$). Candidate order is shuffled per
session (seeded, recorded). The model must pick exactly one and submit
`network_id, predicted_function, confidence ∈ [0,1], mathematical_explanation,
supporting_evidence, uncertainties`. It is told that the candidate list does
not imply each candidate appears equally often.

### 5.4 Isolation

Every session = new conversation, no shared history, no feedback. All A
sessions run in one pass, all B sessions in a separate pass, each in an
independently shuffled order. The exact prompt, prompt version, tool schema
hash, model ID requested and served, inference parameters, every tool call
(name, arguments, result hash), token usage, latency, cost and request IDs
are stored with the session record before any evaluation.

### 5.5 Controls

- **Chance:** 1/3 for Protocol B.
- **Architecture-only:** prompt with architecture, activations, domain and
  output rule but no parameters and no tools except submit. Protocol B ×5
  repeats, Protocol A ×3 repeats. Because every network has this
  architecture, any consistent choice is uninformative; we report the choice
  distribution and confidence.
- **Random-init:** 5 untrained networks per stage (PyTorch default init,
  own seeds), mixed blind into the pool, run under both protocols. There is
  no correct task; we report choice distribution, confidence (compared with
  trained units), whether the model flags the network as untrained or
  structureless, and Protocol A fidelity to the network's actual behaviour.
- **Seeds:** per-seed accuracy across the 5 (later 10/20) seeds.

## 6. Permutation robustness

For each trained network, k = 1 (initial stage) independent permutation pair
$(P_1,P_2)$ of the two 16-unit hidden layers, both non-identity:
$W_1' = P_1W_1$, $b_1'=P_1b_1$, $W_2'=P_2W_2P_1^\top$, $b_2'=P_2b_2$,
$W_3'=W_3P_2^\top$, $b_3'=b_3$. Equivalence is verified privately on 4,096
seeded points: max |logit difference| ≤ 1e-5, and identical predicted
classes. Permuted units get their own opaque IDs and are interpreted blind
under both protocols. The permutation indices are stored privately.

Reported per protocol: accuracy on permuted units; pairwise agreement of
original vs permuted prediction (B: same label; A: same implied family, §7.2,
and |Δ agreement score|); confidence change; qualitative explanation
differences for disagreeing pairs. A pair is **stable** if both predictions
match, **unstable** otherwise. Permuted results are never pooled into the
primary accuracy, because they are not independent networks.

## 7. Scoring (pre-registered)

### 7.1 Protocol B

- Correct iff `predicted_function` equals the true task.
- Invalid response (counted as incorrect in accuracy, also reported
  separately): no submission within the turn cap, schema violation,
  label outside the candidate set, refusal, or API failure after retries.
- Primary metric: accuracy on the 15 original trained networks.
- Also: confusion matrix, per-task and per-seed accuracy, accuracy on
  permuted units, Brier score and reliability bins, invalid rate.

### 7.2 Protocol A

On a private 201×201 grid over the domain, the parsed rule $R$ is compared
with each task $f_t$ and with the network's own thresholded output $g$:
agreement $= $ fraction of grid points where they match; IoU of positive
regions is also reported.

- **Implied family** = argmax over tasks of agreement($R$, $f_t$).
- **Correct** iff the implied family is the true task **and**
  agreement($R$, $f_{true}$) ≥ 0.90.
- **Fidelity** = agreement($R$, $g$) — how well the description matches what
  the network computes (meaningful for random-init units too).
- Invalid: missing or unparsable rule, schema violation, refusal, failure.
- Free-text fields are not scored for credit. They are summarised
  descriptively (e.g. whether the geometry text mentions curvature,
  linearity, axis-aligned edges), with the limitation that this is a
  qualitative reading.
- Protocol A and B scores are always reported separately and never combined.

### 7.3 Chance reference for Protocol A

With no information, a rule matching one family at ≥ 0.90 agreement by luck
is unlikely; the reference is the architecture-only Protocol A sessions and
the random-init units, reported alongside.

### 7.4 Post-hoc transcript checks

Visible reasoning summaries and final explanations are scanned for patterns
suggesting point-wise forward simulation (e.g. explicit evaluation of the
network at a listed coordinate through all layers). Counts are reported;
sessions are not excluded on this basis.

## 8. Expansion criteria (pre-registered decision rule)

These are operational decision rules, not significance claims.

**Expand from 15 to 30 networks only if all of the following hold:**

1. All 15 initial networks are validated (test accuracy ≥ 95%, including
   any documented replacements).
2. All leakage tests pass and the reproducibility check passes (retraining
   from manifests reproduces checkpoints bit-for-bit on CPU).
3. Protocol B accuracy on the 15 original networks ≥ 9/15.
   (Under pure guessing P(X ≥ 9) ≈ 0.031.)
4. Not driven by one task: after removing the task with the highest
   per-task accuracy, accuracy on the remaining 10 networks ≥ 4/10 (> 1/3).
5. No collapse under permutation: Protocol B accuracy on the 15 permuted
   units ≥ 8/15 **and** original/permuted label agreement ≥ 10/15 pairs.

**Expand from 30 to 60 only if:** criteria 1–2 hold for all 30; Protocol B
accuracy on the 30 originals ≥ 18/30 (60%; P ≈ 0.0025 under chance) with
the 95% Wilson lower bound > 1/3; criterion 4 holds on 30 (remaining 20 ≥
8/20); and on the 30 permuted units accuracy ≥ 16/30 and agreement ≥ 20/30.

If criteria fail: stop expanding, record which criteria failed, analyse
failure modes, and do not change the criteria or re-run the stage with
altered settings under the same stage name. Protocol A results are reported
but are not an expansion criterion (they are the harder, secondary test).

## 9. Statistics

- Raw counts with denominators, percentages, 95% Wilson intervals.
- Exact one-sided binomial tail vs 1/3 for Protocol B, reported as
  descriptive; no multiple-comparison claims at the exploratory stage.
- Calibration: Brier score; reliability table in 5 bins (small n caveat).
- Agreement: raw pair agreement and Cohen's κ.
- Seeds are the independent unit; permutations and repeats are not.
- Results across stages are shown cumulatively (15 → 30 → 60) with intervals.

## 10. Reproducibility

Each stage manifest records config hash, source-tree hash, environment, seed
derivations, export hashes, checkpoint hashes, prompt version, tool-schema
hash, and model parameters. Prediction files are frozen by SHA-256 before
evaluation; evaluation refuses changed files. API outputs are not
deterministic, so "reproduction" of the interpretation step means re-running
under the same recorded configuration, not identical text.

## Amendments

**2026-11-04 (before any interpretation session was run).** §6 equivalence
check: the forward passes used for the private permutation check are computed
in float64 instead of float32. In float32, reordering hidden units changes the
summation order and produced a max logit difference of 1.14e-5 (> 1e-5) for
one square network with large logits. The tolerance (1e-5) and the
same-class requirement are unchanged; float64 makes the check stricter. No
scoring rule or expansion criterion is affected.
