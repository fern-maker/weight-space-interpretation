# Architecture

Canonical software design for the weight-space interpretation experiment.
Read together with `docs/METHODOLOGY.md` (procedures, criteria) and
`docs/EXPERIMENT_PLAN.md` (stages, budget, risks).

## 1. Design goals

1. **Separation of stages.** Training, inspection/export, interpretation,
   evaluation and reporting are separate modules with file-based hand-offs.
   Every stage can be run alone and is idempotent (re-running skips completed,
   hash-verified work).
2. **Leakage is a security boundary.** Anything the LLM can see is produced by
   one narrow path: `weight_exporter` → `anonymization` → public export JSON →
   `prompts` / `tool_server`. Ground truth lives only under `artifacts/private/`
   and is read only by `training`, `permutation`, `evaluation`, `reporting`.
3. **Reproducibility.** Every artifact has a manifest entry with config hash,
   seeds, library versions, source-tree hash and content SHA-256.
4. **Small-machine friendly.** Target: M2 Pro, 16 GB unified memory. Networks
   have 337 parameters; everything trains on CPU in seconds. Sequential
   execution; no multiprocessing pools; no local LLM.
5. **Extensible.** The weight-inspection layer and the tool server are
   generic over `nn.Sequential` MLPs so later weight-*editing* experiments can
   reuse them (an edit tool would be a new, separately-gated tool set).

## 2. Repository layout

The repository root is this directory (`Read+WriteNeuralWeights/`); the
proposed `weight-space-experiment/` wrapper directory is dropped because the
workspace itself is the project.

```text
.
├── AGENTS.md                    standing rules (read first every session)
├── README.md
├── pyproject.toml               deps + ruff/mypy/pytest config
├── .gitignore  .env.example
├── docs/
│   ├── ARCHITECTURE.md  EXPERIMENT_PLAN.md  METHODOLOGY.md
│   ├── EXPERIMENT_STATUS.md     live execution state
│   └── RESULTS.md               verified findings only
├── configs/
│   ├── default.yaml             top-level; includes the others
│   ├── training.yaml  interpretation.yaml  expansion.yaml
│   └── smoke.yaml               dev-only smoke test (never counts)
├── src/weightspace/
│   ├── utils.py                 seeding, hashing, json io, env/hardware info
│   ├── config.py                load/merge/validate YAML → dataclasses
│   ├── tasks.py                 3 task functions + dataset sampling
│   ├── models.py                MLP factory (single architecture)
│   ├── training.py              train/validate one network
│   ├── checkpointing.py         save/load/verify checkpoints + registry
│   ├── weight_inspector.py      pure analytic inspection functions
│   ├── weight_exporter.py       model → canonical JSON (no metadata)
│   ├── anonymization.py         opaque IDs, private id↔truth map, leak scan
│   ├── permutation.py           function-preserving neuron permutations
│   ├── expression.py            safe parser for open-ended rules (AST whitelist)
│   ├── tool_server.py           isolated subprocess serving allowlisted tools
│   ├── tools.py                 tool schemas + pure tool implementations
│   ├── prompts.py               prompt templates for each protocol/condition
│   ├── llm_client.py            Anthropic client wrapper, retries, cost, logs
│   ├── interpreter.py           one isolated session = one agent loop
│   ├── evaluation.py            private scoring of frozen predictions
│   ├── statistics.py            Wilson CI, binomial tests, calibration
│   ├── baselines.py             chance, architecture-only, random-init units
│   ├── experiment.py            stage orchestration + registry + freeze
│   └── reporting.py             figures + RESULTS.md tables
├── scripts/
│   ├── train_models.py  inspect_weights.py  interpret_weights.py
│   ├── run_experiment.py  expand_experiment.py  generate_report.py
├── tests/                       pytest suite (see §9)
├── artifacts/
│   ├── private/                 NEVER LLM-visible
│   │   ├── checkpoints/         *.pt state dicts
│   │   ├── ground_truth.json    network_id → {task, seed, kind, parent}
│   │   ├── training_runs.jsonl  every run, success or failure
│   │   └── permutations/        permutation index vectors
│   ├── public/
│   │   └── weight_exports/      network_XXX.json (LLM-facing)
│   └── manifests/               stage manifests, freeze manifests, registry
├── results/
│   ├── predictions/<stage>/<protocol>/<network_id>.json   frozen records
│   ├── transcripts/<stage>/...  full request/response logs (redacted)
│   ├── metrics/                 evaluation outputs (json)
│   ├── logs/                    run logs
│   └── figures/
└── notebooks/exploration.ipynb  post-hoc only, never used by the interpreter
```

Changes versus the proposed structure, and why:

| Change | Reason |
|---|---|
| `artifacts/private/` vs `artifacts/public/` split | Makes the leakage boundary a directory boundary that tests can check. |
| `config.py` | Validation logic should not live in scripts. |
| `tools.py` + `tool_server.py` | Tool *logic* is pure and unit-testable; the *server* is the isolated process the agent loop talks to. |
| `prompts.py` | Prompt text is versioned and leak-scanned separately from loop logic. |
| `expression.py` | Open-ended answers carry a symbolic rule that must be parsed without `eval`. |
| `statistics.py` | Keeps `evaluation.py` small; reused by reporting. |
| `configs/smoke.yaml` | Smoke runs write to a separate stage name and can never be mistaken for the 15-network study. |
| `results/transcripts/` | Full interaction logs kept apart from compact prediction records. |

## 3. Data flow

```text
configs/*.yaml ──► config.py (validate, hash)
                       │
                       ▼
 tasks.py ──datasets──► training.py ──► checkpointing.py ──► artifacts/private/checkpoints
   (in-memory, seeded)        │                 │
                              ▼                 ▼
               training_runs.jsonl     permutation.py (private forward-pass check)
                                                │
          models (orig + permuted + random-init)│
                                                ▼
                      weight_exporter.py ─► anonymization.py
                                                │   ├─► artifacts/private/ground_truth.json
                                                ▼   │
                         artifacts/public/weight_exports/network_XXX.json
                                                │
   ═════════════════════ LEAKAGE BOUNDARY ══════╪══════════════════════════
                                                ▼
           interpreter.py  ──prompt──►  llm_client.py ──HTTPS──► Claude API
                │   ▲                                   (model from config)
                │   └── tool calls ── tool_server.py (subprocess, export only)
                ▼
     results/predictions/... (one JSON per session)  ─► freeze manifest (SHA-256)
   ═════════════════════════════════════════════════════════════════════════
                                                ▼
     evaluation.py (reads ground_truth + checkpoints, verifies freeze hashes)
                                                ▼
                results/metrics/*.json ─► reporting.py ─► figures, docs/RESULTS.md
```

## 4. Module responsibilities and interfaces

Signatures are indicative; types use dataclasses in `config.py`.

**utils.py** — `set_global_seed(seed)`, `derive_seed(*parts) -> int`
(SHA-256 based, stable across runs), `sha256_file(path)`, `sha256_json(obj)`,
`canonical_json(obj) -> str` (sorted keys, fixed float format),
`environment_info() -> dict` (python, torch, numpy, platform, device
availability), `source_tree_hash() -> str`, `resource_snapshot() -> dict`
(RSS via `resource.getrusage`).

**config.py** — `load_config(path) -> ExperimentConfig`; merges `include:`
files, validates ranges (exactly 3 tasks, seeds per task ≥ 1, accuracy target
in (0,1]), returns `config_hash`.

**tasks.py** — `TASKS: dict[str, Callable[[ndarray], ndarray]]` with
`circle`, `diagonal`, `square`; `sample_dataset(task, n, seed) ->
(X: float32[n,2], y: float32[n])`; `class_balance(y) -> float`. Pure numpy.

**models.py** — `build_mlp() -> nn.Sequential` (exactly
`Linear(2,16) Tanh Linear(16,16) Tanh Linear(16,1)`); `ARCH_SPEC` constant;
`assert_architecture(model)`.

**training.py** — `train_network(task, seed_index, cfg) -> TrainingResult`
(loss curve, val/test accuracy, epochs, wall time, status). BCEWithLogits,
Adam lr 1e-3, mini-batches from a seeded generator. Never raises on low
accuracy; returns `status="failed_accuracy"`.

**checkpointing.py** — `save_checkpoint(model, path, meta)`,
`load_checkpoint(path) -> nn.Sequential`, `verify_checkpoint(path,
expected_sha)`, `Registry` (append-only JSONL of training runs; rejects a
duplicate `(task, seed_index, config_hash)`).

**weight_inspector.py** — the required API, all pure and analytic:
`inspect_architecture`, `inspect_weight_matrix`, `inspect_bias_vector`,
`inspect_neuron`, `compute_weight_statistics`, `compute_singular_values`,
`export_weights`. Works on a model or on an exported dict (so the tool server
needs no torch).

**weight_exporter.py** — `model_to_export(model, network_id) -> dict` and
`write_export(dict, path)`. Allowlisted keys only:
`network_id, format_version, architecture, activations, forward_equations,
input_domain, output_rule, precision, parameters{layer: {weight, bias, shape}}`.
Floats rounded to 6 significant digits (documented as `precision`). Includes
`parameters_sha256` of the rounded payload so exports are verifiable.

**anonymization.py** — `allocate_ids(units, rng) -> mapping`: shuffles all
units (originals, permuted, random-init) together, assigns
`network_001…`; writes `ground_truth.json` privately;
`scan_for_leaks(text) -> list[str]` detecting task names, private paths,
seeds, checkpoint names, and threshold constants.

**permutation.py** — `permute_hidden(model, perms: list[LongTensor]) ->
model`, `random_permutations(model, rng)`, `verify_equivalence(a, b, X,
atol)`. Optional `sign_flip` (tanh oddness) for later robustness work.

**expression.py** — `parse_rule(text) -> Rule` via `ast` whitelist (numbers,
`x1`, `x2`, `+ - * / **`, unary minus, `abs`, `min`, `max`, `sqrt`,
comparisons, `and/or/not`); `Rule.evaluate(X) -> bool[n]`. No `eval`, no
attribute access, no names beyond the whitelist, exponent bounded.

**tools.py** — JSON schemas (strict, `additionalProperties:false`) and pure
implementations over an export dict:
`get_architecture`, `get_weight_matrix(layer)`, `get_bias_vector(layer)`,
`get_neuron(layer, index)`, `weight_statistics(layer)`,
`singular_values(layer)`, `matrix_product(left_layer, right_layer)` (products
of weight matrices only), `first_layer_hyperplanes()` (each first-layer
unit's line `w·x + b = 0`: unit normal, signed offset, whether it crosses the
domain). **No tool accepts input points, file paths, or code.** Also
`submit_interpretation` (protocol-specific schema) which ends the session.

**tool_server.py** — run as `python -I -m weightspace.tool_server` with an
empty environment, an empty temp directory as CWD, and the single export JSON
on stdin. It imports only `json`, `numpy`, `weightspace.tools`,
`weightspace.weight_inspector` (no torch, no checkpoint or ground-truth
code). Line-delimited JSON requests/responses. Unknown tool → error.
Optionally wrapped with macOS `sandbox-exec` (deny network, deny file reads
under `artifacts/private` and `results`) when `interpretation.os_sandbox:
true`.

**prompts.py** — `build_protocol_a(export)`, `build_protocol_b(export,
candidate_order)`, `build_architecture_only(protocol)`; templates carry a
`PROMPT_VERSION`. Protocol A templates are leak-scanned for task words.

**llm_client.py** — `LLMClient(cfg)` around `anthropic.Anthropic()`
(credentials from environment only). Streaming `messages.stream(...)
.get_final_message()`; adaptive thinking with `display: "summarized"`;
`output_config.effort` from config; explicit `max_tokens`; prompt caching on
the stable prefix; typed error handling; SDK retries plus a bounded outer
retry for 429/5xx; per-call token/cost accounting; records `request_id` and
the served `model`. Secrets redacted from logs. Server-side refusal fallbacks
are **off by default** (a fallback would silently change the model under
test); a refusal is recorded as an invalid response.

**interpreter.py** — `run_session(unit_export, condition, cfg) ->
SessionRecord`. Fresh message list per session; manual tool loop with a hard
cap on turns; each tool call proxied to the tool server and logged
(`name, args, result_sha`); session ends on `submit_interpretation` or cap.
Output validated against the protocol schema. Nothing from other sessions is
ever included.

**evaluation.py** — loads frozen predictions, verifies freeze hashes, joins
with `ground_truth.json`. Protocol B: accuracy, confusion, per-task, per-seed,
calibration, invalid rate. Protocol A: rule parsing, agreement with each
ground-truth task and with the network's own thresholded output on a private
grid. Permutation pairs: agreement, confidence deltas. Controls scored
separately. Produces `results/metrics/<stage>.json` and expansion verdict.

**statistics.py** — Wilson interval, exact one-sided binomial p vs 1/3,
Brier score, reliability bins, Cohen's κ.

**baselines.py** — defines control units: chance (analytic), architecture-only
sessions (no weights), random-init networks (untrained, own seeds).

**experiment.py** — `run_stage(stage_name, cfg)` executing the pipeline steps
in order, with stage markers in `artifacts/manifests/stages.json`;
`freeze_predictions(stage)`; `update_status_doc()`; `check_expansion()`.

**reporting.py** — figures (matplotlib, Agg backend) and Markdown tables for
`docs/RESULTS.md`.

## 5. Execution architecture

`scripts/run_experiment.py --config configs/default.yaml` runs, for stage
`initial_15`:

1. validate config, record environment and hardware (`torch.backends.mps`),
2. seed, 3. generate datasets (in memory, per network), 4. train 15 networks
sequentially, 5. save + verify checkpoints, 6. create permuted variants (k per
network, verified privately) and random-init controls, 7. export + anonymize,
8. Protocol A sessions, 9. Protocol B sessions (fresh, independent), 10.
permuted units are interpreted inside 8–9 because they are mixed into the
pool, 11. architecture-only sessions, 12. freeze, 13. evaluate, 14. report,
15. update status doc, 16. expansion verdict.

Steps 8–11 need API credentials. Without them the pipeline stops after step 7
with a clear message, and no interpretation results are produced or invented.

Each step writes a marker with input/output hashes; a re-run verifies and
skips completed steps. Interpretation sessions are individually resumable
(a session file existing and valid = done).

`scripts/expand_experiment.py --target-networks 30|60` loads the previous
stage's metrics, checks the pre-registered criteria, trains only missing seed
indices (registry prevents duplicates), creates exports with new IDs,
interprets only new units, and evaluates both the new stage and the pooled
set.

## 6. Hardware plan (M2 Pro, 16 GB)

- Default device **CPU**: 337-parameter models; MPS kernel launch overhead
  exceeds the work, and CPU gives bit-reproducible results. `device: mps` is
  supported in config; ops fall back to CPU if MPS is unavailable.
- `torch.set_num_threads(1)` during training for determinism and low load;
  datasets ≤ 10k×2 float32 (80 KB).
- Sequential training; models deleted and `gc.collect()` between networks.
- Peak memory estimate: < 600 MB (Python + torch import). RSS is logged per
  stage.
- LLM calls are sequential (no concurrency) to keep rate limits and logs
  simple.

## 7. Information isolation

Layers of defence, from strongest to weakest:

1. **Process isolation.** Tools run in a separate interpreter (`-I`, empty
   env, empty CWD) that only receives the export JSON. The agent loop never
   passes paths. The LLM has no filesystem, shell or Python access.
2. **Capability allowlist.** Tool set is fixed per condition; no tool takes
   input-space points, so no forward pass can be requested. Calls to unknown
   tools are rejected and logged.
3. **Data minimisation.** Exports contain only allowlisted keys; IDs are
   shuffled across tasks, permuted variants and random controls; no seeds,
   timestamps, task strings, file names or accuracy numbers.
4. **Automated scans.** Every prompt and every tool response is scanned with
   `scan_for_leaks` before sending; a hit aborts the session.
5. **Prompt instruction.** Last line of defence only.

Accepted residual risk: the model can do arithmetic in its own reasoning, so
it could mentally evaluate the network at points. This cannot be blocked
technically; the prompt forbids it, and transcripts are scanned post hoc for
signs of explicit point-wise forward simulation (reported, not filtered).

## 8. Artifact management and reproducibility

- `artifacts/manifests/registry.jsonl`: every training run (success/failure).
- `artifacts/manifests/<stage>_manifest.json`: config hash, source hash,
  environment, list of units with export SHA-256 and checkpoint SHA-256.
- `artifacts/manifests/<stage>_freeze.json`: SHA-256 of every prediction file
  written before evaluation; evaluation refuses to run if any hash changed.
- `scripts/train_models.py --verify` retrains from manifests and checks
  bitwise-equal checkpoints (CPU).
- Generated artifacts are git-ignored except manifests and metrics.

## 9. Test plan (pytest)

| File | Covers |
|---|---|
| test_tasks.py | task formulas on known points, sampling seeds, balance |
| test_models.py | exact architecture, parameter count 337 |
| test_training.py | tiny run converges; failure recorded not raised |
| test_reproducibility.py | same seed → identical weights; derive_seed stable |
| test_weight_export.py | allowlisted keys, determinism, sha, round-trip |
| test_information_leakage.py | no task words/paths/seeds in exports and prompts; tool server has no torch, rejects unknown tools, has no path args; ID shuffling |
| test_permutation.py | equivalence within tolerance; non-identity perms |
| test_evaluation.py | scoring on synthetic predictions, rule parser safety, response parsing |
| test_expansion.py | duplicate-seed rejection, criteria logic, stage reuse |

Lint/format: `ruff`; types: `mypy` on `src/` (non-strict).
