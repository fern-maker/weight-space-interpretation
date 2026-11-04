# Weight-space interpretation experiment

Can an LLM infer what a small neural network computes from its weights alone?
A controlled 2-D analog of Experiment 1 in "Sparks of Weight-Space
Programming". Start with `AGENTS.md`, then `docs/`.

## Setup (macOS, Apple Silicon)

```bash
UV_CACHE_DIR=$TMPDIR/uvcache uv venv --python 3.12 .venv
UV_CACHE_DIR=$TMPDIR/uvcache uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env (git-ignored)
```

## Run

```bash
.venv/bin/pytest -q                                              # tests
.venv/bin/python scripts/run_experiment.py --config configs/smoke.yaml    # smoke (dev only)
.venv/bin/python scripts/run_experiment.py --config configs/default.yaml  # 15-network study
.venv/bin/python scripts/expand_experiment.py --target-networks 30        # only if eligible
.venv/bin/python scripts/expand_experiment.py --target-networks 60
.venv/bin/python scripts/generate_report.py --stage initial_15
```

Every step is idempotent: re-running skips completed, hash-verified work and
resumes interrupted interpretation passes. Without API credentials the
pipeline stops after training/export and says so.

Other entry points: `scripts/train_models.py --verify` (retrain and compare
hashes), `scripts/interpret_weights.py --protocol A|B`,
`scripts/inspect_weights.py network_001 --what hyperplanes`.

## Layout

`src/weightspace/` code · `configs/` YAML · `artifacts/private/` checkpoints
and ground truth (never shown to the LLM) · `artifacts/public/weight_exports/`
LLM-facing JSON · `artifacts/manifests/` registry, stage manifests, freeze
hashes · `results/` predictions, transcripts, metrics, figures.
