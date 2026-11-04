"""Load, merge and validate YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import sha256_json

REQUIRED_TASKS = ["circle", "diagonal", "square"]


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for inc in raw.pop("include", []) or []:
        merged = _deep_merge(merged, _load_raw(path.parent / inc))
    return _deep_merge(merged, raw)


@dataclass
class ExperimentConfig:
    raw: dict
    path: Path
    root: Path
    stage: str
    stage_cfg: dict
    training: dict
    permutation: dict
    interpretation: dict
    tasks: list[str]
    namespace: str
    device: str
    hash: str = field(default="")

    @property
    def training_hash(self) -> str:
        """Hash of everything that determines trained weights."""
        return sha256_json({"training": self.training, "namespace": self.namespace})

    @property
    def interpretation_hash(self) -> str:
        return sha256_json(self.interpretation)

    def stage_def(self, name: str) -> dict:
        return self.raw["stages"][name]

    # canonical directories ------------------------------------------------
    @property
    def private_dir(self) -> Path:
        return self.root / "artifacts" / "private"

    @property
    def public_dir(self) -> Path:
        return self.root / "artifacts" / "public"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "artifacts" / "manifests"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"


def load_config(path: str | Path, stage: str | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = _load_raw(path)
    exp = raw.get("experiment", {})
    stage = stage or exp.get("stage")
    tasks = exp.get("tasks")
    if tasks != REQUIRED_TASKS:
        raise ConfigError(f"tasks must be exactly {REQUIRED_TASKS}, got {tasks}")
    stages = raw.get("stages", {})
    if stage not in stages:
        raise ConfigError(f"unknown stage {stage!r}; defined: {list(stages)}")
    _validate_stages(stages)
    tr = raw["training"]
    if not 0 < tr["accuracy_target"] <= 1:
        raise ConfigError("accuracy_target must be in (0, 1]")
    if stage == "initial_15" and len(stages[stage]["seed_indices"]) != 5:
        raise ConfigError("initial_15 must use exactly 5 seeds per task (15 networks)")
    device = exp.get("device", "cpu")
    if device not in ("cpu", "mps"):
        raise ConfigError("device must be cpu or mps")
    # repository root is the directory containing configs/
    repo_root = path.parent.parent
    root = (repo_root / raw.get("paths", {}).get("root", ".")).resolve()
    cfg = ExperimentConfig(
        raw=raw,
        path=path,
        root=root,
        stage=stage,
        stage_cfg=stages[stage],
        training=tr,
        permutation=raw["permutation"],
        interpretation=raw["interpretation"],
        tasks=tasks,
        namespace=exp["namespace"],
        device=device,
    )
    cfg.hash = sha256_json({k: v for k, v in raw.items() if k != "paths"})
    return cfg


def _validate_stages(stages: dict[str, Any]) -> None:
    seen: dict[int, str] = {}
    seen_rand: dict[int, str] = {}
    for name, s in stages.items():
        for idx in s["seed_indices"]:
            if idx in seen:
                raise ConfigError(f"seed index {idx} used by both {seen[idx]} and {name}")
            seen[idx] = name
        for idx in s.get("random_init_indices", []):
            if idx in seen_rand:
                raise ConfigError(f"random-init index {idx} reused in {name}")
            seen_rand[idx] = name
        if s.get("previous") and s["previous"] not in stages:
            raise ConfigError(f"stage {name} has unknown previous {s['previous']}")


def stage_chain(cfg: ExperimentConfig, stage: str | None = None) -> list[str]:
    """Stages from the first up to `stage` (inclusive)."""
    stage = stage or cfg.stage
    chain = []
    cur: str | None = stage
    while cur:
        chain.append(cur)
        cur = cfg.raw["stages"][cur].get("previous")
    return list(reversed(chain))
