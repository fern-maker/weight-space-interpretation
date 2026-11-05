"""Stage orchestration. Each step is idempotent and hash-verified."""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import baselines
from .anonymization import allocate_ids, scan_for_leaks
from .checkpointing import Registry, load_checkpoint, save_checkpoint, state_hash
from .config import ExperimentConfig, stage_chain
from .evaluation import evaluate, expansion_criteria
from .permutation import permute_hidden, random_permutations, verify_equivalence
from .prompts import PROMPT_VERSION
from .tasks import sample_inputs
from .training import random_init_network, train_network
from .utils import (
    derive_seed,
    environment_info,
    read_json,
    resource_snapshot,
    sha256_file,
    source_tree_hash,
    write_json,
)
from .weight_exporter import export_text, model_to_export, write_export

log = logging.getLogger("weightspace")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Paths:
    def __init__(self, cfg: ExperimentConfig):
        self.root = cfg.root
        self.ckpt = cfg.private_dir / "checkpoints"
        self.truth = cfg.private_dir / "ground_truth.json"
        self.perm = cfg.private_dir / "permutations"
        self.exports = cfg.public_dir / "weight_exports"
        self.manifests = cfg.manifest_dir
        self.registry = cfg.manifest_dir / "registry.jsonl"
        self.stages = cfg.manifest_dir / "stages.json"
        self.preds = cfg.results_dir / "predictions"
        self.transcripts = cfg.results_dir / "transcripts"
        self.metrics = cfg.results_dir / "metrics"

    def rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.root))


# ------------------------------------------------------------------ markers
def _markers(P: Paths) -> dict:
    return read_json(P.stages) if P.stages.exists() else {}


def mark(P: Paths, stage: str, step: str, info: dict | None = None) -> None:
    m = _markers(P)
    m.setdefault(stage, {})[step] = {"done_at": now(), **(info or {})}
    write_json(P.stages, m)


def is_done(P: Paths, stage: str, step: str) -> bool:
    return step in _markers(P).get(stage, {})


def load_truth(P: Paths) -> dict:
    return read_json(P.truth) if P.truth.exists() else {}


# ------------------------------------------------------------------ training
def train_stage(cfg: ExperimentConfig) -> list[dict]:
    """Train (or verify) every network of the stage. Returns validated entries."""
    P = Paths(cfg)
    reg = Registry(P.registry)
    th = cfg.training_hash
    validated = []
    lo, hi = cfg.training["replacement_seed_range"]
    for task in cfg.tasks:
        for idx in cfg.stage_cfg["seed_indices"]:
            candidates = [idx] + list(range(lo, hi + 1))
            for attempt, seed_idx in enumerate(candidates):
                if attempt > 0 and _replacement_used(reg, task, seed_idx, th, idx):
                    continue
                entry = reg.find(task, seed_idx, th)
                if entry is None:
                    log.info("train %s seed %d", task, seed_idx)
                    model, res = train_network(task, seed_idx, cfg.training, cfg.namespace,
                                               cfg.device)
                    path = P.ckpt / f"{task}_s{seed_idx:03d}.pt"
                    entry = {**res.to_dict(), "training_hash": th, "stage": cfg.stage,
                             "replaces": idx if attempt else None,
                             "checkpoint": P.rel(path), "checkpoint_sha256": save_checkpoint(
                                 model, path), "state_hash": state_hash(model),
                             "registered_at": now(), "resources": resource_snapshot()}
                    reg.add(entry)
                    del model
                    gc.collect()
                elif state_hash(load_checkpoint(cfg.root / entry["checkpoint"])) != \
                        entry["state_hash"]:
                    raise RuntimeError(f"checkpoint corrupted: {entry['checkpoint']}")
                if entry["status"] == "ok":
                    validated.append(entry)
                    break
                log.warning("network %s seed %d failed (test acc %.4f); using replacement",
                            task, seed_idx, entry["test_acc"])
            else:
                raise RuntimeError(f"no validated network for {task} slot {idx}")
    mark(P, cfg.stage, "train", {"n_validated": len(validated),
                                 "failed_runs": [e for e in reg.entries()
                                                 if e["stage"] == cfg.stage
                                                 and e["status"] != "ok"]})
    return validated


def _replacement_used(reg: Registry, task: str, seed_idx: int, th: str, slot: int) -> bool:
    e = reg.find(task, seed_idx, th)
    return e is not None and e.get("replaces") not in (None, slot)


# ------------------------------------------------------------------ units
def build_units(cfg: ExperimentConfig, validated: list[dict]) -> dict:
    """Permuted variants, random-init controls, anonymized exports, ground truth."""
    P = Paths(cfg)
    truth = load_truth(P)
    if any(t["stage"] == cfg.stage for t in truth.values()):
        _verify_exports(P, cfg.stage, truth)
        return truth
    X = sample_inputs(cfg.permutation["verify_points"], derive_seed(cfg.namespace, "verify"))
    units: list[dict] = []
    for e in validated:
        key = f"{e['task']}_s{e['seed_index']:03d}"
        units.append({"kind": "original", "task": e["task"], "seed_index": e["seed_index"],
                      "checkpoint": e["checkpoint"], "key": key})
        model = load_checkpoint(cfg.root / e["checkpoint"])
        for j in range(cfg.stage_cfg["permutations_per_network"]):
            perms = random_permutations(model, derive_seed(cfg.namespace, e["task"],
                                                           e["seed_index"], "permutation", j))
            pm = permute_hidden(model, perms)
            eq = verify_equivalence(model, pm, X, cfg.permutation["atol"])
            if not eq["equivalent"]:
                raise RuntimeError(f"permutation not equivalent for {key}: {eq}")
            path = P.ckpt / f"{key}_perm{j}.pt"
            save_checkpoint(pm, path)
            write_json(P.perm / f"{key}_perm{j}.json",
                       {"perms": [p.tolist() for p in perms], "verification": eq})
            units.append({"kind": "permuted", "task": e["task"], "seed_index": e["seed_index"],
                          "checkpoint": P.rel(path), "parent_key": key, "perm_index": j,
                          "verification": eq, "key": f"{key}_perm{j}"})
    for ridx in cfg.stage_cfg["random_init_indices"]:
        model, seed = random_init_network(cfg.namespace, ridx)
        path = P.ckpt / f"random_init_{ridx:03d}.pt"
        save_checkpoint(model, path)
        units.append({"kind": "random_init", "task": None, "seed_index": None,
                      "random_index": ridx, "init_seed": seed, "checkpoint": P.rel(path),
                      "key": f"random_init_{ridx:03d}"})
    start = 1 + max([int(n.split("_")[1]) for n in truth] or [0])
    allocated = allocate_ids(units, derive_seed(cfg.namespace, cfg.stage, "ids"), start)
    key_to_id = {u["key"]: u["network_id"] for u in allocated}
    digits = int(cfg.interpretation["export_sig_digits"])
    for u in allocated:
        exp = model_to_export(load_checkpoint(cfg.root / u["checkpoint"]), u["network_id"], digits)
        hits = scan_for_leaks(export_text(exp))
        if hits:
            raise RuntimeError(f"leak in export {u['network_id']}: {hits}")
        out = P.exports / f"{u['network_id']}.json"
        write_export(exp, out)
        u["export"] = P.rel(out)
        u["export_sha256"] = sha256_file(out)
        u["stage"] = cfg.stage
        if u["kind"] == "permuted":
            u["parent"] = key_to_id[u["parent_key"]]
        truth[u.pop("network_id")] = u
    write_json(P.truth, truth)
    mark(P, cfg.stage, "units", {"n_units": len(allocated)})
    return truth


def _verify_exports(P: Paths, stage: str, truth: dict) -> None:
    for nid, u in truth.items():
        if u["stage"] == stage and sha256_file(P.root / u["export"]) != u["export_sha256"]:
            raise RuntimeError(f"export changed: {nid}")


def write_stage_manifest(cfg: ExperimentConfig, truth: dict) -> Path:
    from .tools import tool_schema_hash, tool_set

    P = Paths(cfg)
    path = P.manifests / f"{cfg.stage}_manifest.json"
    units = {nid: {"kind": u["kind"], "export_sha256": u["export_sha256"]}
             for nid, u in truth.items() if u["stage"] == cfg.stage}
    write_json(path, {
        "stage": cfg.stage, "written_at": now(), "config_hash": cfg.hash,
        "training_hash": cfg.training_hash, "interpretation_hash": cfg.interpretation_hash,
        "interpretation": cfg.interpretation, "prompt_version": PROMPT_VERSION,
        "tool_schema_sha256": {
            "A": tool_schema_hash(tool_set("A", "weights_tools")),
            "B": tool_schema_hash(tool_set("B", "weights_tools", sorted(cfg.tasks)))},
        "source_tree_sha256": source_tree_hash(), "environment": environment_info(),
        "namespace": cfg.namespace, "stage_def": cfg.stage_cfg,
        "methodology_sha256": sha256_file(Path(__file__).resolve().parents[2] / "docs" /
                                          "METHODOLOGY.md"),
        "units": units})
    return path


# ------------------------------------------------------------------ interpretation
def interpret_stage(cfg: ExperimentConfig, protocol: str, turn_fn=None,
                    request_params: dict | None = None,
                    cc_executable: list[str] | None = None) -> dict:
    """Run every not-yet-recorded session for one protocol (resumable)."""
    from .interpreter import run_session

    P = Paths(cfg)
    if is_done(P, cfg.stage, "freeze"):
        raise RuntimeError("stage already frozen; predictions are immutable")
    truth = load_truth(P)
    icfg = cfg.interpretation
    use_cc = icfg.get("backend", "anthropic_api") == "claude_code" and turn_fn is None
    if use_cc:
        from .claude_code_backend import HarnessError, passing_probe_versions, run_session_cc

        repo = Path(__file__).resolve().parents[2]
        probed = None if cc_executable else passing_probe_versions(repo)
    elif turn_fn is None:
        from .llm_client import LLMClient

        client = LLMClient(icfg)
        turn_fn, request_params = client, client.request_params()
    ids = sorted(n for n, u in truth.items() if u["stage"] == cfg.stage)
    rng = np.random.default_rng(derive_seed(cfg.namespace, cfg.stage, protocol, "order"))
    order = [ids[i] for i in rng.permutation(len(ids))]
    jobs = [(nid, "weights_tools") for nid in order]
    n_arch = cfg.stage_cfg.get(f"arch_only_{protocol.lower()}", 0)
    jobs += [(f"arch_rep_{i + 1}", "architecture_only") for i in range(n_arch)]
    spent = stage_cost(P, cfg.stage)
    deny = [cfg.private_dir, cfg.results_dir, cfg.manifest_dir]
    done = skipped = 0
    stopped: str | None = None
    aborts_path = P.preds / cfg.stage / "_harness_aborts.jsonl"
    for nid, cond in jobs:
        sub = protocol if cond == "weights_tools" else f"arch_{protocol}"
        out = P.preds / cfg.stage / sub / f"{nid}.json"
        if out.exists():
            skipped += 1
            continue
        if spent >= cfg.interpretation["max_cost_usd"]:
            log.error("cost cap reached (%.2f USD); stopping pass", spent)
            break
        cands = None
        if protocol == "B":
            crng = np.random.default_rng(derive_seed(cfg.namespace, cfg.stage, nid, "cands"))
            cands = [sorted(cfg.tasks)[i] for i in crng.permutation(3)]
        export = read_json(P.root / truth[nid]["export"]) if cond == "weights_tools" else None
        shown_id = nid if cond == "weights_tools" else "network_000"
        os_sb = bool(icfg.get("os_sandbox", True))
        if use_cc:
            try:
                rec = run_session_cc(network_id=shown_id, protocol=protocol, condition=cond,
                                     icfg=icfg, export=export, candidate_order=cands,
                                     deny_dirs=deny, os_sandbox=os_sb, executable=cc_executable,
                                     probed_versions=probed)
            except HarnessError as e:
                prior = _counted_aborts(aborts_path, sub, nid)
                _append_jsonl(aborts_path, {"unit": nid, "pass": sub, "at": now(),
                                            "reason": e.reason, "detail": e.detail[:300],
                                            "counted": e.counted})
                max_att = int(icfg.get("claude_code", {}).get("max_attempts_per_session", 3))
                if not (e.counted and prior + 1 >= max_att):
                    stopped = f"harness error on {sub}/{nid}: {e.reason}"
                    log.error("%s; pass stopped (resumable). %s", stopped, e.detail[:300])
                    break
                rec = _harness_failure_record(shown_id, protocol, cond, cands, e)
        else:
            rec = run_session(network_id=shown_id, protocol=protocol, condition=cond,
                              turn_fn=turn_fn, request_params=request_params or {},
                              max_turns=int(icfg["max_turns"]), export=export,
                              candidate_order=cands, deny_dirs=deny, os_sandbox=os_sb)
        transcript = rec.pop("_transcript")
        rec["unit"] = nid
        write_json(P.transcripts / cfg.stage / sub / f"{nid}.json", transcript)
        write_json(out, rec)
        spent += rec["cost_usd"]
        done += 1
        log.info("%s %s %s: %s (%.3f USD, total %.2f)", cfg.stage, sub, nid, rec["status"],
                 rec["cost_usd"], spent)
    complete = all((P.preds / cfg.stage / (protocol if c == "weights_tools" else f"arch_{protocol}")
                    / f"{n}.json").exists() for n, c in jobs)
    if complete:
        mark(P, cfg.stage, f"interpret_{protocol}", {"sessions": len(jobs)})
    return {"done": done, "skipped": skipped, "complete": complete, "spent_usd": spent,
            "stopped": stopped}


def _append_jsonl(path: Path, obj: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _counted_aborts(path: Path, sub: str, nid: str) -> int:
    import json

    if not path.exists():
        return 0
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return sum(1 for r in rows if r["unit"] == nid and r["pass"] == sub and r["counted"])


def _harness_failure_record(shown_id: str, protocol: str, cond: str,
                            cands: list[str] | None, e: Exception) -> dict:
    """METHODOLOGY §7.1: failure after retries = invalid response (kept, not dropped)."""
    return {"network_id": shown_id, "protocol": protocol, "condition": cond,
            "candidate_order": cands, "prompt_version": PROMPT_VERSION,
            "backend": "claude_code", "status": "harness_failure", "valid": False,
            "validation_problems": [str(e)[:300]], "submission": None, "n_turns": 0,
            "n_tool_calls": 0, "tool_calls": [], "turns": [], "usage_total": {},
            "cost_usd": 0.0, "completed_at": now(), "_transcript": []}


def stage_cost(P: Paths, stage: str) -> float:
    return sum(read_json(p).get("cost_usd", 0) for p in (P.preds / stage).glob("*/*.json"))


def freeze_stage(cfg: ExperimentConfig) -> Path:
    P = Paths(cfg)
    path = P.manifests / f"{cfg.stage}_freeze.json"
    files = {P.rel(p): sha256_file(p) for p in sorted((P.preds / cfg.stage).glob("*/*.json"))}
    if path.exists():
        old = read_json(path)
        if old["files"] != files:
            raise RuntimeError("predictions differ from existing freeze manifest")
        return path
    write_json(path, {"stage": cfg.stage, "frozen_at": now(), "files": files})
    mark(P, cfg.stage, "freeze", {"n_files": len(files)})
    return path


# ------------------------------------------------------------------ checks
def run_checks(cfg: ExperimentConfig, validated: list[dict]) -> dict:
    """Leakage test suite + bit-for-bit retraining of this stage's networks."""
    P = Paths(cfg)
    repo = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(repo / "tests" / "test_information_leakage.py")],
                       capture_output=True, text=True, cwd=repo)
    leak_ok = r.returncode == 0
    mismatches = []
    for e in validated:
        model, _ = train_network(e["task"], e["seed_index"], cfg.training, cfg.namespace, "cpu")
        if state_hash(model) != e["state_hash"]:
            mismatches.append(e["checkpoint"])
        del model
        gc.collect()
    checks = {"leak_tests_passed": leak_ok, "leak_test_tail": r.stdout[-300:],
              "reproducible": not mismatches, "mismatches": mismatches,
              "all_validated": all(e["status"] == "ok" for e in validated)}
    write_json(P.manifests / f"{cfg.stage}_checks.json", checks)
    mark(P, cfg.stage, "checks", {"leak_ok": leak_ok, "reproducible": not mismatches})
    return checks


# ------------------------------------------------------------------ evaluation
def evaluate_stage(cfg: ExperimentConfig) -> dict:
    P = Paths(cfg)
    chain = stage_chain(cfg)
    for st in chain:
        if not is_done(P, st, "freeze"):
            raise RuntimeError(f"stage {st} is not frozen; cannot evaluate")
    truth = load_truth(P)
    m = evaluate(cfg.root, chain, truth)
    checks_all = [read_json(P.manifests / f"{st}_checks.json") for st in chain
                  if (P.manifests / f"{st}_checks.json").exists()]
    checks = {k: bool(checks_all) and all(c.get(k) for c in checks_all) for k in
              ("leak_tests_passed", "reproducible", "all_validated")}
    n_orig = m["unit_counts"]["original"]
    m["checks"] = checks
    m["expansion"] = expansion_criteria(m, n_orig, checks) if n_orig else None
    m["evaluated_at"] = now()
    write_json(P.metrics / f"{cfg.stage}.json", m)
    mark(P, cfg.stage, "evaluate", {"eligible": m["expansion"] and m["expansion"]["eligible"]})
    return m


def check_compatible(cfg: ExperimentConfig) -> None:
    """Expansion may only reuse earlier stages run under identical conditions."""
    P = Paths(cfg)
    prev = cfg.stage_cfg.get("previous")
    if not prev:
        return
    mpath = P.manifests / f"{prev}_manifest.json"
    if not mpath.exists() or not is_done(P, prev, "evaluate"):
        raise RuntimeError(f"previous stage {prev} is not complete")
    old = read_json(mpath)
    if old["interpretation_hash"] != cfg.interpretation_hash or \
            old["prompt_version"] != PROMPT_VERSION or old["training_hash"] != cfg.training_hash:
        raise RuntimeError("configuration changed since previous stage; not compatible")
    metrics = read_json(P.metrics / f"{prev}.json")
    if not (metrics.get("expansion") or {}).get("eligible"):
        raise RuntimeError(f"expansion criteria not met at stage {prev}")


def run_stage(cfg: ExperimentConfig, skip_interpretation: bool = False) -> dict:
    """Full pipeline for cfg.stage. Returns a summary dict."""
    from .claude_code_backend import backend_ready

    check_compatible(cfg)
    validated = train_stage(cfg)
    truth = build_units(cfg, validated)
    write_stage_manifest(cfg, truth)
    summary: dict = {"stage": cfg.stage, "validated": len(validated),
                     "units": sum(u["stage"] == cfg.stage for u in truth.values())}
    if not is_done(Paths(cfg), cfg.stage, "checks"):
        summary["checks"] = run_checks(cfg, validated)
    if skip_interpretation:
        summary["stopped"] = "interpretation skipped by request"
        return summary
    ready, why = backend_ready(cfg.interpretation)
    if not ready:
        summary["stopped"] = f"{why}; interpretation not run (no results invented)"
        return summary
    for proto in ("A", "B"):
        r = interpret_stage(cfg, proto)
        summary[f"interpret_{proto}"] = r
        if not r["complete"]:
            summary["stopped"] = (f"protocol {proto} incomplete ({r.get('stopped') or 'cost cap'}"
                                  "); resumable")
            return summary
    freeze_stage(cfg)
    summary["metrics"] = evaluate_stage(cfg)
    return summary


__all__ = ["run_stage", "train_stage", "build_units", "interpret_stage", "freeze_stage",
           "evaluate_stage", "run_checks", "baselines"]
