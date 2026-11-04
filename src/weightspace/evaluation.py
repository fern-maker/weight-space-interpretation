"""Private evaluation of frozen predictions. Never imported by the interpreter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .checkpointing import load_checkpoint
from .expression import RuleError, parse_rule
from .models import logits
from .statistics import binom_tail_ge, brier, cohen_kappa, proportion, reliability
from .tasks import TASKS, grid
from .utils import read_json, sha256_file

TASK_NAMES = list(TASKS)
A_THRESHOLD = 0.90


class FreezeError(RuntimeError):
    pass


def verify_freeze(freeze_path: Path, root: Path) -> dict:
    frozen = read_json(freeze_path)
    for rel, sha in frozen["files"].items():
        p = root / rel
        if not p.exists() or sha256_file(p) != sha:
            raise FreezeError(f"prediction file changed or missing after freeze: {rel}")
    return frozen


def load_predictions(pred_dir: Path) -> dict[str, dict[str, dict]]:
    """{subdir: {name: record}} for A, B, arch_A, arch_B."""
    out: dict[str, dict[str, dict]] = {}
    for sub in ("A", "B", "arch_A", "arch_B"):
        d = pred_dir / sub
        out[sub] = {p.stem: read_json(p) for p in sorted(d.glob("*.json"))} if d.exists() else {}
    return out


# --------------------------------------------------------------------------- B
def score_b(records: dict[str, dict], truth: dict[str, dict], ids: list[str]) -> dict:
    rows = []
    for nid in ids:
        r = records.get(nid)
        sub = (r or {}).get("submission") or {}
        valid = bool(r and r.get("valid"))
        pred = sub.get("predicted_function") if valid else None
        t = truth[nid].get("task")
        rows.append({"network_id": nid, "task": t, "seed_index": truth[nid].get("seed_index"),
                     "pred": pred, "valid": valid,
                     "confidence": sub.get("confidence") if valid else None,
                     "correct": bool(valid and t is not None and pred == t)})
    k = sum(r["correct"] for r in rows)
    n = len(rows)
    out: dict = {"n": n, "rows": rows,
                 "invalid": proportion(sum(not r["valid"] for r in rows), n),
                 "prediction_counts": dict(Counter(r["pred"] or "invalid" for r in rows)),
                 "mean_confidence": _mean([r["confidence"] for r in rows if r["valid"]])}
    if all(r["task"] for r in rows) and n:
        out["accuracy"] = proportion(k, n)
        out["p_chance_ge"] = binom_tail_ge(k, n)
        out["confusion"] = {t: {(p or "invalid"): sum(1 for r in rows
                                                     if r["task"] == t and r["pred"] == p)
                                for p in [*TASK_NAMES, None]} for t in TASK_NAMES}
        out["per_task"] = {t: proportion(sum(r["correct"] for r in rows if r["task"] == t),
                                         sum(1 for r in rows if r["task"] == t))
                           for t in TASK_NAMES}
        seeds = sorted({r["seed_index"] for r in rows})
        out["per_seed"] = {str(s): proportion(sum(r["correct"] for r in rows
                                                  if r["seed_index"] == s),
                                              sum(1 for r in rows if r["seed_index"] == s))
                           for s in seeds}
        vr = [r for r in rows if r["valid"]]
        out["brier"] = brier([r["confidence"] for r in vr], [r["correct"] for r in vr])
        out["reliability"] = reliability([r["confidence"] for r in vr], [r["correct"] for r in vr])
        out["mean_conf_correct"] = _mean([r["confidence"] for r in vr if r["correct"]])
        out["mean_conf_incorrect"] = _mean([r["confidence"] for r in vr if not r["correct"]])
    return out


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# --------------------------------------------------------------------------- A
_GRID = None
_TASK_GRIDS: dict[str, np.ndarray] = {}


def _grid() -> np.ndarray:
    global _GRID
    if _GRID is None:
        _GRID = grid(201)
        for t, f in TASKS.items():
            _TASK_GRIDS[t] = f(_GRID).astype(bool)
    return _GRID


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def score_rule(rule_text: str | None, net_out: np.ndarray | None) -> dict:
    X = _grid()
    try:
        R = parse_rule(rule_text or "").evaluate(X)
    except (RuleError, ZeroDivisionError, ValueError, TypeError) as e:
        return {"parsed": False, "error": str(e)}
    agree = {t: float(np.mean(R == g)) for t, g in _TASK_GRIDS.items()}
    implied = max(agree, key=lambda t: agree[t])
    out = {"parsed": True, "agreement": agree, "iou": {t: _iou(R, g) for t, g in
                                                        _TASK_GRIDS.items()},
           "implied_family": implied, "positive_fraction": float(R.mean())}
    if net_out is not None:
        out["fidelity"] = float(np.mean(R == net_out))
        out["fidelity_iou"] = _iou(R, net_out)
    return out


def network_grid_output(ckpt: Path) -> np.ndarray:
    return (logits(load_checkpoint(ckpt), _grid()) > 0).numpy()


def score_a(records: dict[str, dict], truth: dict[str, dict], ids: list[str],
            root: Path) -> dict:
    rows = []
    for nid in ids:
        r = records.get(nid)
        sub = (r or {}).get("submission") or {}
        info = truth.get(nid, {})
        ckpt = info.get("checkpoint")
        net = network_grid_output(root / ckpt) if ckpt else None
        sc = score_rule(sub.get("proposed_rule"), net) if r and r.get("valid") else \
            {"parsed": False, "error": "invalid session"}
        t = info.get("task")
        correct = bool(sc.get("parsed") and t and sc["implied_family"] == t
                       and sc["agreement"][t] >= A_THRESHOLD)
        rows.append({"network_id": nid, "task": t, "seed_index": info.get("seed_index"),
                     "rule": sub.get("proposed_rule"), "rule_confidence":
                     sub.get("rule_confidence"), "valid": sc.get("parsed", False),
                     "score": sc, "correct": correct})
    n = len(rows)
    out: dict = {"n": n, "rows": rows,
                 "invalid": proportion(sum(not r["valid"] for r in rows), n),
                 "implied_family_counts": dict(Counter(
                     r["score"].get("implied_family", "invalid") for r in rows)),
                 "mean_fidelity": _mean([r["score"].get("fidelity") for r in rows])}
    if n and all(r["task"] for r in rows):
        out["accuracy"] = proportion(sum(r["correct"] for r in rows), n)
        out["implied_family_matches_task"] = proportion(
            sum(r["valid"] and r["score"]["implied_family"] == r["task"] for r in rows), n)
        out["per_task"] = {t: proportion(sum(r["correct"] for r in rows if r["task"] == t),
                                         sum(1 for r in rows if r["task"] == t))
                           for t in TASK_NAMES}
        out["mean_agreement_true_task"] = _mean(
            [r["score"]["agreement"][r["task"]] for r in rows if r["valid"]])
    return out


# ----------------------------------------------------------------------- pairs
def pair_metrics(b_orig: dict, b_perm: dict, a_orig: dict, a_perm: dict,
                 pairs: list[tuple[str, str]]) -> dict:
    bo = {r["network_id"]: r for r in b_orig["rows"]}
    bp = {r["network_id"]: r for r in b_perm["rows"]}
    ao = {r["network_id"]: r for r in a_orig["rows"]}
    ap = {r["network_id"]: r for r in a_perm["rows"]}
    rows = []
    for o, p in pairs:
        fo = ao[o]["score"].get("implied_family")
        fp = ap[p]["score"].get("implied_family")
        co, cp = bo[o]["confidence"], bp[p]["confidence"]
        rows.append({
            "original": o, "permuted": p, "task": bo[o]["task"],
            "B_orig": bo[o]["pred"], "B_perm": bp[p]["pred"],
            "B_stable": bo[o]["pred"] is not None and bo[o]["pred"] == bp[p]["pred"],
            "B_conf_delta": (cp - co) if co is not None and cp is not None else None,
            "A_orig_family": fo, "A_perm_family": fp,
            "A_stable": fo is not None and fo == fp,
            "A_true_agreement_delta": (
                ap[p]["score"]["agreement"][ap[p]["task"]]
                - ao[o]["score"]["agreement"][ao[o]["task"]]
                if ao[o]["valid"] and ap[p]["valid"] else None),
        })
    n = len(rows)
    return {
        "n_pairs": n, "rows": rows,
        "B_agreement": proportion(sum(r["B_stable"] for r in rows), n),
        "B_kappa": cohen_kappa([str(r["B_orig"]) for r in rows], [str(r["B_perm"]) for r in rows]),
        "B_mean_abs_conf_delta": _mean([abs(r["B_conf_delta"]) for r in rows
                                        if r["B_conf_delta"] is not None]),
        "A_family_agreement": proportion(sum(r["A_stable"] for r in rows), n),
        "unstable_B": [r for r in rows if not r["B_stable"]],
    }


# ------------------------------------------------------------------ expansion
def expansion_criteria(m: dict, n_networks: int, checks: dict) -> dict:
    """Pre-registered rule from METHODOLOGY §8."""
    big = n_networks >= 30
    acc = m["B"]["original"]["accuracy"]
    per_task = m["B"]["original"]["per_task"]
    best = max(per_task, key=lambda t: per_task[t]["k"])
    rest_k = acc["k"] - per_task[best]["k"]
    rest_n = acc["n"] - per_task[best]["n"]
    perm_acc = m["B"]["permuted"]["accuracy"]
    agree = m["pairs"]["B_agreement"]
    need = {"acc": 18 if big else 9, "rest": 8 if big else 4, "perm": 16 if big else 8,
            "agree": 20 if big else 10}
    if n_networks not in (15, 30):
        scale = n_networks / (30 if big else 15)
        need = {k: int(np.ceil(v * scale)) for k, v in need.items()}
    c = {
        "1_all_networks_validated": checks.get("all_validated", False),
        "2_leak_and_reproducibility_checks": bool(checks.get("leak_tests_passed")
                                                  and checks.get("reproducible")),
        "3_mc_accuracy": acc["k"] >= need["acc"],
        "4_not_single_task": rest_k >= need["rest"],
        "5_permutation_no_collapse": perm_acc["k"] >= need["perm"] and agree["k"] >= need["agree"],
    }
    if big:
        c["6_wilson_lower_bound_above_chance"] = acc["wilson95"][0] > 1 / 3
    return {"criteria": c, "thresholds": need, "best_task_removed": best,
            "rest": {"k": rest_k, "n": rest_n}, "eligible": all(c.values())}


def evaluate(root: Path, stages: list[str], truth: dict[str, dict]) -> dict:
    """Pooled evaluation across `stages` (each must be frozen)."""
    preds: dict[str, dict[str, dict]] = {s: {} for s in ("A", "B", "arch_A", "arch_B")}
    for st in stages:
        verify_freeze(root / "artifacts" / "manifests" / f"{st}_freeze.json", root)
        for sub, recs in load_predictions(root / "results" / "predictions" / st).items():
            preds[sub].update({f"{st}:{k}" if sub.startswith("arch") else k: v
                               for k, v in recs.items()})
    units = {nid: t for nid, t in truth.items() if t["stage"] in stages}
    by_kind = {k: sorted(n for n, t in units.items() if t["kind"] == k)
               for k in ("original", "permuted", "random_init")}
    m: dict = {"stages": stages, "unit_counts": {k: len(v) for k, v in by_kind.items()},
               "A": {}, "B": {}}
    for kind, ids in by_kind.items():
        m["B"][kind] = score_b(preds["B"], units, ids)
        m["A"][kind] = score_a(preds["A"], units, ids, root)
    pairs = sorted((units[p]["parent"], p) for p in by_kind["permuted"])
    m["pairs"] = pair_metrics(m["B"]["original"], m["B"]["permuted"], m["A"]["original"],
                              m["A"]["permuted"], pairs)
    m["arch_only"] = {
        "B": {"n": len(preds["arch_B"]), "prediction_counts": dict(Counter(
            (r.get("submission") or {}).get("predicted_function", "invalid")
            if r.get("valid") else "invalid" for r in preds["arch_B"].values())),
            "confidences": [(r.get("submission") or {}).get("confidence")
                            for r in preds["arch_B"].values()]},
        "A": {"n": len(preds["arch_A"]), "rows": [
            {"id": k, "rule": (r.get("submission") or {}).get("proposed_rule"),
             "score": score_rule((r.get("submission") or {}).get("proposed_rule"), None)
             if r.get("valid") else {"parsed": False}}
            for k, r in preds["arch_A"].items()]},
    }
    usage = Counter()
    for sub in preds.values():
        for r in sub.values():
            usage["sessions"] += 1
            usage["tool_calls"] += r.get("n_tool_calls", 0)
            usage["turns"] += r.get("n_turns", 0)
            for k, v in (r.get("usage_total") or {}).items():
                usage[k] += v
    m["cost_usd"] = round(sum(r.get("cost_usd", 0) for s in preds.values() for r in s.values()), 3)
    m["usage"] = dict(usage)
    m["models_served"] = sorted({ms for s in preds.values() for r in s.values()
                                 for ms in r.get("models_served", [])})
    return m
