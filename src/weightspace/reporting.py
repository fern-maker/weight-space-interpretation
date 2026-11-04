"""Figures and Markdown tables. Report-only; never shown to the interpreter."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .checkpointing import load_checkpoint  # noqa: E402
from .evaluation import _grid, network_grid_output  # noqa: E402
from .expression import parse_rule  # noqa: E402
from .tasks import TASKS  # noqa: E402
from .utils import read_json, read_jsonl  # noqa: E402
from .weight_inspector import compute_singular_values, inspect_weight_matrix  # noqa: E402

# validated categorical order (dataviz reference palette, light mode)
COLORS = {"circle": "#2a78d6", "diagonal": "#eb6834", "square": "#1baf7a",
          "random_init": "#eda100", "arch_only": "#52514e"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
TASK_NAMES = list(TASKS)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9, "lines.linewidth": 2,
    "legend.frameon": False,
})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def fig_training(registry: list[dict], out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, t in zip(axes, TASK_NAMES, strict=True):
        for e in [e for e in registry if e["task"] == t]:
            ax.plot(e["loss_curve"], color=COLORS[t], alpha=0.8, linewidth=1.2)
        ax.set_title(f"{t}: training loss")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("BCE loss (log)")
    names = [_save(fig, out / "training_loss.png")]
    fig, ax = plt.subplots(figsize=(7, 3))
    xs = np.arange(len(registry))
    ax.bar(xs, [e["test_acc"] for e in registry], color=[COLORS[e["task"]] for e in registry],
           width=0.7)
    ax.axhline(0.95, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(registry) - 0.5, 0.951, "95% target", ha="right", va="bottom", color=MUTED)
    ax.set_ylim(0.9, 1.0)
    ax.set_xticks(xs, [f"{e['task'][:3]}{e['seed_index']}" for e in registry], rotation=90)
    ax.set_ylabel("held-out test accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in TASK_NAMES]
    ax.legend(handles, TASK_NAMES, loc="lower left", ncol=3)
    names.append(_save(fig, out / "test_accuracy.png"))
    return names


def fig_weights(registry: list[dict], root: Path, out: Path) -> list[str]:
    examples = [next(e for e in registry if e["task"] == t) for t in TASK_NAMES]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8),
                             gridspec_kw={"width_ratios": [1, 6, 6]})
    for row, e in zip(axes, examples, strict=True):
        m = load_checkpoint(root / e["checkpoint"])
        for ax, layer in zip(row, ["layer1", "layer2", "layer3"], strict=True):
            w = np.array(inspect_weight_matrix(m, layer))
            if layer == "layer3":
                w = w.reshape(4, 4)
            lim = np.abs(w).max()
            im = ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{e['task']} s{e['seed_index']} {layer}"
                         + (" (16 values as 4x4)" if layer == "layer3" else ""), fontsize=8)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    names = [_save(fig, out / "weight_heatmaps.png")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, layer in zip(axes, ["layer1", "layer2"], strict=True):
        for t in TASK_NAMES:
            for i, e in enumerate([e for e in registry if e["task"] == t]):
                sv = compute_singular_values(load_checkpoint(root / e["checkpoint"]), layer)
                ax.plot(range(1, len(sv) + 1), sv, color=COLORS[t], alpha=0.7, linewidth=1.2,
                        marker="o", markersize=3, label=t if i == 0 else None)
        ax.set_title(f"singular values, {layer}")
        ax.set_xlabel("index")
    axes[0].legend()
    names.append(_save(fig, out / "singular_values.png"))
    return names


def fig_confusion(b: dict, out: Path, name: str, title: str) -> str:
    preds = [*TASK_NAMES, "invalid"]
    mat = np.array([[b["confusion"][t].get(p, 0) for p in preds] for t in TASK_NAMES])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.max()))
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else INK)
    ax.set_xticks(range(4), [*TASK_NAMES, "invalid"])
    ax.set_yticks(range(3), TASK_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true task")
    ax.set_title(title)
    return _save(fig, out / name)


def fig_accuracy_by_task(m: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3))
    groups = [("B original", m["B"]["original"]), ("B permuted", m["B"]["permuted"]),
              ("A original", m["A"]["original"]), ("A permuted", m["A"]["permuted"])]
    w = 0.2
    for gi, (label, g) in enumerate(groups):
        for ti, t in enumerate(TASK_NAMES):
            p = g["per_task"][t]
            x = ti + (gi - 1.5) * w
            ax.bar(x, p["rate"], width=w * 0.9, color=COLORS[t],
                   alpha=[1, 0.55, 1, 0.55][gi], hatch=[None, None, "//", "//"][gi],
                   edgecolor=SURFACE)
            lo, hi = p["wilson95"]
            ax.plot([x, x], [lo, hi], color=MUTED, linewidth=1)
    ax.axhline(1 / 3, color=MUTED, linestyle="--", linewidth=1)
    ax.text(2.45, 1 / 3 + 0.01, "chance (B)", ha="right", color=MUTED)
    ax.set_xticks(range(3), TASK_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (95% Wilson)")
    ax.set_title("Per-task accuracy. Bars left→right: B orig, B perm, A orig, A perm")
    return _save(fig, out / "accuracy_by_task.png")


def fig_confidence(m: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3))
    series = []
    rows = m["B"]["original"]["rows"]
    series.append(("trained, correct", [r["confidence"] for r in rows if r["correct"]],
                   COLORS["circle"]))
    series.append(("trained, wrong", [r["confidence"] for r in rows
                                      if r["valid"] and not r["correct"]], COLORS["diagonal"]))
    series.append(("random-init", [r["confidence"] for r in m["B"]["random_init"]["rows"]
                                   if r["valid"]], COLORS["random_init"]))
    series.append(("architecture-only", [c for c in m["arch_only"]["B"]["confidences"]
                                         if c is not None], COLORS["arch_only"]))
    for i, (label, vals, c) in enumerate(series):
        ys = np.full(len(vals), i) + np.linspace(-0.15, 0.15, max(1, len(vals)))[:len(vals)]
        ax.scatter(vals, ys, color=c, s=40, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if vals:
            ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.5)
    ax.set_yticks(range(len(series)), [f"{s[0]} (n={len(s[1])})" for s in series])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Protocol B stated confidence (black tick = mean)")
    return _save(fig, out / "confidence_distributions.png")


def fig_permutation(m: dict, out: Path) -> str:
    rows = m["pairs"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    co = {r["network_id"]: r["confidence"] for r in m["B"]["original"]["rows"]}
    cp = {r["network_id"]: r["confidence"] for r in m["B"]["permuted"]["rows"]}
    for r in rows:
        a, b = co.get(r["original"]), cp.get(r["permuted"])
        if a is None or b is None:
            continue
        ax.scatter(a, b, s=55, color=COLORS[r["task"]], marker="o" if r["B_stable"] else "X",
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("confidence, original")
    ax.set_ylabel("confidence, permuted copy")
    ag = m["pairs"]["B_agreement"]
    ax.set_title(f"Protocol B original vs permuted: {ag['k']}/{ag['n']} same choice "
                 "(o same, x changed)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[t]) for t in TASK_NAMES]
    ax.legend(handles, TASK_NAMES, loc="lower right")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    return _save(fig, out / "permutation_agreement.png")


def fig_sample_sizes(all_metrics: list[dict], out: Path) -> str | None:
    if not all_metrics:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ns, rates, los, his = [], [], [], []
    for m in all_metrics:
        a = m["B"]["original"]["accuracy"]
        ns.append(a["n"])
        rates.append(a["rate"])
        los.append(a["wilson95"][0])
        his.append(a["wilson95"][1])
    ax.fill_between(ns, los, his, color=COLORS["circle"], alpha=0.2, linewidth=0)
    ax.plot(ns, rates, color=COLORS["circle"], marker="o", markersize=8, label="Protocol B")
    a_rates = [m["A"]["original"]["accuracy"]["rate"] for m in all_metrics]
    ax.plot(ns, a_rates, color=COLORS["diagonal"], marker="s", markersize=8, label="Protocol A")
    ax.axhline(1 / 3, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("trained networks (cumulative)")
    ax.set_ylabel("accuracy, original networks")
    ax.legend()
    return _save(fig, out / "accuracy_vs_sample_size.png")


def fig_boundaries(m: dict, truth: dict, root: Path, out: Path) -> str:
    """Report-only: each original network's region vs its Protocol A rule."""
    rows = sorted(m["A"]["original"]["rows"], key=lambda r: (r["task"], r["seed_index"]))
    n = len(rows)
    cols = 5
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2, nrows * 2.1), squeeze=False)
    X = _grid()
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows, strict=False):
        net = network_grid_output(root / truth[r["network_id"]]["checkpoint"]).reshape(201, 201)
        ax.imshow(net, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys", vmin=0, vmax=2.5)
        try:
            R = parse_rule(r["rule"] or "").evaluate(X).reshape(201, 201)
            ax.contour(np.linspace(-1, 1, 201), np.linspace(-1, 1, 201), R.astype(float),
                       levels=[0.5], colors=[COLORS[r["task"]]], linewidths=1.5)
        except Exception:
            pass
        ax.set_title(f"{r['task']} s{r['seed_index']} {'✓' if r['correct'] else '✗'}",
                     fontsize=8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Grey: network class-1 region. Line: Protocol A proposed rule boundary",
                 fontsize=9)
    return _save(fig, out / "protocol_a_boundaries.png")


def make_figures(root: Path, stage: str, chain: list[str]) -> list[str]:
    out = root / "results" / "figures" / stage
    registry = [e for e in read_jsonl(root / "artifacts" / "manifests" / "registry.jsonl")
                if e["stage"] in chain and e["status"] == "ok"]
    registry.sort(key=lambda e: (e["task"], e["seed_index"]))
    names = fig_training(registry, out) + fig_weights(registry, root, out)
    mpath = root / "results" / "metrics" / f"{stage}.json"
    if mpath.exists():
        m = read_json(mpath)
        truth = read_json(root / "artifacts" / "private" / "ground_truth.json")
        names.append(fig_confusion(m["B"]["original"], out, "confusion_B_original.png",
                                   "Protocol B, original networks"))
        names.append(fig_confusion(m["B"]["permuted"], out, "confusion_B_permuted.png",
                                   "Protocol B, permuted copies"))
        names.append(fig_accuracy_by_task(m, out))
        names.append(fig_confidence(m, out))
        names.append(fig_permutation(m, out))
        names.append(fig_boundaries(m, truth, root, out))
        series = [read_json(root / "results" / "metrics" / f"{s}.json") for s in chain
                  if (root / "results" / "metrics" / f"{s}.json").exists()]
        n = fig_sample_sizes(series, out)
        if n:
            names.append(n)
    return names


def _p(x: dict) -> str:
    if x.get("rate") is None:
        return "n/a"
    lo, hi = x["wilson95"]
    return f"{x['k']}/{x['n']} ({x['rate']:.0%}; 95% CI {lo:.0%}–{hi:.0%})"


def metrics_tables(m: dict) -> str:
    """Markdown tables of the key numbers (inserted in RESULTS.md)."""
    L = []
    bo, bp = m["B"]["original"], m["B"]["permuted"]
    ao, ap = m["A"]["original"], m["A"]["permuted"]
    L.append(f"Stages: {', '.join(m['stages'])}. Units: {m['unit_counts']}. "
             f"Models served: {', '.join(m['models_served']) or 'n/a'}. "
             f"API cost: ${m['cost_usd']:.2f}.\n")
    L.append("| Metric | Protocol B (multiple choice) | Protocol A (open-ended) |")
    L.append("|---|---|---|")
    L.append(f"| Accuracy, original networks | {_p(bo['accuracy'])} | {_p(ao['accuracy'])} |")
    L.append(f"| Accuracy, permuted copies | {_p(bp['accuracy'])} | {_p(ap['accuracy'])} |")
    L.append(f"| Invalid responses, original | {_p(bo['invalid'])} | {_p(ao['invalid'])} |")
    L.append(f"| P(≥k correct) under chance 1/3 | {bo['p_chance_ge']:.4f} | — |")
    L.append(f"| Orig/perm agreement | {_p(m['pairs']['B_agreement'])} | "
             f"{_p(m['pairs']['A_family_agreement'])} (implied family) |")
    L.append(f"| Mean confidence (orig) | {_fmt(bo['mean_confidence'])} | — |")
    L.append(f"| Brier score (orig) | {_fmt(bo.get('brier'))} | — |")
    L.append(f"| Mean agreement of rule with true task (orig) | — | "
             f"{_fmt(ao.get('mean_agreement_true_task'))} |")
    L.append(f"| Mean fidelity of rule to network (orig) | — | {_fmt(ao['mean_fidelity'])} |")
    L.append("")
    L.append("Per task (original networks):\n")
    L.append("| Task | B accuracy | A accuracy | B permuted | A permuted |")
    L.append("|---|---|---|---|---|")
    for t in TASK_NAMES:
        L.append(f"| {t} | {_p(bo['per_task'][t])} | {_p(ao['per_task'][t])} | "
                 f"{_p(bp['per_task'][t])} | {_p(ap['per_task'][t])} |")
    L.append("")
    L.append("Protocol B confusion (rows = true task, original networks):\n")
    L.append("| true \\ predicted | circle | diagonal | square | invalid |")
    L.append("|---|---|---|---|---|")
    for t in TASK_NAMES:
        c = bo["confusion"][t]
        L.append(f"| {t} | {c.get('circle', 0)} | {c.get('diagonal', 0)} | "
                 f"{c.get('square', 0)} | {c.get('invalid', 0)} |")
    L.append("")
    rb, ra = m["B"]["random_init"], m["A"]["random_init"]
    L.append(f"Random-init controls (n={rb['n']}): Protocol B choices "
             f"{rb['prediction_counts']}, mean confidence {_fmt(rb['mean_confidence'])}; "
             f"Protocol A implied families {ra['implied_family_counts']}, mean fidelity "
             f"{_fmt(ra['mean_fidelity'])}.\n")
    L.append(f"Architecture-only: Protocol B choices {m['arch_only']['B']['prediction_counts']}, "
             f"confidences {m['arch_only']['B']['confidences']}; Protocol A rules "
             f"{[r['rule'] for r in m['arch_only']['A']['rows']]}.\n")
    e = m.get("expansion")
    if e:
        L.append("Expansion criteria (METHODOLOGY §8):\n")
        L.append("| Criterion | Pass |")
        L.append("|---|---|")
        for k, v in e["criteria"].items():
            L.append(f"| {k} | {'yes' if v else 'no'} |")
        L.append(f"\nThresholds: {e['thresholds']}; remaining after removing best task "
                 f"({e['best_task_removed']}): {e['rest']['k']}/{e['rest']['n']}. "
                 f"**Eligible: {'yes' if e['eligible'] else 'no'}.**\n")
    return "\n".join(L)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"
