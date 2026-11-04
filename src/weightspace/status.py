"""Write an auto-generated per-stage block into docs/EXPERIMENT_STATUS.md."""

from __future__ import annotations

import re
from pathlib import Path

from .config import ExperimentConfig
from .experiment import Paths, _markers, now

REPO = Path(__file__).resolve().parents[2]


def stage_block(cfg: ExperimentConfig, summary: dict) -> str:
    P = Paths(cfg)
    marks = _markers(P).get(cfg.stage, {})
    lines = [f"### Stage `{cfg.stage}` (auto-generated {now()})", ""]
    lines.append(f"- Root: `{cfg.root.relative_to(REPO) if cfg.root != REPO else '.'}`; "
                 f"counts toward study: {cfg.stage_cfg.get('counts_toward_study')}")
    lines.append(f"- Steps done: {', '.join(marks) or 'none'}")
    lines.append(f"- Validated networks: {summary.get('validated')}; units: {summary.get('units')}")
    if "train" in marks:
        fails = marks["train"].get("failed_runs", [])
        lines.append(f"- Failed training runs: {len(fails)}")
    if "checks" in marks:
        lines.append(f"- Leak tests passed: {marks['checks']['leak_ok']}; "
                     f"reproducible: {marks['checks']['reproducible']}")
    for p in ("A", "B"):
        r = summary.get(f"interpret_{p}")
        if r:
            lines.append(f"- Protocol {p}: complete={r['complete']}, new={r['done']}, "
                         f"skipped={r['skipped']}, stage spend ${r['spent_usd']:.2f}")
    if summary.get("stopped"):
        lines.append(f"- **Stopped:** {summary['stopped']}")
    m = summary.get("metrics")
    if m:
        bo, ao = m["B"]["original"]["accuracy"], m["A"]["original"]["accuracy"]
        lines.append(f"- Protocol B accuracy (originals): {bo['k']}/{bo['n']}; "
                     f"Protocol A: {ao['k']}/{ao['n']} (reported separately)")
        lines.append(f"- Expansion eligible: {m['expansion']['eligible']}")
    return "\n".join(lines)


def update_status(cfg: ExperimentConfig, summary: dict) -> None:
    path = REPO / "docs" / "EXPERIMENT_STATUS.md"
    text = path.read_text()
    start, end = f"<!-- AUTO {cfg.stage} START -->", f"<!-- AUTO {cfg.stage} END -->"
    block = f"{start}\n{stage_block(cfg, summary)}\n{end}"
    if start in text:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text,
                      flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Auto-generated stage records\n\n" * (
            "## Auto-generated stage records" not in text) + "\n" + block + "\n"
    path.write_text(text)
