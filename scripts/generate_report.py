"""Regenerate figures and the metrics tables for a stage from saved metrics."""

import argparse

from _common import setup_logging  # noqa: F401

from weightspace.config import load_config, stage_chain
from weightspace.reporting import make_figures, metrics_tables
from weightspace.utils import read_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--stage")
a = ap.parse_args()
cfg = load_config(a.config, a.stage)
print(make_figures(cfg.root, cfg.stage, stage_chain(cfg)))
mp = cfg.results_dir / "metrics" / f"{cfg.stage}.json"
if mp.exists():
    text = metrics_tables(read_json(mp))
    (cfg.results_dir / "metrics" / f"{cfg.stage}_tables.md").write_text(text + "\n")
    print(text)
