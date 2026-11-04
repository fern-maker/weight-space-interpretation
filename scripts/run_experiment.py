"""Full pipeline for one stage (default: initial_15). Idempotent and resumable."""

import argparse
import json

from _common import load_dotenv, setup_logging

from weightspace.config import load_config, stage_chain
from weightspace.experiment import run_stage
from weightspace.reporting import make_figures
from weightspace.status import update_status

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--stage")
ap.add_argument("--skip-interpretation", action="store_true")
a = ap.parse_args()
load_dotenv()
cfg = load_config(a.config, a.stage)
setup_logging(cfg.root, f"run_{cfg.stage}")
summary = run_stage(cfg, skip_interpretation=a.skip_interpretation)
figs = make_figures(cfg.root, cfg.stage, stage_chain(cfg))
update_status(cfg, summary)
m = summary.pop("metrics", None)
print(json.dumps(summary, indent=1, default=str))
if m:
    print("expansion:", json.dumps(m["expansion"], indent=1))
print("figures:", figs)
