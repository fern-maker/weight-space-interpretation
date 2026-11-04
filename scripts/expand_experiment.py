"""Expand to 30 or 60 networks if the pre-registered criteria of the previous stage pass."""

import argparse
import json

from _common import load_dotenv, setup_logging

from weightspace.config import load_config, stage_chain
from weightspace.experiment import run_stage
from weightspace.reporting import make_figures
from weightspace.status import update_status

STAGE_FOR = {30: "expand_30", 60: "expand_60"}
ap = argparse.ArgumentParser()
ap.add_argument("--target-networks", type=int, choices=[30, 60], required=True)
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
load_dotenv()
cfg = load_config(a.config, STAGE_FOR[a.target_networks])
setup_logging(cfg.root, f"run_{cfg.stage}")
summary = run_stage(cfg)  # raises if previous stage incomplete or criteria not met
figs = make_figures(cfg.root, cfg.stage, stage_chain(cfg))
update_status(cfg, summary)
m = summary.pop("metrics", None)
print(json.dumps(summary, indent=1, default=str))
if m:
    print("expansion:", json.dumps(m["expansion"], indent=1))
print("figures:", figs)
