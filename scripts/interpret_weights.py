"""Run (resumable) interpretation sessions for one protocol of a stage."""

import argparse

from _common import load_dotenv, setup_logging

from weightspace.claude_code_backend import backend_ready
from weightspace.config import load_config
from weightspace.experiment import interpret_stage

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--stage")
ap.add_argument("--protocol", choices=["A", "B"], required=True)
a = ap.parse_args()
load_dotenv()
cfg = load_config(a.config, a.stage)
setup_logging(cfg.root, f"interpret_{cfg.stage}_{a.protocol}")
ready, why = backend_ready(cfg.interpretation)
if not ready:
    raise SystemExit(f"Interpretation backend not ready: {why}. Nothing was run.")
print(interpret_stage(cfg, a.protocol))
