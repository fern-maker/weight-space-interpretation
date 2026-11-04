"""Run (resumable) interpretation sessions for one protocol of a stage."""

import argparse

from _common import load_dotenv, setup_logging

from weightspace.config import load_config
from weightspace.experiment import interpret_stage
from weightspace.llm_client import has_credentials

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--stage")
ap.add_argument("--protocol", choices=["A", "B"], required=True)
a = ap.parse_args()
load_dotenv()
cfg = load_config(a.config, a.stage)
setup_logging(cfg.root, f"interpret_{cfg.stage}_{a.protocol}")
if not has_credentials():
    raise SystemExit("No Anthropic credentials (set ANTHROPIC_API_KEY). Nothing was run.")
print(interpret_stage(cfg, a.protocol))
