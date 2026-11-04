"""Train (or verify) every network of a stage. Use --verify to retrain and compare hashes."""

import argparse

from _common import setup_logging

from weightspace.config import load_config
from weightspace.experiment import run_checks, train_stage

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--stage")
ap.add_argument("--verify", action="store_true", help="retrain and compare state hashes")
a = ap.parse_args()
cfg = load_config(a.config, a.stage)
setup_logging(cfg.root, "train")
validated = train_stage(cfg)
for e in validated:
    print(f"{e['task']:9s} seed {e['seed_index']:3d}  test_acc={e['test_acc']:.4f}  "
          f"epochs={e['epochs']}  {e['wall_time_s']}s")
if a.verify:
    print(run_checks(cfg, validated))
