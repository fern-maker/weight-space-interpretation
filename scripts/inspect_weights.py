"""Print the public export / analytic inspection of one network id (no forward passes)."""

import argparse
import json

from _common import setup_logging  # noqa: F401

from weightspace import weight_inspector as wi
from weightspace.config import load_config
from weightspace.utils import read_json

ap = argparse.ArgumentParser()
ap.add_argument("network_id")
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--what", default="architecture",
                choices=["architecture", "stats", "svd", "hyperplanes", "raw"])
a = ap.parse_args()
cfg = load_config(a.config)
exp = read_json(cfg.public_dir / "weight_exports" / f"{a.network_id}.json")
out = {"architecture": lambda: wi.inspect_architecture(exp),
       "stats": lambda: wi.compute_weight_statistics(exp),
       "svd": lambda: {name: wi.compute_singular_values(exp, name) for name in wi.LAYERS},
       "hyperplanes": lambda: wi.first_layer_hyperplanes(exp),
       "raw": lambda: exp}[a.what]()
print(json.dumps(out, indent=1))
