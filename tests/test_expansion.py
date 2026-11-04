"""End-to-end pipeline with a scripted LLM on a tiny config (no API calls)."""

import re
import shutil
from pathlib import Path

import pytest
import yaml
from fakes import FakeLLM

from weightspace import experiment as ex
from weightspace.config import ConfigError, load_config
from weightspace.evaluation import FreezeError
from weightspace.utils import read_json, write_json

REPO = Path(__file__).resolve().parents[1]


def make_repo(tmp: Path, extra_stages: dict | None = None) -> Path:
    (tmp / "configs").mkdir()
    for f in ("training.yaml", "interpretation.yaml"):
        shutil.copy(REPO / "configs" / f, tmp / "configs" / f)
    tr = yaml.safe_load((tmp / "configs" / "training.yaml").read_text())
    tr["training"].update(n_train=1500, n_val=400, n_test=800, max_epochs=40, lr=0.01,
                          early_stop_patience=2, accuracy_target=0.85)
    (tmp / "configs" / "training.yaml").write_text(yaml.safe_dump(tr))
    stages = {
        "s1": {"seed_indices": [0], "random_init_indices": [0], "permutations_per_network": 1,
               "arch_only_a": 1, "arch_only_b": 1, "previous": None,
               "counts_toward_study": False},
        "s2": {"seed_indices": [1], "random_init_indices": [1], "permutations_per_network": 1,
               "arch_only_a": 0, "arch_only_b": 0, "previous": "s1",
               "counts_toward_study": False},
        **(extra_stages or {}),
    }
    (tmp / "configs" / "expansion.yaml").write_text(yaml.safe_dump({"stages": stages}))
    (tmp / "configs" / "t.yaml").write_text(yaml.safe_dump({
        "include": ["training.yaml", "interpretation.yaml", "expansion.yaml"],
        "experiment": {"namespace": "t", "stage": "s1",
                       "tasks": ["circle", "diagonal", "square"], "device": "cpu"},
        "paths": {"root": "."}}))
    return tmp / "configs" / "t.yaml"


def oracle(truth):
    def answer(protocol):
        def f(user0, tools):
            m = re.search(r"network_\d{3}", user0)
            task = truth.get(m.group(0), {}).get("task") if m else None
            task = task or "circle"
            if protocol == "B":
                return {"network_id": "x", "predicted_function": task, "confidence": 0.8,
                        "mathematical_explanation": "", "supporting_evidence": [],
                        "uncertainties": []}
            rule = {"circle": "x1**2+x2**2<=0.36", "diagonal": "x2>x1",
                    "square": "abs(x1)<=0.6 and abs(x2)<=0.6"}[task]
            return {"network_id": "x", "behavior_summary": "", "boundary_geometry": "",
                    "symmetries": [], "proposed_rule": rule, "rule_confidence": 0.5,
                    "supporting_evidence": [], "alternative_explanations": [],
                    "uncertainties": []}
        return f
    return answer


def run_fake_stage(cfg, checks_ok=True):
    validated = ex.train_stage(cfg)
    truth = ex.build_units(cfg, validated)
    ex.write_stage_manifest(cfg, truth)
    write_json(cfg.manifest_dir / f"{cfg.stage}_checks.json",
               {"leak_tests_passed": checks_ok, "reproducible": True, "all_validated": True})
    ans = oracle(truth)
    for p in ("A", "B"):
        r = ex.interpret_stage(cfg, p, FakeLLM(ans(p)), {"model": "fake"})
        assert r["complete"]
    ex.freeze_stage(cfg)
    return validated, truth, ex.evaluate_stage(cfg)


def test_pipeline_and_expansion(tmp_path):
    cpath = make_repo(tmp_path)
    cfg = load_config(cpath, "s1")
    validated, truth, m = run_fake_stage(cfg)
    assert len(validated) == 3
    assert m["unit_counts"] == {"original": 3, "permuted": 3, "random_init": 1}
    assert m["B"]["original"]["accuracy"]["k"] == 3
    assert m["A"]["original"]["accuracy"]["k"] == 3
    assert m["pairs"]["B_agreement"]["k"] == 3
    # ids are shared across kinds and parents are linked
    perm = [u for u in truth.values() if u["kind"] == "permuted"]
    assert all(truth[u["parent"]]["kind"] == "original" for u in perm)
    # rerun is idempotent: nothing retrained or reinterpreted
    n_reg = len((cfg.manifest_dir / "registry.jsonl").read_text().splitlines())
    assert ex.train_stage(cfg) and len(
        (cfg.manifest_dir / "registry.jsonl").read_text().splitlines()) == n_reg
    with pytest.raises(RuntimeError):  # frozen predictions are immutable
        ex.interpret_stage(cfg, "A", FakeLLM(oracle(truth)("A")), {})
    # tampering with a frozen prediction is detected
    f = next((cfg.results_dir / "predictions" / "s1" / "B").glob("*.json"))
    rec = read_json(f)
    write_json(f, {**rec, "submission": {**rec["submission"], "confidence": 0.123}})
    with pytest.raises(FreezeError):
        ex.evaluate_stage(cfg)
    write_json(f, rec)
    ex.evaluate_stage(cfg)

    # expansion: new seeds only, earlier records preserved, pooled evaluation
    cfg2 = load_config(cpath, "s2")
    m["expansion"] = {"eligible": True}
    write_json(cfg.results_dir / "metrics" / "s1.json", m)  # tiny stage: force eligibility
    _, truth2, m2 = run_fake_stage(cfg2)
    assert m2["unit_counts"]["original"] == 6
    ids1 = {n for n, u in truth2.items() if u["stage"] == "s1"}
    ids2 = {n for n, u in truth2.items() if u["stage"] == "s2"}
    assert ids1.isdisjoint(ids2) and min(ids2) > max(ids1)
    seeds = [(u["task"], u["seed_index"]) for u in truth2.values() if u["kind"] == "original"]
    assert len(seeds) == len(set(seeds))


def test_expansion_refused_when_not_eligible(tmp_path):
    cpath = make_repo(tmp_path)
    cfg = load_config(cpath, "s1")
    run_fake_stage(cfg)
    m = read_json(cfg.results_dir / "metrics" / "s1.json")
    m["expansion"] = {"eligible": False}
    write_json(cfg.results_dir / "metrics" / "s1.json", m)
    with pytest.raises(RuntimeError, match="criteria not met"):
        ex.check_compatible(load_config(cpath, "s2"))


def test_duplicate_seed_indices_rejected(tmp_path):
    cpath = make_repo(tmp_path, {"s3": {"seed_indices": [0], "random_init_indices": [5],
                                        "permutations_per_network": 1, "previous": "s2"}})
    with pytest.raises(ConfigError, match="seed index 0"):
        load_config(cpath, "s3")


def test_real_config_is_valid():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert cfg.stage == "initial_15" and len(cfg.stage_cfg["seed_indices"]) == 5
    assert cfg.tasks == ["circle", "diagonal", "square"]
    for s in ("expand_30", "expand_60"):
        load_config(REPO / "configs" / "default.yaml", s)
