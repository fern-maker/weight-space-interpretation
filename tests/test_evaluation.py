import numpy as np
import pytest

from weightspace.evaluation import expansion_criteria, score_b, score_rule
from weightspace.expression import RuleError, parse_rule
from weightspace.interpreter import validate_submission
from weightspace.statistics import binom_tail_ge, cohen_kappa, wilson
from weightspace.tasks import FORMULAS, TASKS, grid


def test_parser_matches_task_functions():
    X = grid(101)
    for t, f in FORMULAS.items():
        assert np.array_equal(parse_rule(f).evaluate(X), TASKS[t](X).astype(bool))


def test_parser_accepts_common_notation():
    X = grid(51)
    a = parse_rule("|x1| ≤ 0.6 ∧ |x2| ≤ 0.6").evaluate(X)
    b = parse_rule("max(abs(x1), abs(x2)) <= 0.6").evaluate(X)
    assert np.array_equal(a, b)
    assert parse_rule("x1^2 + x2^2 <= 0.36").evaluate(np.array([[0.0, 0.0]]))[0]
    assert parse_rule("-0.6 <= x1 <= 0.6").evaluate(np.array([[0.7, 0.0]]))[0] == False  # noqa


@pytest.mark.parametrize("bad", ["__import__('os')", "x1.real > 0", "open('f')", "x3 > 0",
                                 "x1 ** x2 > 1", "x1 + 1", "lambda: 1", "[x1 > 0]", "",
                                 "x1 == 0", "x1 ** 100 > 0"])
def test_parser_rejects_unsafe_or_invalid(bad):
    with pytest.raises(RuleError):
        parse_rule(bad)


def test_score_rule_families():
    for t, f in FORMULAS.items():
        s = score_rule(f, None)
        assert s["implied_family"] == t and s["agreement"][t] == 1.0
    s = score_rule("x1**2 + x2**2 <= 0.30", None)
    assert s["implied_family"] == "circle" and s["agreement"]["circle"] > 0.9
    assert score_rule("not a rule", None)["parsed"] is False


def test_score_b_counts():
    truth = {f"n{i}": {"task": t, "seed_index": i} for i, t in
             enumerate(["circle", "circle", "square", "diagonal"])}
    recs = {"n0": {"valid": True, "submission": {"predicted_function": "circle", "confidence": .9}},
            "n1": {"valid": True, "submission": {"predicted_function": "square", "confidence": .6}},
            "n2": {"valid": False, "submission": None},
            "n3": {"valid": True, "submission": {"predicted_function": "diagonal",
                                                  "confidence": 1.0}}}
    out = score_b(recs, truth, sorted(truth))
    assert out["accuracy"]["k"] == 2 and out["accuracy"]["n"] == 4
    assert out["invalid"]["k"] == 1
    assert out["confusion"]["circle"]["square"] == 1
    assert out["confusion"]["square"]["invalid"] == 1
    assert out["per_task"]["circle"]["k"] == 1


def test_submission_validation():
    assert validate_submission("B", {"predicted_function": "hexagon", "confidence": 2},
                               ["circle"]) == ["predicted_function not a candidate",
                                               "confidence not in [0,1]"]
    assert validate_submission("A", {"proposed_rule": "x1>0", "rule_confidence": 0.5}, None) == []


def test_statistics():
    lo, hi = wilson(9, 15)
    assert 0.35 < lo < 0.37 and 0.80 < hi < 0.82
    assert abs(binom_tail_ge(9, 15) - 0.0308) < 1e-3
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0


def _m(acc_k, per_task_k, perm_k, agree_k, n=15):
    per = {t: {"k": k, "n": n // 3} for t, k in zip(["circle", "diagonal", "square"], per_task_k, strict=True)}
    return {"B": {"original": {"accuracy": {"k": acc_k, "n": n, "wilson95": wilson(acc_k, n)},
                               "per_task": per},
                  "permuted": {"accuracy": {"k": perm_k, "n": n}}},
            "pairs": {"B_agreement": {"k": agree_k, "n": n}}}


def test_expansion_criteria_rule():
    ok = {"all_validated": True, "leak_tests_passed": True, "reproducible": True}
    assert expansion_criteria(_m(9, [3, 3, 3], 8, 10), 15, ok)["eligible"]
    assert not expansion_criteria(_m(8, [3, 3, 2], 8, 10), 15, ok)["eligible"]
    r = expansion_criteria(_m(9, [5, 3, 1], 8, 10), 15, ok)
    assert r["criteria"]["4_not_single_task"] is True  # 4/10
    r = expansion_criteria(_m(9, [5, 3, 1], 8, 10), 15, ok)
    assert r["rest"] == {"k": 4, "n": 10}
    r = expansion_criteria(_m(9, [5, 2, 2], 8, 10), 15, ok)
    assert r["criteria"]["4_not_single_task"] is True
    r = expansion_criteria(_m(9, [5, 4, 0], 8, 10), 15, ok)
    assert r["criteria"]["4_not_single_task"] is True  # 4/10 still >= 4
    r = expansion_criteria(_m(9, [5, 1, 3], 8, 10), 15, ok)
    assert r["criteria"]["4_not_single_task"] is True
    r = expansion_criteria(_m(8, [5, 3, 0], 8, 10), 15, ok)
    assert r["criteria"]["4_not_single_task"] is False  # 3/10
    r = expansion_criteria(_m(9, [5, 2, 2], 7, 10), 15, ok)
    assert not r["criteria"]["5_permutation_no_collapse"]
    assert not expansion_criteria(_m(9, [3, 3, 3], 8, 10), 15,
                                  {**ok, "reproducible": False})["eligible"]
    big = expansion_criteria(_m(18, [6, 6, 6], 16, 20, n=30), 30, ok)
    assert big["eligible"] and "6_wilson_lower_bound_above_chance" in big["criteria"]
