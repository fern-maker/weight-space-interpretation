"""Safe parser/evaluator for open-ended boolean rules over x1, x2.

Only a whitelisted subset of Python expression syntax is accepted; nothing is
passed to eval(). Used by the private evaluator to score Protocol A answers.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

import numpy as np

MAX_LEN = 500
MAX_EXPONENT = 8

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCS = {"abs": (1, np.abs), "sqrt": (1, np.sqrt), "min": (2, np.minimum), "max": (2, np.maximum)}
_VARS = {"x1": 0, "x2": 1}


class RuleError(ValueError):
    pass


def _normalize(text: str) -> str:
    t = text.strip().strip("`").strip()
    for a, b in (("≤", "<="), ("≥", ">="), ("^", "**"), ("&&", " and "), ("||", " or "),
                 ("|x1|", "abs(x1)"), ("|x2|", "abs(x2)"), ("∧", " and "), ("∨", " or ")):
        t = t.replace(a, b)
    return t


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        return _check(node.body)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        for v in node.values:
            _check(v)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
        return _check(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        if isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                    and abs(node.right.value) <= MAX_EXPONENT):
                raise RuleError("exponent must be a numeric constant <= 8")
        _check(node.left)
        _check(node.right)
        return
    if isinstance(node, ast.Compare):
        if not all(type(op) in _CMPOPS for op in node.ops):
            raise RuleError("only < <= > >= comparisons allowed")
        _check(node.left)
        for c in node.comparators:
            _check(c)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise RuleError("only abs/sqrt/min/max calls allowed")
        if len(node.args) != _FUNCS[node.func.id][0]:
            raise RuleError(f"{node.func.id} takes {_FUNCS[node.func.id][0]} args")
        for a in node.args:
            _check(a)
        return
    if isinstance(node, ast.Name):
        if node.id not in _VARS:
            raise RuleError(f"unknown name {node.id!r}")
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return
    raise RuleError(f"disallowed syntax: {type(node).__name__}")


def _eval(node: ast.AST, X: np.ndarray):
    if isinstance(node, ast.Expression):
        return _eval(node.body, X)
    if isinstance(node, ast.BoolOp):
        vals = [np.asarray(_eval(v, X), dtype=bool) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, X)
        if isinstance(node.op, ast.Not):
            return ~np.asarray(v, dtype=bool)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, X), _eval(node.right, X))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, X)
        out = np.ones(len(X), dtype=bool)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval(comp, X)
            out &= np.asarray(_CMPOPS[type(op)](left, right), dtype=bool)
            left = right
        return out
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id][1](*[_eval(a, X) for a in node.args])
    if isinstance(node, ast.Name):
        return X[:, _VARS[node.id]]
    return float(node.value)


@dataclass
class Rule:
    text: str
    tree: ast.Expression

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out = _eval(self.tree, np.asarray(X, dtype=np.float64))
        out = np.broadcast_to(np.asarray(out), (len(X),))
        if out.dtype != bool:
            raise RuleError("rule does not evaluate to a boolean region")
        return out.copy()


def parse_rule(text: str) -> Rule:
    if not isinstance(text, str) or not text.strip():
        raise RuleError("empty rule")
    if len(text) > MAX_LEN:
        raise RuleError("rule too long")
    norm = _normalize(text)
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError as e:
        raise RuleError(f"syntax error: {e.msg}") from e
    _check(tree)
    rule = Rule(text=norm, tree=tree)
    rule.evaluate(np.zeros((2, 2)))  # must produce booleans
    return rule
