"""Model -> canonical, anonymized, LLM-facing JSON. Allowlisted keys only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import canonical_json, round_sig, sha256_json
from .weight_inspector import LAYERS, _params

FORMAT_VERSION = "1.0"
EXPORT_KEYS = {
    "network_id",
    "format_version",
    "architecture",
    "activations",
    "forward_equations",
    "input_domain",
    "output_rule",
    "precision",
    "parameters",
    "parameters_sha256",
}
PARAM_KEYS = {"weight", "bias", "weight_shape", "bias_shape"}


def _round(obj: Any, digits: int) -> Any:
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return round_sig(obj, digits)


def model_to_export(model: Any, network_id: str, sig_digits: int = 6) -> dict:
    p = _params(model)
    params = {}
    for name in LAYERS:
        w, b = p[name]["weight"], p[name]["bias"]
        params[name] = {
            "weight": _round(w.tolist(), sig_digits),
            "bias": _round(b.tolist(), sig_digits),
            "weight_shape": list(w.shape),
            "bias_shape": list(b.shape),
        }
    return {
        "network_id": network_id,
        "format_version": FORMAT_VERSION,
        "architecture": [
            "layer1: Linear(in=2, out=16)",
            "tanh",
            "layer2: Linear(in=16, out=16)",
            "tanh",
            "layer3: Linear(in=16, out=1)",
        ],
        "activations": {"tanh": "tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))"},
        "forward_equations": [
            "x = [x1, x2]",
            "h1 = tanh(W1 @ x + b1)    W1 = layer1.weight (16x2), b1 = layer1.bias (16)",
            "h2 = tanh(W2 @ h1 + b2)   W2 = layer2.weight (16x16), b2 = layer2.bias (16)",
            "z  = W3 @ h2 + b3         W3 = layer3.weight (1x16), b3 = layer3.bias (1)",
            "Row i of a weight matrix holds the incoming weights of output unit i.",
        ],
        "input_domain": "x1 in [-1, 1], x2 in [-1, 1]",
        "output_rule": "predicted class = 1 if z > 0 else 0 (sigmoid(z) > 0.5)",
        "precision": f"float32 parameters rounded to {sig_digits} significant digits",
        "parameters": params,
        "parameters_sha256": sha256_json(params),
    }


def validate_export(exp: dict) -> None:
    if set(exp) != EXPORT_KEYS:
        raise ValueError(f"export keys not allowlisted: {set(exp) ^ EXPORT_KEYS}")
    if set(exp["parameters"]) != set(LAYERS):
        raise ValueError("unexpected parameter layers")
    for v in exp["parameters"].values():
        if set(v) != PARAM_KEYS:
            raise ValueError("unexpected parameter keys")
    if sha256_json(exp["parameters"]) != exp["parameters_sha256"]:
        raise ValueError("parameter hash mismatch")


def export_text(exp: dict) -> str:
    """Deterministic text form used in prompts and on disk."""
    return canonical_json(exp)


def write_export(exp: dict, path: str | Path) -> None:
    validate_export(exp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_text(exp) + "\n")
