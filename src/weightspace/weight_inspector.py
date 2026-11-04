"""Analytic, read-only inspection of MLP parameters.

Every function accepts either a torch ``nn.Sequential`` or an exported
parameter dict (``export["parameters"]``), so the isolated tool server can use
this module without importing torch. No function here evaluates the network
on input points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

LAYERS = ["layer1", "layer2", "layer3"]
HIDDEN_LAYERS = ["layer1", "layer2"]


def _params(model_or_params: Any) -> dict[str, dict[str, np.ndarray]]:
    if isinstance(model_or_params, dict):
        src = model_or_params.get("parameters", model_or_params)
        return {
            name: {
                "weight": np.asarray(src[name]["weight"], dtype=np.float64),
                "bias": np.asarray(src[name]["bias"], dtype=np.float64),
            }
            for name in LAYERS
        }
    linears = [m for m in model_or_params if hasattr(m, "weight")]
    return {
        name: {
            "weight": lin.weight.detach().cpu().double().numpy(),
            "bias": lin.bias.detach().cpu().double().numpy(),
        }
        for name, lin in zip(LAYERS, linears, strict=True)
    }


def _check_layer(layer: str) -> None:
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; valid: {LAYERS}")


def inspect_architecture(model: Any) -> dict:
    p = _params(model)
    layers = []
    for i, name in enumerate(LAYERS):
        w = p[name]["weight"]
        layers.append(
            {
                "name": name,
                "type": "Linear",
                "in_features": int(w.shape[1]),
                "out_features": int(w.shape[0]),
                "weight_shape": list(w.shape),
                "bias_shape": list(p[name]["bias"].shape),
                "activation_after": "tanh" if i < 2 else "none (logit)",
            }
        )
    return {
        "layers": layers,
        "total_parameters": int(sum(v["weight"].size + v["bias"].size for v in p.values())),
        "forward_equations": [
            "h1 = tanh(W1 @ x + b1)",
            "h2 = tanh(W2 @ h1 + b2)",
            "z = W3 @ h2 + b3",
        ],
        "output_rule": "class 1 iff z > 0 (equivalently sigmoid(z) > 0.5)",
    }


def inspect_weight_matrix(model: Any, layer: str) -> list[list[float]]:
    _check_layer(layer)
    return _params(model)[layer]["weight"].tolist()


def inspect_bias_vector(model: Any, layer: str) -> list[float]:
    _check_layer(layer)
    return _params(model)[layer]["bias"].tolist()


def inspect_neuron(model: Any, layer: str, index: int) -> dict:
    """Incoming weights, bias and outgoing weights of one unit."""
    _check_layer(layer)
    p = _params(model)
    w = p[layer]["weight"]
    if not 0 <= index < w.shape[0]:
        raise ValueError(f"index out of range 0..{w.shape[0] - 1}")
    out: dict[str, Any] = {
        "layer": layer,
        "index": index,
        "incoming_weights": w[index].tolist(),
        "bias": float(p[layer]["bias"][index]),
        "incoming_weight_norm": float(np.linalg.norm(w[index])),
    }
    nxt_i = LAYERS.index(layer) + 1
    if nxt_i < len(LAYERS):
        nxt = p[LAYERS[nxt_i]]["weight"][:, index]
        out["outgoing_weights"] = nxt.tolist()
        out["outgoing_layer"] = LAYERS[nxt_i]
    return out


def _stats(a: np.ndarray) -> dict:
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "l2_norm": float(np.linalg.norm(a)),
        "abs_mean": float(np.abs(a).mean()),
    }


def compute_weight_statistics(model: Any, layer: str | None = None) -> dict:
    p = _params(model)
    names = [layer] if layer else LAYERS
    out = {}
    for name in names:
        _check_layer(name)
        w = p[name]["weight"]
        out[name] = {
            "weight": _stats(w),
            "bias": _stats(p[name]["bias"]),
            "row_norms": np.linalg.norm(w, axis=1).tolist(),
            "column_norms": np.linalg.norm(w, axis=0).tolist(),
        }
    return out


def compute_singular_values(model: Any, layer: str) -> list[float]:
    _check_layer(layer)
    return np.linalg.svd(_params(model)[layer]["weight"], compute_uv=False).tolist()


def matrix_product(model: Any, left_layer: str, right_layer: str) -> list[list[float]]:
    """Product W_left @ W_right of two weight matrices (shapes must chain)."""
    _check_layer(left_layer)
    _check_layer(right_layer)
    p = _params(model)
    a, b = p[left_layer]["weight"], p[right_layer]["weight"]
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes do not chain: {a.shape} @ {b.shape}")
    return (a @ b).tolist()


def first_layer_hyperplanes(model: Any) -> list[dict]:
    """Each first-layer unit's zero line w.x + b = 0, in closed form."""
    p = _params(model)["layer1"]
    out = []
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    for i, (w, b) in enumerate(zip(p["weight"], p["bias"], strict=True)):
        norm = float(np.linalg.norm(w))
        vals = corners @ w + b
        out.append(
            {
                "index": i,
                "weight": w.tolist(),
                "bias": float(b),
                "weight_norm": norm,
                "unit_normal": (w / norm).tolist() if norm > 0 else [0.0, 0.0],
                "normal_angle_deg": float(np.degrees(np.arctan2(w[1], w[0]))),
                "signed_offset_from_origin": float(-b / norm) if norm > 0 else None,
                "line_crosses_domain": bool(vals.min() < 0 < vals.max()),
            }
        )
    return out


def export_weights(model: Any, output_path: str | Path, network_id: str = "network_000") -> dict:
    """Thin wrapper kept for the documented API; see weight_exporter."""
    from .weight_exporter import model_to_export, write_export

    exp = model_to_export(model, network_id)
    write_export(exp, output_path)
    return exp
