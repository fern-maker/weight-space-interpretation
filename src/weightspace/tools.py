"""Tool schemas and pure implementations for the interpreter.

Every analysis tool operates only on the in-memory export dict. No tool takes
input-space points, file paths, or code, so forward passes cannot be
requested. Imports: numpy + weight_inspector only (no torch).
"""

from __future__ import annotations

import json
from typing import Any

from . import weight_inspector as wi
from .utils import round_sig, sha256_json

LAYER_ENUM = {"type": "string", "enum": wi.LAYERS}
HIDDEN_LAYER_ENUM = {"type": "string", "enum": wi.HIDDEN_LAYERS}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


ANALYSIS_TOOLS: list[dict] = [
    {
        "name": "get_architecture",
        "description": "Layer list, shapes, activations, forward equations and output rule.",
        "input_schema": _schema({}),
    },
    {
        "name": "get_weight_matrix",
        "description": "Full weight matrix of a layer (row i = incoming weights of unit i).",
        "input_schema": _schema({"layer": LAYER_ENUM}),
    },
    {
        "name": "get_bias_vector",
        "description": "Bias vector of a layer.",
        "input_schema": _schema({"layer": LAYER_ENUM}),
    },
    {
        "name": "get_neuron",
        "description": "Incoming weights, bias and outgoing weights of one unit.",
        "input_schema": _schema(
            {"layer": LAYER_ENUM, "index": {"type": "integer", "description": "unit index 0-15 (layer3 has only index 0)"}}
        ),
    },
    {
        "name": "weight_statistics",
        "description": "Mean/std/min/max/norms of a layer's weights and bias, plus row and "
        "column norms.",
        "input_schema": _schema({"layer": LAYER_ENUM}),
    },
    {
        "name": "singular_values",
        "description": "Singular values of a layer's weight matrix.",
        "input_schema": _schema({"layer": LAYER_ENUM}),
    },
    {
        "name": "matrix_product",
        "description": "Product left_layer.weight @ right_layer.weight of two weight matrices "
        "(e.g. layer3 @ layer2, layer2 @ layer1). Shapes must chain.",
        "input_schema": _schema({"left_layer": LAYER_ENUM, "right_layer": LAYER_ENUM}),
    },
    {
        "name": "first_layer_hyperplanes",
        "description": "For each layer1 unit, its zero line w.x + b = 0 in input space: unit "
        "normal, normal angle, signed offset from the origin, and whether the line crosses "
        "the input domain. Closed-form, computed from parameters only.",
        "input_schema": _schema({}),
    },
]


def submit_tool(protocol: str, candidates: list[str] | None = None) -> dict:
    str_list = {"type": "array", "items": {"type": "string"}}
    conf = {"type": "number", "description": "probability between 0 and 1"}
    if protocol == "A":
        props = {
            "network_id": {"type": "string"},
            "behavior_summary": {"type": "string"},
            "boundary_geometry": {"type": "string"},
            "symmetries": str_list,
            "proposed_rule": {
                "type": "string",
                "description": "Boolean expression over x1, x2 that is true exactly on the "
                "class-1 region. Allowed: numbers, x1, x2, + - * / **, abs(), sqrt(), min(a,b), "
                "max(a,b), < <= > >=, and, or, not.",
            },
            "rule_confidence": conf,
            "supporting_evidence": str_list,
            "alternative_explanations": str_list,
            "uncertainties": str_list,
        }
    elif protocol == "B":
        assert candidates
        props = {
            "network_id": {"type": "string"},
            "predicted_function": {"type": "string", "enum": list(candidates)},
            "confidence": conf,
            "mathematical_explanation": {"type": "string"},
            "supporting_evidence": str_list,
            "uncertainties": str_list,
        }
    else:
        raise ValueError(protocol)
    return {
        "name": "submit_interpretation",
        "description": "Submit your final interpretation. Call exactly once; this ends the "
        "session.",
        "input_schema": _schema(props),
    }


def tool_set(protocol: str, condition: str, candidates: list[str] | None = None) -> list[dict]:
    tools = ANALYSIS_TOOLS if condition == "weights_tools" else []
    return [{**t, "strict": True} for t in [*tools, submit_tool(protocol, candidates)]]


def tool_schema_hash(tools: list[dict]) -> str:
    return sha256_json(tools)


ANALYSIS_NAMES = {t["name"] for t in ANALYSIS_TOOLS}


def _compact(obj: Any) -> Any:
    if isinstance(obj, float):
        return round_sig(obj, 6)
    if isinstance(obj, list):
        return [_compact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    return obj


def run_tool(export: dict, name: str, args: dict) -> str:
    """Execute an analysis tool on the export; returns a JSON string."""
    if name not in ANALYSIS_NAMES:
        raise ValueError(f"tool not allowed: {name}")
    args = args or {}
    allowed = set(next(t for t in ANALYSIS_TOOLS if t["name"] == name)["input_schema"]
                  ["properties"])
    if set(args) - allowed:
        raise ValueError(f"unexpected arguments: {sorted(set(args) - allowed)}")
    if name == "get_architecture":
        out: Any = wi.inspect_architecture(export)
    elif name == "get_weight_matrix":
        out = wi.inspect_weight_matrix(export, args["layer"])
    elif name == "get_bias_vector":
        out = wi.inspect_bias_vector(export, args["layer"])
    elif name == "get_neuron":
        out = wi.inspect_neuron(export, args["layer"], int(args["index"]))
    elif name == "weight_statistics":
        out = wi.compute_weight_statistics(export, args["layer"])
    elif name == "singular_values":
        out = wi.compute_singular_values(export, args["layer"])
    elif name == "matrix_product":
        out = wi.matrix_product(export, args["left_layer"], args["right_layer"])
    elif name == "first_layer_hyperplanes":
        out = wi.first_layer_hyperplanes(export)
    else:  # pragma: no cover
        raise ValueError(name)
    return json.dumps(_compact(out), separators=(",", ":"))
