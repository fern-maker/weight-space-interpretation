"""The single MLP architecture used by every network."""

from __future__ import annotations

import torch
from torch import nn

ARCH_SPEC = [
    {"name": "layer1", "type": "Linear", "in": 2, "out": 16},
    {"name": "act1", "type": "Tanh"},
    {"name": "layer2", "type": "Linear", "in": 16, "out": 16},
    {"name": "act2", "type": "Tanh"},
    {"name": "layer3", "type": "Linear", "in": 16, "out": 1},
]
LINEAR_INDICES = {"layer1": 0, "layer2": 2, "layer3": 4}
N_PARAMS = 2 * 16 + 16 + 16 * 16 + 16 + 16 + 1  # 337


def build_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    )


def assert_architecture(model: nn.Module) -> None:
    ref = build_mlp()
    if not isinstance(model, nn.Sequential) or len(model) != len(ref):
        raise ValueError("model is not the reference Sequential MLP")
    for a, b in zip(model, ref, strict=True):
        if type(a) is not type(b):
            raise ValueError(f"layer type mismatch: {type(a)} vs {type(b)}")
        if isinstance(a, nn.Linear) and (a.in_features, a.out_features) != (
            b.in_features,
            b.out_features,
        ):
            raise ValueError("linear layer shape mismatch")
    if sum(p.numel() for p in model.parameters()) != N_PARAMS:
        raise ValueError("parameter count mismatch")


def logits(model: nn.Module, X) -> torch.Tensor:
    """Private forward pass (training/evaluation/verification only)."""
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1)
