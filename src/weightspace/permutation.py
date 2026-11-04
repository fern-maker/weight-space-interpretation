"""Function-preserving hidden-neuron permutations (and sign flips)."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn



def _linears(model: nn.Sequential) -> list[nn.Linear]:
    return [m for m in model if isinstance(m, nn.Linear)]


def random_permutations(model: nn.Sequential, seed: int) -> list[np.ndarray]:
    """One non-identity permutation per hidden layer."""
    rng = np.random.default_rng(seed)
    perms = []
    for lin in _linears(model)[:-1]:
        n = lin.out_features
        while True:
            p = rng.permutation(n)
            if not np.array_equal(p, np.arange(n)):
                break
        perms.append(p)
    return perms


def permute_hidden(model: nn.Sequential, perms: list[np.ndarray]) -> nn.Sequential:
    """Return a copy where hidden layer k's units are reordered by perms[k].

    new unit j = old unit perms[k][j]. Rows of the layer producing the hidden
    units and the matching bias entries are permuted, as are the columns of the
    next layer's weight matrix.
    """
    new = copy.deepcopy(model)
    lins = _linears(new)
    if len(perms) != len(lins) - 1:
        raise ValueError("need one permutation per hidden layer")
    with torch.no_grad():
        for k, p in enumerate(perms):
            idx = torch.as_tensor(p, dtype=torch.long)
            lins[k].weight.copy_(lins[k].weight[idx])
            lins[k].bias.copy_(lins[k].bias[idx])
            lins[k + 1].weight.copy_(lins[k + 1].weight[:, idx])
    return new


def sign_flip(model: nn.Sequential, layer_k: int, units: list[int]) -> nn.Sequential:
    """tanh is odd: negating a unit's incoming weights, bias and outgoing weights
    preserves the function. Optional extra robustness transform."""
    new = copy.deepcopy(model)
    lins = _linears(new)
    with torch.no_grad():
        for u in units:
            lins[layer_k].weight[u] *= -1
            lins[layer_k].bias[u] *= -1
            lins[layer_k + 1].weight[:, u] *= -1
    return new


def verify_equivalence(
    a: nn.Sequential, b: nn.Sequential, X: np.ndarray, atol: float = 1e-5
) -> dict:
    """Private check: same logits within tolerance and same predicted classes.

    Computed in float64 so float32 summation-order rounding (which reordering
    hidden units changes) does not mask or mimic a real difference.
    """
    Xd = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        la = copy.deepcopy(a).double()(Xd).squeeze(-1)
        lb = copy.deepcopy(b).double()(Xd).squeeze(-1)
    max_diff = float((la - lb).abs().max())
    same_class = bool(torch.equal(la > 0, lb > 0))
    return {"max_abs_logit_diff": max_diff, "same_classes": same_class,
            "equivalent": max_diff <= atol and same_class}
