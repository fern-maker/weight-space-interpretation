"""The three binary classification tasks and dataset sampling (pure numpy)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DOMAIN = (-1.0, 1.0)


def circle(X: np.ndarray) -> np.ndarray:
    return (X[:, 0] ** 2 + X[:, 1] ** 2 <= 0.36).astype(np.float32)


def diagonal(X: np.ndarray) -> np.ndarray:
    return (X[:, 1] > X[:, 0]).astype(np.float32)


def square(X: np.ndarray) -> np.ndarray:
    return ((np.abs(X[:, 0]) <= 0.6) & (np.abs(X[:, 1]) <= 0.6)).astype(np.float32)


TASKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "circle": circle,
    "diagonal": diagonal,
    "square": square,
}

FORMULAS = {
    "circle": "x1**2 + x2**2 <= 0.36",
    "diagonal": "x2 > x1",
    "square": "abs(x1) <= 0.6 and abs(x2) <= 0.6",
}


def sample_inputs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(DOMAIN[0], DOMAIN[1], size=(n, 2)).astype(np.float32)


def sample_dataset(task: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X = sample_inputs(n, seed)
    return X, TASKS[task](X)


def class_balance(y: np.ndarray) -> float:
    return float(np.mean(y))


def grid(resolution: int = 201) -> np.ndarray:
    """Regular evaluation grid over the domain (private evaluator only)."""
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution, dtype=np.float64)
    g1, g2 = np.meshgrid(xs, xs, indexing="xy")
    return np.stack([g1.ravel(), g2.ravel()], axis=1)
