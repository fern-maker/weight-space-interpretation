"""Small-sample statistics: Wilson intervals, binomial tails, calibration, kappa."""

from __future__ import annotations

from math import comb, sqrt


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_tail_ge(k: int, n: int, p: float = 1 / 3) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def proportion(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def brier(confidences: list[float], correct: list[bool]) -> float | None:
    if not confidences:
        return None
    return sum((c - float(o)) ** 2 for c, o in zip(confidences, correct, strict=True)) / len(
        confidences
    )


def reliability(confidences: list[float], correct: list[bool], bins: int = 5) -> list[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1)]
        if idx:
            out.append({"bin": [lo, hi], "n": len(idx),
                        "mean_conf": sum(confidences[i] for i in idx) / len(idx),
                        "accuracy": sum(correct[i] for i in idx) / len(idx)})
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)
