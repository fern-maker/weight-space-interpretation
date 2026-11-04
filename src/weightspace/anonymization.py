"""Opaque IDs, the private id->truth map, and leak scanning."""

from __future__ import annotations

import re

import numpy as np

# Words that must never reach the interpreter in strict / Protocol A contexts.
TASK_WORDS = ["circle", "circular", "diagonal", "square", "disk", "disc", "radius"]
META_WORDS = [
    "seed", "checkpoint", "ground_truth", "ground truth", "artifacts", "private",
    "permut", "random_init", "random-init", "untrained", "manifest", "accuracy",
    "label", "training data", "dataset", ".pt", "task_",
]
THRESHOLD_CONSTANTS = ["0.36", "0.6"]

_word_res = {w: re.compile(rf"(?<![a-z]){re.escape(w)}", re.IGNORECASE) for w in
             TASK_WORDS + META_WORDS}
_const_res = {c: re.compile(rf"(?<![\d.]){re.escape(c)}(?!\d)") for c in THRESHOLD_CONSTANTS}


def scan_for_leaks(
    text: str,
    allow: set[str] | None = None,
    check_constants: bool = False,
    extra_terms: list[str] | None = None,
) -> list[str]:
    """Return forbidden terms found in `text` (empty list = clean).

    `allow` lists terms permitted in this context (e.g. candidate labels in
    Protocol B). Threshold constants are only checked in prompt templates,
    since parameter values may legitimately contain those digits.
    """
    allow = {a.lower() for a in (allow or set())}
    hits = [w for w, r in _word_res.items() if w.lower() not in allow and r.search(text)]
    if check_constants:
        hits += [c for c, r in _const_res.items() if c not in allow and r.search(text)]
    for t in extra_terms or []:
        if t and t in text:
            hits.append(t)
    return hits


def allocate_ids(units: list[dict], seed: int, start: int = 1) -> list[dict]:
    """Shuffle all units together and assign network_XXX ids in shuffled order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    out = []
    for rank, i in enumerate(order):
        u = dict(units[int(i)])
        u["network_id"] = f"network_{start + rank:03d}"
        out.append(u)
    return sorted(out, key=lambda u: u["network_id"])
