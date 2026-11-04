"""Control conditions (see METHODOLOGY §5.5).

- chance: analytic 1/3 for the 3-way multiple choice;
- architecture-only: sessions with condition "architecture_only" (no parameters,
  no analysis tools), scheduled by experiment.interpret_stage;
- random-init: untrained networks built by training.random_init_network and
  mixed blind into the unit pool by experiment.build_units.
"""

from __future__ import annotations

from .statistics import binom_tail_ge

CHANCE_B = 1 / 3


def chance_summary(k: int, n: int) -> dict:
    """Expected correct under guessing and the tail probability of >= k."""
    return {"chance_rate": CHANCE_B, "expected_correct": n * CHANCE_B,
            "p_ge_observed": binom_tail_ge(k, n, CHANCE_B)}
