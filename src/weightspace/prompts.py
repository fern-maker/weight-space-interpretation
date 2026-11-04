"""Prompt templates. Versioned; Protocol A text is leak-scanned in tests."""

from __future__ import annotations

from .tasks import FORMULAS
from .weight_exporter import export_text

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """\
You analyse the parameters of a small feed-forward neural network that maps a point \
(x1, x2) to a binary class. Your goal is to work out, from the numerical parameters and \
architecture alone, what function the network computes.

Rules for this analysis:
- You have no examples of inputs, outputs or classes, and you cannot run the network.
- Reason analytically from parameter structure: first-layer lines w.x + b = 0, signs and \
magnitudes, how hidden units combine, symmetries, norms, singular values, matrix products.
- Do not simulate the full network (all three layers) on chosen input coordinates as a \
substitute for structural analysis.
- The analysis tools only compute quantities from the parameters. Use them as needed.
- When finished, call submit_interpretation exactly once. That ends the session.
- Be honest about uncertainty. If the parameters do not show clear structure, say so."""

_A_TASK = """\
Task (open-ended): describe the function this network computes. In your submission give:
1. the likely mathematical behaviour of the network,
2. the approximate geometry of the decision boundary in the input domain,
3. any symmetries you infer,
4. an approximate rule, written as a boolean expression over x1 and x2 that is true on the \
class-1 region (proposed_rule),
5. the evidence from the parameters supporting your interpretation,
6. your uncertainty and alternative explanations."""

_B_TASK = """\
Task (multiple choice): exactly one of the following candidate functions best describes \
what this network was fitted to compute (class 1 where the condition holds). The list does \
not imply that every candidate occurs equally often across networks.

{candidates}

Select the most likely candidate in predicted_function, give a confidence between 0 and 1, \
and explain the mathematical evidence from the parameters."""

_ARCH_ONLY = """\
Only the architecture is available; the parameter values are withheld.

Architecture:
  layer1: Linear(in=2, out=16) -> tanh -> layer2: Linear(in=16, out=16) -> tanh -> \
layer3: Linear(in=16, out=1)
  forward: h1 = tanh(W1 x + b1); h2 = tanh(W2 h1 + b2); z = W3 h2 + b3
  input domain: x1 in [-1, 1], x2 in [-1, 1]
  output: class 1 if z > 0 else 0
Network id: {network_id}"""


def candidate_block(order: list[str]) -> str:
    return "\n".join(f"- {name}: {FORMULAS[name]}" for name in order)


def _weights_block(export: dict) -> str:
    return (
        f"Network id: {export['network_id']}\n"
        "Complete network specification and parameters (JSON):\n"
        f"{export_text(export)}"
    )


def build_user_message(
    protocol: str,
    condition: str,
    export: dict | None = None,
    network_id: str | None = None,
    candidate_order: list[str] | None = None,
) -> str:
    if condition == "weights_tools":
        assert export is not None
        head = _weights_block(export)
    elif condition == "architecture_only":
        head = _ARCH_ONLY.format(network_id=network_id)
    else:
        raise ValueError(condition)
    if protocol == "A":
        task = _A_TASK
    elif protocol == "B":
        assert candidate_order
        task = _B_TASK.format(candidates=candidate_block(candidate_order))
    else:
        raise ValueError(protocol)
    return f"{head}\n\n{task}"
