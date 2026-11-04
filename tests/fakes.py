"""Scripted stand-in for the LLM (tests only)."""

from weightspace.llm_client import TurnResult


class FakeLLM:
    """Calls the given tools in order, then submits `answer`."""

    def __init__(self, answer_fn, tool_calls=(("get_architecture", {}),)):
        self.answer_fn = answer_fn
        self.tool_calls = list(tool_calls)
        self.seen_messages = []

    def __call__(self, system, messages, tools):
        self.seen_messages.append([m for m in messages])
        n_assist = sum(1 for m in messages if m["role"] == "assistant")
        if n_assist < len(self.tool_calls):
            name, args = self.tool_calls[n_assist]
            block = {"type": "tool_use", "id": f"t{n_assist}", "name": name, "input": args}
            stop = "tool_use"
        else:
            user0 = messages[0]["content"]
            block = {"type": "tool_use", "id": "sub", "name": "submit_interpretation",
                     "input": self.answer_fn(user0, tools)}
            stop = "tool_use"
        return TurnResult(content=[block], content_dicts=[block], stop_reason=stop,
                          model_served="fake-model", request_id=None,
                          usage={"input_tokens": 100, "output_tokens": 10}, cost_usd=0.001,
                          latency_s=0.0, attempts=1)


def answer_b(choice="circle", conf=0.7):
    def f(user0, tools):
        return {"network_id": "x", "predicted_function": choice, "confidence": conf,
                "mathematical_explanation": "m", "supporting_evidence": ["e"],
                "uncertainties": ["u"]}
    return f


def answer_a(rule="x1**2 + x2**2 <= 0.36", conf=0.6):
    def f(user0, tools):
        return {"network_id": "x", "behavior_summary": "s", "boundary_geometry": "g",
                "symmetries": [], "proposed_rule": rule, "rule_confidence": conf,
                "supporting_evidence": [], "alternative_explanations": [], "uncertainties": []}
    return f
