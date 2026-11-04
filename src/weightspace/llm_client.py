"""Anthropic API wrapper: one streamed turn, retries, usage and cost accounting.

Credentials come only from the environment / SDK credential chain; nothing is
hardcoded or logged. The model under test is fixed by config; refusal
fallbacks to other models are disabled by default so that every recorded
answer comes from the configured model (the served model is recorded).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

_SECRET_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: str) -> str:
    return _SECRET_RE.sub("sk-ant-<redacted>", text)


def has_credentials() -> bool:
    """True only if an explicit credential is in the environment (or .env)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


@dataclass
class TurnResult:
    content: list[Any]          # SDK content blocks (passed back unchanged)
    content_dicts: list[dict]   # serialisable copy for logs
    stop_reason: str | None
    model_served: str
    request_id: str | None
    usage: dict
    cost_usd: float
    latency_s: float
    attempts: int


class TurnFn(Protocol):
    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> TurnResult: ...


class LLMClient:
    def __init__(self, icfg: dict):
        import anthropic

        self._anthropic = anthropic
        self.cfg = icfg
        self.client = anthropic.Anthropic(timeout=float(icfg["request_timeout_s"]), max_retries=3)

    def request_params(self) -> dict:
        """Inference parameters recorded with every session."""
        return {
            "model": self.cfg["model"],
            "max_tokens": self.cfg["max_tokens"],
            "thinking": {"type": "adaptive", "display": self.cfg["thinking_display"]},
            "output_config": {"effort": self.cfg["effort"]},
            "cache_control": {"type": "ephemeral"},
            "tool_choice": {"type": "auto"},
        }

    def cost(self, usage: dict) -> float:
        c = self.cfg
        return (
            usage.get("input_tokens", 0) * c["price_input_per_mtok"]
            + usage.get("output_tokens", 0) * c["price_output_per_mtok"]
            + (usage.get("cache_read_input_tokens") or 0) * c["price_cache_read_per_mtok"]
            + (usage.get("cache_creation_input_tokens") or 0) * c["price_cache_write_per_mtok"]
        ) / 1e6

    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> TurnResult:
        a = self._anthropic
        params = self.request_params()
        attempts = 0
        last_err: Exception | None = None
        while attempts <= int(self.cfg["outer_retries"]):
            attempts += 1
            t0 = time.perf_counter()
            try:
                with self.client.messages.stream(
                    system=system, messages=messages, tools=tools, **params
                ) as stream:
                    msg = stream.get_final_message()
                usage = msg.usage.model_dump() if msg.usage else {}
                return TurnResult(
                    content=list(msg.content),
                    content_dicts=[b.model_dump() for b in msg.content],
                    stop_reason=msg.stop_reason,
                    model_served=msg.model,
                    request_id=getattr(msg, "_request_id", None),
                    usage=usage,
                    cost_usd=self.cost(usage),
                    latency_s=round(time.perf_counter() - t0, 2),
                    attempts=attempts,
                )
            except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError,
                    a.NotFoundError):
                raise  # not retryable
            except (a.RateLimitError, a.InternalServerError, a.APIConnectionError,
                    a.APIStatusError) as e:
                if isinstance(e, a.APIStatusError) and e.status_code < 500 and \
                        not isinstance(e, a.RateLimitError):
                    raise
                last_err = e
                time.sleep(min(60.0, 5.0 * 2 ** (attempts - 1)))
        raise RuntimeError(redact(f"API failed after {attempts} attempts: {last_err}"))
