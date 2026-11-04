"""One isolated interpretation session = one fresh conversation + tool loop."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .anonymization import scan_for_leaks
from .llm_client import TurnFn
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from .tool_server import ToolServer
from .tools import tool_schema_hash, tool_set
from .utils import sha256_bytes
from .weight_exporter import export_text

NUDGE = "Please finish now by calling submit_interpretation exactly once."


class LeakError(RuntimeError):
    pass


def _get(block: Any, attr: str) -> Any:
    return block.get(attr) if isinstance(block, dict) else getattr(block, attr, None)


def check_prompt(protocol: str, user_msg: str, export: dict | None,
                 candidates: list[str] | None) -> None:
    """Leak-scan the prompt: template text strictly, export text for words."""
    template = user_msg
    if export is not None:
        etxt = export_text(export)
        template = user_msg.replace(etxt, "")
        hits = scan_for_leaks(etxt)
        if hits:
            raise LeakError(f"export contains forbidden terms: {hits}")
    allow = set(candidates or []) | ({"0.36", "0.6"} if protocol == "B" else set())
    hits = scan_for_leaks(SYSTEM_PROMPT + "\n" + template, allow=allow, check_constants=True)
    if hits:
        raise LeakError(f"prompt contains forbidden terms: {hits}")


def validate_submission(protocol: str, sub: dict, candidates: list[str] | None) -> list[str]:
    problems = []
    if protocol == "B":
        if sub.get("predicted_function") not in (candidates or []):
            problems.append("predicted_function not a candidate")
        c = sub.get("confidence")
        if not isinstance(c, (int, float)) or not 0 <= c <= 1:
            problems.append("confidence not in [0,1]")
    else:
        if not isinstance(sub.get("proposed_rule"), str) or not sub["proposed_rule"].strip():
            problems.append("missing proposed_rule")
        c = sub.get("rule_confidence")
        if not isinstance(c, (int, float)) or not 0 <= c <= 1:
            problems.append("rule_confidence not in [0,1]")
    return problems


def run_session(
    *,
    network_id: str,
    protocol: str,
    condition: str,
    turn_fn: TurnFn,
    request_params: dict,
    max_turns: int,
    export: dict | None = None,
    candidate_order: list[str] | None = None,
    deny_dirs: list[Path] | None = None,
    os_sandbox: bool = True,
) -> dict:
    tools = tool_set(protocol, condition, candidate_order)
    user_msg = build_user_message(protocol, condition, export=export, network_id=network_id,
                                  candidate_order=candidate_order)
    check_prompt(protocol, user_msg, export, candidate_order)
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    transcript: list[dict] = [{"role": "user", "content": user_msg}]
    tool_log: list[dict] = []
    turns: list[dict] = []
    submission: dict | None = None
    status = "no_submission"
    nudged = False
    t_start = time.perf_counter()
    server = ToolServer(export, deny_dirs, os_sandbox) if condition == "weights_tools" else None
    try:
        for _turn in range(max_turns):
            res = turn_fn(SYSTEM_PROMPT, messages, tools)
            turns.append({"stop_reason": res.stop_reason, "model_served": res.model_served,
                          "request_id": res.request_id, "usage": res.usage,
                          "cost_usd": res.cost_usd, "latency_s": res.latency_s,
                          "attempts": res.attempts})
            transcript.append({"role": "assistant", "content": res.content_dicts})
            messages.append({"role": "assistant", "content": res.content})
            if res.stop_reason == "refusal":
                status = "refusal"
                break
            uses = [b for b in res.content if _get(b, "type") == "tool_use"]
            sub = next((b for b in uses if _get(b, "name") == "submit_interpretation"), None)
            if sub is not None:
                submission = dict(_get(sub, "input") or {})
                status = "submitted"
                break
            if not uses:
                if nudged and res.stop_reason == "end_turn":
                    status = "no_submission"
                    break
                messages.append({"role": "user", "content": NUDGE})
                transcript.append({"role": "user", "content": NUDGE})
                nudged = True
                continue
            results = []
            for b in uses:
                name, args = _get(b, "name"), _get(b, "input") or {}
                if server is None:
                    resp = {"ok": False, "error": f"tool not available: {name}"}
                else:
                    resp = server.call(name, args)
                content = resp["result"] if resp["ok"] else resp["error"]
                if resp["ok"] and scan_for_leaks(content):
                    raise LeakError(f"tool output leak in {name}")
                tool_log.append({"name": name, "args": args, "ok": resp["ok"],
                                 "result_sha256": sha256_bytes(content.encode()),
                                 "result_chars": len(content)})
                results.append({"type": "tool_result", "tool_use_id": _get(b, "id"),
                                "content": content, **({} if resp["ok"] else {"is_error": True})})
            messages.append({"role": "user", "content": results})
            transcript.append({"role": "user", "content": results})
        else:
            status = "turn_cap"
    finally:
        sandbox_active = bool(server and server.os_sandbox_active)
        if server:
            server.close()

    problems = validate_submission(protocol, submission, candidate_order) if submission else []
    valid = status == "submitted" and not problems
    usage_tot: dict[str, int] = {}
    for t in turns:
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens"):
            usage_tot[k] = usage_tot.get(k, 0) + int(t["usage"].get(k) or 0)
    return {
        "network_id": network_id,
        "protocol": protocol,
        "condition": condition,
        "candidate_order": candidate_order,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
        "user_prompt_sha256": sha256_bytes(user_msg.encode()),
        "tool_schema_sha256": tool_schema_hash(tools),
        "request_params": request_params,
        "models_served": sorted({t["model_served"] for t in turns if t["model_served"]}),
        "status": status,
        "valid": valid,
        "validation_problems": problems,
        "submission": submission,
        "n_turns": len(turns),
        "n_tool_calls": len(tool_log),
        "tool_calls": tool_log,
        "turns": turns,
        "usage_total": usage_tot,
        "cost_usd": round(sum(t["cost_usd"] for t in turns), 5),
        "wall_time_s": round(time.perf_counter() - t_start, 2),
        "os_sandbox_active": sandbox_active,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "_transcript": transcript,
    }
