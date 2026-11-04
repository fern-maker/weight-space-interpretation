import json
import subprocess
import sys

import pytest
from fakes import FakeLLM, answer_a, answer_b

from weightspace.anonymization import allocate_ids, scan_for_leaks
from weightspace.interpreter import LeakError, check_prompt, run_session
from weightspace.prompts import SYSTEM_PROMPT, build_user_message
from weightspace.tool_server import ToolServer
from weightspace.tools import ANALYSIS_TOOLS, run_tool, tool_set
from weightspace.weight_exporter import export_text, model_to_export

CANDS = ["square", "circle", "diagonal"]


@pytest.fixture
def export(rand_model):
    return model_to_export(rand_model, "network_017")


def test_export_has_no_forbidden_terms(export):
    assert scan_for_leaks(export_text(export)) == []


def test_protocol_a_prompt_is_clean(export):
    msg = build_user_message("A", "weights_tools", export=export)
    template = msg.replace(export_text(export), "")
    assert scan_for_leaks(SYSTEM_PROMPT + template, check_constants=True) == []
    for formula_bit in ["x1**2 + x2**2", "x2 > x1", "abs(x1)"]:
        assert formula_bit not in msg
    check_prompt("A", msg, export, None)


def test_arch_only_prompt_has_no_parameters():
    msg = build_user_message("A", "architecture_only", network_id="network_000")
    assert '"parameters"' not in msg and '"weight"' not in msg and "withheld" in msg
    assert scan_for_leaks(SYSTEM_PROMPT + msg, check_constants=True) == []


def test_protocol_b_prompt_only_leaks_candidates(export):
    msg = build_user_message("B", "weights_tools", export=export, candidate_order=CANDS)
    assert all(c in msg for c in CANDS)
    check_prompt("B", msg, export, CANDS)
    template = msg.replace(export_text(export), "")
    assert set(scan_for_leaks(template, check_constants=True)) <= {"circle", "square",
                                                                    "diagonal", "0.36", "0.6"}


def test_leaky_export_is_blocked(export):
    bad = dict(export, network_id="circle_seed_3")
    msg = build_user_message("A", "weights_tools", export=bad)
    with pytest.raises(LeakError):
        check_prompt("A", msg, bad, None)


def test_scan_catches_meta_terms():
    assert "seed" in scan_for_leaks("trained with seed 4")
    assert "artifacts" in scan_for_leaks("see artifacts/private/ground_truth.json")
    assert "0.36" in scan_for_leaks("r^2 <= 0.36", check_constants=True)
    assert scan_for_leaks("value 0.3612", check_constants=True) == []


def test_ids_are_shuffled_and_opaque():
    units = [{"task": t, "kind": "original", "i": i} for t in ("circle", "diagonal", "square")
             for i in range(5)]
    a = allocate_ids(units, 42)
    b = allocate_ids(units, 42)
    assert [u["network_id"] for u in a] == [f"network_{i:03d}" for i in range(1, 16)]
    assert a == b
    tasks_in_id_order = [u["task"] for u in a]
    assert tasks_in_id_order != sorted(tasks_in_id_order)
    assert all(not any(t in u["network_id"] for t in ("circle", "diagonal", "square"))
               for u in a)


def test_no_tool_accepts_points_paths_or_code():
    banned = {"x", "x1", "x2", "point", "points", "input", "inputs", "path", "file", "code",
              "expression", "query"}
    for t in ANALYSIS_TOOLS:
        assert not (set(t["input_schema"]["properties"]) & banned), t["name"]
        assert t["input_schema"]["additionalProperties"] is False
    assert {t["name"] for t in tool_set("B", "architecture_only", CANDS)} == {
        "submit_interpretation"}


def test_run_tool_rejects_unknown_and_bad_args(export):
    for name, args in [("forward", {"x": [0, 0]}), ("evaluate", {}), ("read_file", {"path": "/"})]:
        with pytest.raises(ValueError):
            run_tool(export, name, args)
    with pytest.raises(ValueError):
        run_tool(export, "get_weight_matrix", {"layer": "../artifacts/private"})
    with pytest.raises(ValueError):
        run_tool(export, "get_bias_vector", {"layer": "layer1", "x": [0.1, 0.2]})


def test_tool_server_is_isolated(export, tmp_path):
    with ToolServer(export, [tmp_path]) as s:
        h = s.hello
        assert h["forbidden_loaded"] == []
        assert "ANTHROPIC_API_KEY" not in h["env_keys"]
        assert "ws_tools_" in h["cwd"]
        assert s.call("forward", {"x": [0, 0]})["ok"] is False
        r = s.call("get_bias_vector", {"layer": "layer3"})
        assert r["ok"] and len(json.loads(r["result"])) == 1


def test_interpreter_import_does_not_load_private_modules():
    code = ("import sys, weightspace.interpreter; "
            "bad=[m for m in ('torch','weightspace.evaluation','weightspace.checkpointing',"
            "'weightspace.experiment','weightspace.training') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_full_session_sees_only_public_data(export):
    fake = FakeLLM(answer_b("square"), tool_calls=[("forward", {"x": [0.1, 0.2]}),
                                                   ("first_layer_hyperplanes", {})])
    rec = run_session(network_id=export["network_id"], protocol="B", condition="weights_tools",
                      turn_fn=fake, request_params={}, max_turns=6, export=export,
                      candidate_order=CANDS)
    assert rec["valid"] and rec["submission"]["predicted_function"] == "square"
    assert rec["tool_calls"][0]["ok"] is False  # forward pass refused
    everything = json.dumps(rec["_transcript"], default=str)
    leaks = set(scan_for_leaks(everything))
    assert leaks <= {"circle", "square", "diagonal"}


def test_protocol_a_session_transcript_has_no_task_words(export):
    fake = FakeLLM(answer_a("x1 > 0"), tool_calls=[("singular_values", {"layer": "layer2"})])
    rec = run_session(network_id=export["network_id"], protocol="A", condition="weights_tools",
                      turn_fn=fake, request_params={}, max_turns=6, export=export)
    assert rec["valid"]
    # the fake's own submission is the only model-written content; inputs must be clean
    inputs = [m for m in rec["_transcript"] if m["role"] == "user"]
    assert scan_for_leaks(json.dumps(inputs)) == []
