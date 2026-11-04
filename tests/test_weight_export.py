import json

import numpy as np

from weightspace import weight_inspector as wi
from weightspace.weight_exporter import EXPORT_KEYS, model_to_export, validate_export, write_export


def test_export_keys_and_hash(rand_model, tmp_path):
    e = model_to_export(rand_model, "network_001")
    assert set(e) == EXPORT_KEYS
    validate_export(e)
    write_export(e, tmp_path / "a.json")
    write_export(model_to_export(rand_model, "network_001"), tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    loaded = json.loads((tmp_path / "a.json").read_text())
    validate_export(loaded)


def test_export_values_match_model(rand_model):
    e = model_to_export(rand_model, "network_001", sig_digits=6)
    w = np.array(e["parameters"]["layer2"]["weight"])
    ref = rand_model[2].weight.detach().numpy()
    assert w.shape == (16, 16)
    assert np.allclose(w, ref, rtol=1e-5, atol=1e-7)
    assert e["parameters"]["layer3"]["weight_shape"] == [1, 16]


def test_tamper_detected(rand_model):
    e = model_to_export(rand_model, "network_001")
    e["parameters"]["layer1"]["bias"][0] += 1
    try:
        validate_export(e)
        raise AssertionError
    except ValueError:
        pass


def test_inspector_on_export_equals_model(rand_model):
    e = model_to_export(rand_model, "network_001", sig_digits=9)
    a = wi.compute_singular_values(e, "layer2")
    b = wi.compute_singular_values(rand_model, "layer2")
    assert np.allclose(a, b, rtol=1e-5)
    n = wi.inspect_neuron(e, "layer1", 3)
    assert len(n["incoming_weights"]) == 2 and len(n["outgoing_weights"]) == 16
    h = wi.first_layer_hyperplanes(e)
    assert len(h) == 16 and all("signed_offset_from_origin" in x for x in h)
    assert np.array(wi.matrix_product(e, "layer3", "layer2")).shape == (1, 16)
