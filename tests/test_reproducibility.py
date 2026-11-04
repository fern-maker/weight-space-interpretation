from weightspace.checkpointing import state_hash
from weightspace.training import seeds_for, train_network
from weightspace.utils import derive_seed


def test_derive_seed_stable():
    assert derive_seed("ns", "circle", 0, "init") == derive_seed("ns", "circle", 0, "init")
    assert derive_seed("ns", "circle", 0, "init") != derive_seed("ns", "circle", 1, "init")


def test_seed_purposes_independent():
    s = seeds_for("ns", "circle", 0)
    assert len(set(s.values())) == len(s)


def test_same_seed_same_weights(fast_cfg):
    cfg = dict(fast_cfg, max_epochs=3)
    a, _ = train_network("square", 3, cfg, "test-ns")
    b, _ = train_network("square", 3, cfg, "test-ns")
    c, _ = train_network("square", 4, cfg, "test-ns")
    assert state_hash(a) == state_hash(b)
    assert state_hash(a) != state_hash(c)
