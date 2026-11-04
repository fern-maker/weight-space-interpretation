from weightspace.checkpointing import (
    DuplicateRunError,
    Registry,
    load_checkpoint,
    save_checkpoint,
    state_hash,
)
from weightspace.training import train_network


def test_training_reaches_target(trained):
    for t, (_, res) in trained.items():
        assert res.status == "ok", (t, res.test_acc)
        assert res.loss_curve[-1] < res.loss_curve[0]


def test_failure_recorded_not_raised(fast_cfg):
    cfg = dict(fast_cfg, max_epochs=1, accuracy_target=0.999, lr=1e-5)
    _, res = train_network("circle", 1, cfg, "test-ns")
    assert res.status == "failed_accuracy"


def test_checkpoint_roundtrip(tmp_path, trained):
    m, _ = trained["circle"]
    save_checkpoint(m, tmp_path / "a.pt")
    assert state_hash(load_checkpoint(tmp_path / "a.pt")) == state_hash(m)


def test_registry_rejects_duplicates(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    e = {"task": "circle", "seed_index": 0, "training_hash": "h", "status": "ok"}
    reg.add(e)
    try:
        reg.add(e)
        raise AssertionError("duplicate accepted")
    except DuplicateRunError:
        pass
    reg.add({**e, "seed_index": 1})
    assert len(reg.entries()) == 2
