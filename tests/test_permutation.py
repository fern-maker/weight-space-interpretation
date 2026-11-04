import numpy as np

from weightspace.models import logits
from weightspace.permutation import (
    permute_hidden,
    random_permutations,
    sign_flip,
    verify_equivalence,
)
from weightspace.tasks import sample_inputs


def test_permutation_preserves_function(trained):
    X = sample_inputs(4096, 0)
    for m, _ in trained.values():
        perms = random_permutations(m, 123)
        assert all(not np.array_equal(p, np.arange(16)) for p in perms)
        pm = permute_hidden(m, perms)
        eq = verify_equivalence(m, pm, X)
        assert eq["equivalent"], eq
        # parameters actually changed
        assert not np.allclose(m[0].weight.detach().numpy(), pm[0].weight.detach().numpy())


def test_permutation_is_exact_reordering(rand_model):
    perms = random_permutations(rand_model, 5)
    pm = permute_hidden(rand_model, perms)
    p1, p2 = perms
    W1, W2 = rand_model[0].weight.detach().numpy(), rand_model[2].weight.detach().numpy()
    assert np.array_equal(pm[0].weight.detach().numpy(), W1[p1])
    assert np.array_equal(pm[2].weight.detach().numpy(), W2[p2][:, p1])
    assert np.array_equal(pm[4].weight.detach().numpy(),
                          rand_model[4].weight.detach().numpy()[:, p2])


def test_wrong_permutation_detected(rand_model):
    """Permuting rows without the matching columns must break equivalence."""
    import copy

    import torch
    bad = copy.deepcopy(rand_model)
    with torch.no_grad():
        bad[0].weight.copy_(bad[0].weight[torch.arange(15, -1, -1)])
    X = sample_inputs(1024, 1)
    assert not verify_equivalence(rand_model, bad, X)["equivalent"]


def test_sign_flip_preserves_function(rand_model):
    X = sample_inputs(1024, 2)
    f = sign_flip(rand_model, 0, [0, 3, 7])
    assert float((logits(rand_model, X) - logits(f, X)).abs().max()) < 1e-5
