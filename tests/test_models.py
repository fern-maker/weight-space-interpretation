import pytest
from torch import nn

from weightspace.models import N_PARAMS, assert_architecture, build_mlp


def test_architecture_exact():
    m = build_mlp()
    assert [type(x) for x in m] == [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
    assert (m[0].in_features, m[0].out_features) == (2, 16)
    assert (m[2].in_features, m[2].out_features) == (16, 16)
    assert (m[4].in_features, m[4].out_features) == (16, 1)
    assert sum(p.numel() for p in m.parameters()) == N_PARAMS == 337
    assert_architecture(m)


def test_rejects_other_architecture():
    with pytest.raises(ValueError):
        assert_architecture(nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)))
