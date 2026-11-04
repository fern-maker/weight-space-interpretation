import numpy as np

from weightspace.tasks import TASKS, class_balance, sample_dataset


def test_known_points():
    X = np.array([[0, 0], [0.59, 0.0], [0.61, 0.0], [0.5, 0.5], [0.3, -0.3], [-0.5, 0.2]],
                 dtype=np.float32)
    assert TASKS["circle"](X).tolist() == [1, 1, 0, 0, 1, 1]
    assert TASKS["diagonal"](X).tolist() == [0, 0, 0, 0, 0, 1]
    assert TASKS["square"](X).tolist() == [1, 1, 0, 1, 1, 1]


def test_boundaries_inclusive():
    X = np.array([[0.6, 0.0], [0.6, 0.6], [0.2, 0.2]], dtype=np.float32)
    assert TASKS["square"](X).tolist() == [1, 1, 1]
    assert TASKS["diagonal"](X)[2] == 0  # strict inequality


def test_sampling_seeded_and_in_domain():
    X1, y1 = sample_dataset("circle", 1000, 7)
    X2, y2 = sample_dataset("circle", 1000, 7)
    X3, _ = sample_dataset("circle", 1000, 8)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)
    assert X1.min() >= -1 and X1.max() <= 1


def test_class_balance_matches_geometry():
    for t, expected in [("circle", np.pi * 0.36 / 4), ("diagonal", 0.5), ("square", 0.36)]:
        _, y = sample_dataset(t, 40000, 1)
        assert abs(class_balance(y) - expected) < 0.01
