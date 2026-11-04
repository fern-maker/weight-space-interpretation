import os

os.environ.setdefault("KMP_WARNINGS", "0")

import pytest  # noqa: E402

from weightspace.training import random_init_network, train_network  # noqa: E402

FAST = {"n_train": 2000, "n_val": 500, "n_test": 1000, "batch_size": 256, "lr": 0.01,
        "max_epochs": 40, "early_stop_val_acc": 0.99, "early_stop_patience": 3,
        "accuracy_target": 0.9, "num_threads": 1}


@pytest.fixture(scope="session")
def fast_cfg():
    return dict(FAST)


@pytest.fixture(scope="session")
def trained(fast_cfg):
    return {t: train_network(t, 0, fast_cfg, "test-ns") for t in ("circle", "diagonal", "square")}


@pytest.fixture
def rand_model():
    return random_init_network("test-ns", 0)[0]
