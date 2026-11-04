"""Train and validate one network. Never raises on low accuracy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from .models import build_mlp, logits
from .tasks import class_balance, sample_dataset
from .utils import derive_seed


@dataclass
class TrainingResult:
    task: str
    seed_index: int
    init_seed: int
    data_seeds: dict
    status: str  # "ok" | "failed_accuracy"
    epochs: int
    train_acc: float
    val_acc: float
    test_acc: float
    class_balance: dict
    wall_time_s: float
    device: str
    loss_curve: list[float] = field(default_factory=list)
    val_acc_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def seeds_for(namespace: str, task: str, seed_index: int) -> dict[str, int]:
    return {
        p: derive_seed(namespace, task, seed_index, p)
        for p in ("init", "train_data", "val_data", "test_data", "batch_order")
    }


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model_cpu = model.to("cpu")
    pred = (logits(model_cpu, X) > 0).numpy().astype(np.float32)
    return float(np.mean(pred == y))


def init_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return build_mlp()


def train_network(
    task: str, seed_index: int, tcfg: dict, namespace: str, device: str = "cpu"
) -> tuple[nn.Sequential, TrainingResult]:
    torch.set_num_threads(int(tcfg.get("num_threads", 1)))
    torch.use_deterministic_algorithms(True)
    seeds = seeds_for(namespace, task, seed_index)
    Xtr, ytr = sample_dataset(task, tcfg["n_train"], seeds["train_data"])
    Xva, yva = sample_dataset(task, tcfg["n_val"], seeds["val_data"])
    Xte, yte = sample_dataset(task, tcfg["n_test"], seeds["test_data"])

    dev = resolve_device(device)
    model = init_model(seeds["init"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(Xtr, device=dev)
    yt = torch.as_tensor(ytr, device=dev)
    gen = torch.Generator().manual_seed(seeds["batch_order"])
    bs = tcfg["batch_size"]

    losses: list[float] = []
    val_curve: list[float] = []
    streak = 0
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(dev)
        total = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(Xt))
        model.eval()
        va = accuracy(model, Xva, yva)
        model.to(dev)
        val_curve.append(va)
        streak = streak + 1 if va >= tcfg["early_stop_val_acc"] else 0
        if streak >= tcfg["early_stop_patience"]:
            break
    wall = time.perf_counter() - t0
    model = model.to("cpu").eval()
    te = accuracy(model, Xte, yte)
    result = TrainingResult(
        task=task,
        seed_index=seed_index,
        init_seed=seeds["init"],
        data_seeds={k: v for k, v in seeds.items() if k != "init"},
        status="ok" if te >= tcfg["accuracy_target"] else "failed_accuracy",
        epochs=epoch,
        train_acc=accuracy(model, Xtr, ytr),
        val_acc=val_curve[-1],
        test_acc=te,
        class_balance={
            "train": class_balance(ytr),
            "val": class_balance(yva),
            "test": class_balance(yte),
        },
        wall_time_s=round(wall, 3),
        device=str(dev),
        loss_curve=[round(v, 6) for v in losses],
        val_acc_curve=[round(v, 5) for v in val_curve],
    )
    return model, result


def random_init_network(namespace: str, index: int) -> tuple[nn.Sequential, int]:
    seed = derive_seed(namespace, "random_init", index, "init")
    return init_model(seed).eval(), seed
