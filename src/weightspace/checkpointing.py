"""Checkpoint save/load/verify and the append-only training registry."""

from __future__ import annotations

import io
from pathlib import Path

import torch
from torch import nn

from .models import assert_architecture, build_mlp
from .utils import append_jsonl, read_jsonl, sha256_bytes


def save_checkpoint(model: nn.Module, path: str | Path) -> str:
    """Save a state dict deterministically; return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, buf)
    data = buf.getvalue()
    path.write_bytes(data)
    return sha256_bytes(data)


def load_checkpoint(path: str | Path) -> nn.Sequential:
    model = build_mlp()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert_architecture(model)
    return model.eval()


def state_hash(model: nn.Module) -> str:
    """Hash of parameter values (independent of torch serialization details)."""
    parts = []
    for k, v in sorted(model.state_dict().items()):
        parts.append(k.encode())
        parts.append(v.detach().cpu().contiguous().numpy().tobytes())
    return sha256_bytes(b"".join(parts))


def verify_checkpoint(path: str | Path, expected_state_hash: str) -> bool:
    return state_hash(load_checkpoint(path)) == expected_state_hash


class DuplicateRunError(RuntimeError):
    pass


class Registry:
    """Append-only JSONL log of every training run (successes and failures)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        return read_jsonl(self.path)

    def find(self, task: str, seed_index: int, training_hash: str) -> dict | None:
        for e in self.entries():
            if (e["task"], e["seed_index"], e["training_hash"]) == (
                task,
                seed_index,
                training_hash,
            ):
                return e
        return None

    def add(self, entry: dict) -> None:
        key = (entry["task"], entry["seed_index"], entry["training_hash"])
        if self.find(*key) is not None:
            raise DuplicateRunError(f"run already registered: {key}")
        append_jsonl(self.path, entry)
