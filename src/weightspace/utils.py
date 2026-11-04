"""Shared helpers: seeding, hashing, canonical JSON, environment info."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent


def derive_seed(*parts: object) -> int:
    """Stable seed from arbitrary parts (SHA-256, mod 2**31)."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # tool server runs without torch
        pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def round_sig(x: float, digits: int) -> float:
    """Round to `digits` significant digits (returns a plain float)."""
    return float(f"{float(x):.{digits}g}")


def source_tree_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def resource_snapshot() -> dict[str, float]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes, Linux in KB
    scale = 1 if sys.platform == "darwin" else 1024
    return {"max_rss_mb": round(ru.ru_maxrss * scale / 2**20, 1), "cpu_user_s": ru.ru_utime}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        pass
    try:
        import anthropic

        info["anthropic_sdk"] = anthropic.__version__
    except ImportError:
        pass
    return info
