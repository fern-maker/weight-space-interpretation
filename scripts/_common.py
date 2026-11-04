"""Shared CLI setup for scripts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_WARNINGS", "0")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def setup_logging(root: Path, name: str) -> None:
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / f"{name}.log")],
    )


def load_dotenv() -> None:
    """Load KEY=VALUE lines from .env (git-ignored) without overriding the shell."""
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
