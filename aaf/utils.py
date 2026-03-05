# aaf/utils.py
from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set RNG seeds for reproducibility.
    Note: this covers Python + NumPy. Add torch seeding here only if you use torch directly.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def now_ms() -> int:
    """Monotonic timestamp in milliseconds (suitable for latency measurements)."""
    return int(time.perf_counter() * 1000)
