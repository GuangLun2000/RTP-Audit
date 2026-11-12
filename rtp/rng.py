"""Deterministic random streams keyed by strings (never Python's salted ``hash``)."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_int(*keys: object) -> int:
    blob = "\x1f".join(str(k) for k in keys).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def rng_for(seed: int, *keys: object) -> np.random.Generator:
    """Independent generator for (seed, *keys); identical across runs, platforms and Python versions."""
    return np.random.default_rng(stable_int(seed, *keys))
