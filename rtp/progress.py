"""Progress reporting that stays readable in Colab cells, log files and feedback bundles.

``RTP_PROGRESS=tqdm`` draws tqdm bars; ``RTP_PROGRESS=log`` (the default when stderr is not a
terminal, e.g. a Colab ``!python`` call or the notebook helper) writes one log line per ~10%
of the work or every ``interval`` seconds, with rate and ETA. Carriage-return bars garble
captured output, which is why they are off by default outside a terminal.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger("rtp.progress")
_SILENCED = 0


@contextmanager
def silenced() -> Iterator[None]:
    """Suppress inner progress reporting (a caller that reports progress at a coarser level itself)."""
    global _SILENCED
    _SILENCED += 1
    try:
        yield
    finally:
        _SILENCED -= 1


def mode() -> str:
    if _SILENCED:
        return "off"
    m = os.environ.get("RTP_PROGRESS", "").strip().lower()
    if m in ("tqdm", "log", "off"):
        return m
    return "tqdm" if sys.stderr.isatty() else "log"


def fmt_seconds(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def track(items: Iterable[T], total: int | None = None, desc: str = "", interval: float = 30.0) -> Iterator[T]:
    """Iterate ``items`` while reporting progress (``total`` defaults to ``len(items)``)."""
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    m = mode()
    if m == "tqdm":
        from tqdm.auto import tqdm

        yield from tqdm(items, total=total, desc=desc, leave=False, mininterval=5.0)
        return
    t0 = last = time.time()
    step = max(1, (total or 10) // 10)
    done = 0
    for x in items:
        yield x
        done += 1
        now = time.time()
        if m == "log" and total and (done % step == 0 or now - last >= interval) and done < total:
            rate = done / max(now - t0, 1e-9)
            log.info("%s: %d/%d (%.0f%%), %.1f/s, ETA %s", desc, done, total, 100 * done / total, rate,
                     fmt_seconds((total - done) / rate))
            last = now
    if m == "log" and total and total > 1 and time.time() - t0 > 5:
        log.info("%s: %d/%d done in %s", desc, done, total, fmt_seconds(time.time() - t0))
