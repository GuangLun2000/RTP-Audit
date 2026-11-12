"""Structured run events: ``$RTP_WORK_DIR/runs/{config}/events.jsonl``.

Every CLI invocation appends ``start`` and ``done``/``error`` events (with exit code, elapsed time
and, for errors, the traceback). While a stage runs, every WARNING/ERROR log record and every
finished shard is appended too. ``rtp.cli status`` and ``rtp.cli feedback`` read this file, so a
coding agent sees exactly what happened in a Colab session without the notebook output.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

_CTX: dict[str, Any] = {}
_PATH: Path | None = None


def configure(path: Path | None, **context: Any) -> None:
    """Start writing events to ``path`` with ``context`` (config, stage, model) attached to each event."""
    global _PATH
    _PATH = Path(path) if path else None
    _CTX.clear()
    _CTX.update({k: v for k, v in context.items() if v is not None})
    if _PATH:
        _PATH.parent.mkdir(parents=True, exist_ok=True)


def redact(obj: Any) -> Any:
    """Mask $HF_TOKEN and hf_... tokens in every string of ``obj`` (messages, tracebacks, argv)."""
    from rtp.log import redact_text

    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def emit(kind: str, message: str = "", **data: Any) -> None:
    if _PATH is None:
        return
    rec = redact({"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z", "kind": kind, **_CTX,
                  "message": message, **({"data": data} if data else {})})
    try:
        with open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except OSError:  # events must never break a run
        pass


def read(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


class EventLogHandler(logging.Handler):
    """Forward WARNING+ log records (already secret-redacted by the logging filter) to the event file."""

    def __init__(self):
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - logging API
        emit("warning" if record.levelno < logging.ERROR else "log_error", record.getMessage(),
             logger=record.name)
