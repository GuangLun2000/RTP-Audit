"""Logging setup with a redaction filter so that tokens never reach logs."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RedactSecrets(logging.Filter):
    """Mask Hugging Face tokens and the literal value of $HF_TOKEN in every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        secret = os.environ.get("HF_TOKEN")
        if secret and len(secret) >= 8:
            msg = msg.replace(secret, "***")
        msg = _TOKEN_RE.sub("hf_***", msg)
        record.msg, record.args = msg, None
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        if getattr(h, "_rtp", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(RedactSecrets())
    handler._rtp = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    for noisy in ("urllib3", "httpx", "filelock", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_file_log(path: Path) -> logging.Handler:
    """Also log to ``path`` (appending); returns the handler so callers can remove it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(RedactSecrets())
    logging.getLogger().addHandler(handler)
    return handler
