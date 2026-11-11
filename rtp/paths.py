"""Directory conventions under $RTP_WORK_DIR.

Layout (spec §9):
    $RTP_WORK_DIR/runs/{config_name}/data/                      model-independent generated data
    $RTP_WORK_DIR/runs/{config_name}/{model_slug}/{stage}/      per-model stage outputs
    $RTP_WORK_DIR/runs/{config_name}/report.md                  cross-model report
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

DEFAULT_WORK_DIR = "/content/drive/MyDrive/rtp"
WORK_DIR_ENV = "RTP_WORK_DIR"


class WorkDirError(RuntimeError):
    pass


def work_dir() -> Path:
    raw = os.environ.get(WORK_DIR_ENV, DEFAULT_WORK_DIR)
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkDirError(
            f"cannot create work dir '{path}' ({exc}). Set {WORK_DIR_ENV} to a writable directory "
            f"(default {DEFAULT_WORK_DIR} assumes Colab with Drive mounted) or pass --work-dir."
        ) from exc
    return path


def set_work_dir(path: str | os.PathLike) -> None:
    os.environ[WORK_DIR_ENV] = str(Path(path).expanduser().resolve())


def run_dir(config_name: str) -> Path:
    return work_dir() / "runs" / config_name


def data_dir(config_name: str) -> Path:
    return run_dir(config_name) / "data"


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug. Hub IDs map 'org/name' -> 'org__name'; local paths get a short hash suffix."""
    if os.path.exists(model_id):
        p = Path(model_id).resolve()
        digest = hashlib.sha256(str(p).encode()).hexdigest()[:8]
        return f"local__{re.sub(r'[^A-Za-z0-9._-]+', '_', p.name)}__{digest}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))


def stage_dir(config_name: str, model_id: str, stage: str) -> Path:
    return run_dir(config_name) / model_slug(model_id) / stage
