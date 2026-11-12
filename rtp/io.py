"""Deterministic file IO, sharded writing with completion markers, and stage bookkeeping.

Resumability contract (spec §9):
  * a long job is split into shards of ``shard_size`` records;
  * each shard is written atomically (tmp file + rename) and then gets a ``.done`` marker;
  * a re-run skips shards whose marker exists;
  * when the whole stage finishes, ``_DONE`` and ``summary.md`` are written.
A stage directory remembers the hash of the config that created it; resuming with a
different config is refused (pass ``force=True`` to wipe and restart).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DONE_FILE = "_DONE"
STAGE_META = "stage_meta.json"


class StageConfigMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------- basic files
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def dumps_record(rec: dict[str, Any]) -> str:
    """Canonical JSON line: key order as constructed, UTF-8, no NaN."""
    return json.dumps(rec, ensure_ascii=False, allow_nan=False)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    lines = [dumps_record(r) for r in records]
    atomic_write_text(path, "".join(line + "\n" for line in lines))
    return len(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_records(name: str, records: Sequence[dict[str, Any]] | pd.DataFrame) -> str:
    """Log the field structure of a dataset on first read (spec §5: never assume field names)."""
    if isinstance(records, pd.DataFrame):
        fields = {c: str(t) for c, t in records.dtypes.items()}
        n = len(records)
        example = records.iloc[0].to_dict() if n else {}
    else:
        n = len(records)
        example = records[0] if n else {}
        fields = {k: type(v).__name__ for k, v in example.items()}
    text = f"dataset '{name}': {n} rows; fields: {fields}"
    log.info(text)
    if n:
        log.info("dataset '%s' first row: %s", name, {k: _short(v) for k, v in example.items()})
    return text


def _short(v: Any, limit: int = 80) -> Any:
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


# --------------------------------------------------------------------------- stage dirs
def is_done(stage_dir: Path) -> bool:
    return (Path(stage_dir) / DONE_FILE).exists()


def mark_done(stage_dir: Path, info: dict[str, Any] | None = None) -> None:
    write_json(Path(stage_dir) / DONE_FILE, info or {})


def stored_stage_hash(stage_dir: Path) -> str | None:
    meta_path = Path(stage_dir) / STAGE_META
    return read_json(meta_path).get("stage_hash") if meta_path.exists() else None


def check_stage_hash(stage_dir: Path, stage_hash: str) -> None:
    """Raise if ``stage_dir`` belongs to a different configuration (finished or not)."""
    stored = stored_stage_hash(stage_dir)
    if stored is not None and stored != stage_hash:
        raise StageConfigMismatch(
            f"{stage_dir} was produced with a different configuration or different seed files (hash {stored} "
            f"!= {stage_hash}). Use a new config name, or pass --force to discard it."
        )


def finished_stage(stage_dir: Path, stage_hash: str, force: bool) -> bool:
    """True if the stage is finished under the SAME configuration (so it can be skipped).

    A finished stage from another configuration raises unless ``force`` (it is then redone).
    """
    if force or not is_done(stage_dir):
        return False
    check_stage_hash(stage_dir, stage_hash)
    return True


def prepare_stage_dir(stage_dir: Path, config_yaml: str, stage_hash: str, force: bool = False,
                      extra_meta: dict[str, Any] | None = None) -> Path:
    """Create/validate a stage dir. Saves a full config copy and the hash that owns the dir."""
    stage_dir = Path(stage_dir)
    meta_path = stage_dir / STAGE_META
    if force and stage_dir.exists():
        log.warning("--force: removing %s", stage_dir)
        shutil.rmtree(stage_dir)
    if meta_path.exists():
        check_stage_hash(stage_dir, stage_hash)
    else:
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_json(meta_path, {"stage_hash": stage_hash, **(extra_meta or {})})
    atomic_write_text(stage_dir / "config.yaml", config_yaml)
    return stage_dir


# --------------------------------------------------------------------------- shards
def shard_path(out_dir: Path, idx: int) -> Path:
    return Path(out_dir) / f"shard_{idx:05d}.parquet"


def run_sharded(records: Sequence[Any], shard_size: int, out_dir: Path,
                process: Callable[[Sequence[Any], int], pd.DataFrame], desc: str = "shards") -> list[Path]:
    """Process ``records`` shard by shard, skipping shards that already have a ``.done`` marker."""
    import time

    from rtp import events
    from rtp.progress import fmt_seconds

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, -(-len(records) // shard_size)) if records else 0
    paths = []
    t0, n_done_now = time.time(), 0
    for idx in range(n_shards):
        path = shard_path(out_dir, idx)
        marker = path.with_suffix(".done")
        paths.append(path)
        if marker.exists() and path.exists():
            log.info("%s: shard %d/%d already done, skipping", desc, idx + 1, n_shards)
            continue
        chunk = records[idx * shard_size: (idx + 1) * shard_size]
        log.info("%s: shard %d/%d (%d records)", desc, idx + 1, n_shards, len(chunk))
        ts = time.time()
        df = process(chunk, idx)
        write_parquet(path, df)
        atomic_write_text(marker, f"{len(df)}\n")
        n_done_now += 1
        per = (time.time() - t0) / n_done_now
        left = sum(1 for j in range(idx + 1, n_shards) if not shard_path(out_dir, j).with_suffix(".done").exists())
        log.info("%s: shard %d/%d finished in %s; ETA for this stage %s", desc, idx + 1, n_shards,
                 fmt_seconds(time.time() - ts), fmt_seconds(per * left))
        events.emit("shard", f"{desc} shard {idx + 1}/{n_shards}", shard=idx, n_shards=n_shards, n=len(chunk),
                    seconds=round(time.time() - ts, 1))
    return paths


def load_shards(out_dir: Path) -> pd.DataFrame:
    paths = sorted(Path(out_dir).glob("shard_*.parquet"))
    paths = [p for p in paths if p.with_suffix(".done").exists()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
