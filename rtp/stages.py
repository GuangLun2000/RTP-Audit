"""Stage identity and provenance checks for per-model stages (measure, open-ended).

A stage dir is owned by one hash of: stage settings, model, split, prompt mode, the sha256 of
the input files it reads, and the digest of the code that computes it. Its metadata also
records the environment; resuming a HALF-finished stage under a different library version,
device or model revision is refused, because shards would then mix numerics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rtp import io
from rtp.config import Config
from rtp.provenance import RESUME_CRITICAL, environment, stage_code_digest

log = logging.getLogger(__name__)

STAGE_MODULES = {"measure": "rtp.measure.choice", "open_ended": "rtp.measure.open_ended",
                 "extract": "rtp.detect.activations", "probe": "rtp.detect.probes"}


def stage_inputs_digest(cfg: Config, kind: str, split: str, model_id: str | None = None) -> str:
    """Digest of what a stage reads: data files from the manifest (requires built data); for the probe stage
    also the identity of the extract stage whose activations it consumes."""
    from rtp.data.build import CATALOGUE, CHOICE, PROBE, require_built

    manifest = io.read_json(require_built(cfg) / "data_manifest.json")
    sha = manifest["sha256"]
    if kind == "measure":
        parts = [sha[f"{split}/{CHOICE}"], sha[CATALOGUE]]
    elif kind == "open_ended":  # the catalogue and the seed snapshot (brands, aliases, Appendix A)
        snap = manifest["seed_snapshot_sha256"]
        parts = [sha[CATALOGUE]] + [snap[k] for k in sorted(snap)]
    elif kind == "extract":
        parts = [sha[f"{split}/{PROBE}"], sha[f"{split}/{CHOICE}"], sha[CATALOGUE]]
    elif kind == "probe":
        from rtp.paths import stage_dir

        edir = stage_dir(cfg.name, model_id, "extract" if split == "dev" else "extract_test")
        if not io.is_done(edir):
            raise io.StageConfigMismatch(f"probe needs a finished extract stage at {edir}; run `extract` first")
        mdir = stage_dir(cfg.name, model_id, "measure")  # read for the familiarity link (optional)
        measured = (io.stored_stage_hash(mdir) or "") if io.is_done(mdir) else "none"
        parts = [io.stored_stage_hash(edir) or "", sha[CATALOGUE], measured]
    else:
        raise ValueError(f"unknown stage kind {kind}")
    return "-".join(p[:16] for p in parts)


def stage_hash(cfg: Config, kind: str, model_id: str, split: str, prompt_mode: str = "auto") -> str:
    return cfg.stage_hash(kind, model_id, split, prompt_mode,
                          inputs=stage_inputs_digest(cfg, kind, split, model_id),
                          code=stage_code_digest(STAGE_MODULES[kind]))


def stage_meta(kind: str, model_id: str, split: str, prompt_mode: str) -> dict[str, Any]:
    return {"model_id": model_id, "split": split, "stage": kind, "prompt_mode": prompt_mode,
            "code_digest": stage_code_digest(STAGE_MODULES[kind]), "created_with": environment()}


def finished_shards(sdir: Path) -> list[Path]:
    """Completion markers of every shard of a stage (measure: shards/, open-ended: shards_*/, extract: <set>/)."""
    return sorted(Path(sdir).rglob("shard_*.done"))


def check_resume_environment(sdir: Path, force: bool = False) -> None:
    """Refuse to continue a half-finished stage under a different torch/transformers/tokenizers/device.

    The baseline is the environment of the first COMPUTED shard: while no shard exists, the stored
    environment (and model revision) is simply replaced by the current one.
    """
    meta_path = Path(sdir) / io.STAGE_META
    if force or io.is_done(sdir) or not meta_path.exists():
        return
    meta = io.read_json(meta_path)
    if not finished_shards(sdir):  # nothing computed yet: (re)set the baseline to this session
        meta["created_with"] = environment()
        meta.pop("model_commit", None)
        io.write_json(meta_path, meta)
        return
    created = meta.get("created_with", {})
    now = environment()
    diff = {k: (created.get(k), now.get(k)) for k in RESUME_CRITICAL if created.get(k) != now.get(k)}
    if diff:
        raise io.StageConfigMismatch(
            f"{sdir} is half-finished under a different environment {diff}; resuming would mix numerics across "
            f"shards. Pass --force to restart the stage.")


def record_model_revision(sdir: Path, model: Any) -> None:
    """Store the loaded model's Hub commit; refuse if a resumed stage was started on another revision."""
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    meta_path = Path(sdir) / io.STAGE_META
    meta = io.read_json(meta_path)
    stored = meta.get("model_commit")
    if stored is not None and commit is not None and stored != commit and finished_shards(sdir):
        raise io.StageConfigMismatch(f"{sdir} was started on model revision {stored}, now {commit}; pass --force")
    if stored is None or not finished_shards(sdir):
        meta["model_commit"] = commit
        io.write_json(meta_path, meta)
