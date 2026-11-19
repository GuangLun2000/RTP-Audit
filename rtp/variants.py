"""Model variants: a registered checkpoint plus an intervention, addressed by one model id.

    <checkpoint>@scale=2        all output logits multiplied by 2 (a pure scale change, i.e. temperature 1/2)
    zoo:<run_id>                the run's base checkpoint with its LoRA adapter loaded (not merged)
    zoo:<run_id>@scale=0.5      both

A variant id is used wherever a model id is (``--model``, configs, cycle files); every stage still loads
exactly one model. ``registry.get_model_spec`` returns the base checkpoint's entry with the variant fields
filled in, so the intervention (and the adapter's sha256) is part of each stage's identity.

Adapters live in the zoo, ``$RTP_WORK_DIR/zoo/<run_id>/adapter/``, and are listed in ``zoo/manifest.json``
with their base checkpoint, sha256 and training metadata. The sha256 is read from the manifest (never
recomputed by ``status`` or ``report``) and checked against the files when the adapter is loaded.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZOO_PREFIX = "zoo:"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class VariantError(ValueError):
    pass


@dataclass(frozen=True)
class Variant:
    model_id: str          # the full variant id
    base_id: str           # registered checkpoint
    logit_scale: float = 1.0
    adapter: str | None = None  # zoo run id


def is_variant(model_id: str) -> bool:
    return model_id.startswith(ZOO_PREFIX) or "@" in model_id


def parse(model_id: str) -> Variant:
    """Split a variant id into its parts. Zoo ids are resolved to their base checkpoint through the manifest."""
    head, *mods = model_id.split("@")
    scale = 1.0
    for mod in mods:
        key, sep, value = mod.partition("=")
        if key != "scale" or not sep:
            raise VariantError(f"unknown variant modifier '@{mod}' in '{model_id}' (supported: @scale=<c>)")
        try:
            scale = float(value)
        except ValueError as exc:
            raise VariantError(f"'{model_id}': scale must be a number, got '{value}'") from exc
        if not scale > 0 or scale != scale or scale == float("inf"):
            raise VariantError(f"'{model_id}': scale must be a positive finite number")
    if head.startswith(ZOO_PREFIX):
        run_id = head[len(ZOO_PREFIX):]
        return Variant(model_id=model_id, base_id=zoo_entry(run_id)["base_id"], logit_scale=scale, adapter=run_id)
    if not head:
        raise VariantError(f"'{model_id}' names no checkpoint")
    return Variant(model_id=model_id, base_id=head, logit_scale=scale)


# --------------------------------------------------------------------------- zoo
def zoo_dir() -> Path:
    from rtp.paths import work_dir

    return work_dir() / "zoo"


def manifest_path() -> Path:
    return zoo_dir() / "manifest.json"


def read_manifest() -> dict[str, dict[str, Any]]:
    from rtp import io

    path = manifest_path()
    return io.read_json(path) if path.exists() else {}


def zoo_entry(run_id: str) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id or ""):
        raise VariantError(f"bad zoo run id '{run_id}'")
    entry = read_manifest().get(run_id)
    if entry is None:
        raise VariantError(f"zoo run '{run_id}' is not in {manifest_path()}; train it first or check RTP_WORK_DIR")
    return entry


def adapter_dir(run_id: str) -> Path:
    return zoo_dir() / run_id / "adapter"


def adapter_sha256(path: Path) -> str:
    """One digest over the adapter's config and weights."""
    h = hashlib.sha256()
    for name in ADAPTER_FILES:
        f = Path(path) / name
        if not f.exists():
            raise VariantError(f"adapter file missing: {f}")
        h.update(name.encode() + b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def register_adapter(run_id: str, base_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add (or replace) a finished adapter at ``zoo/<run_id>/adapter`` in the manifest. ``base_id`` must be a
    registered checkpoint, never a variant: an adapter is trained on exactly one published model."""
    from rtp import io, registry

    if not _RUN_ID.fullmatch(run_id or ""):
        raise VariantError(f"bad zoo run id '{run_id}'")
    if is_variant(base_id) or registry.MODELS.get(base_id) is None:
        raise VariantError(f"adapter base '{base_id}' must be a checkpoint registered in rtp/registry.py")
    entry = {"base_id": base_id, "adapter_sha256": adapter_sha256(adapter_dir(run_id)), "meta": dict(meta or {})}
    manifest = read_manifest()
    manifest[run_id] = entry
    io.write_json(manifest_path(), manifest)
    return entry


def verified_adapter_dir(run_id: str) -> Path:
    """The adapter's directory, after checking its files against the sha256 in the manifest."""
    entry, path = zoo_entry(run_id), adapter_dir(run_id)
    found = adapter_sha256(path)
    if found != entry["adapter_sha256"]:
        raise VariantError(f"adapter '{run_id}' changed on disk (sha256 {found[:12]} != manifest "
                           f"{entry['adapter_sha256'][:12]}); retrain it under a new run id or re-register it")
    return path
