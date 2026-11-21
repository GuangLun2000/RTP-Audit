"""Are the built item files actually there?

`build-data` skips when the data directory holds a completion record for the current data hash. On Colab with
a full Google Drive, cycle C0 wrote every item file to the local Drive cache, but only the small files reached
Drive: `dev/ladder_core.jsonl` (37 MB) and `dev/ladder_ext.jsonl` were lost, the manifest and the completion
record were not. The next session trusted the record, skipped the build and failed on the missing files.

This module checks the files the manifest lists and rebuilds the data when one is missing. It lives OUTSIDE
`rtp.data` on purpose: the data hash covers the generator code (`rtp.data.build` and what it imports), and a
check of the files must not change that hash. A rebuild with the same seed, settings and code writes the same
bytes, so the stages measured on the lost files stay valid (their identity is the files' sha256).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rtp import io
from rtp.config import Config
from rtp.paths import data_dir

log = logging.getLogger(__name__)

MANIFEST = "data_manifest.json"


class DataFilesMissing(RuntimeError):
    pass


def missing_files(cfg: Config) -> list[str]:
    """Files listed in the manifest of the built data that are absent, or empty although the manifest counts
    records in them (relative to the data dir; a split without categories legitimately writes an empty file).
    Empty when the data are not built at all: that case is `build-data`'s own."""
    ddir = data_dir(cfg.name)
    if not io.is_done(ddir) or not (ddir / MANIFEST).exists():
        return []
    manifest = io.read_json(ddir / MANIFEST)
    counts = manifest.get("n_records") or {}
    return sorted(rel for rel in manifest.get("sha256") or {}
                  if not (ddir / rel).exists() or ((ddir / rel).stat().st_size == 0 and counts.get(rel, 1) > 0))


def ensure_data(cfg: Config, force: bool = False) -> Path:
    """`build_data`, then rebuild once if a listed file is missing; raise if it is still missing."""
    from rtp.data.build import build_data

    ddir = build_data(cfg, force=force)
    lost = missing_files(cfg)
    if lost:
        log.warning("the data record at %s is complete but %d item file(s) are missing: %s. Rebuilding (same seed, "
                    "settings and code: the same bytes, so finished stages stay valid). A full Google Drive drops "
                    "large files silently: check the free space.", ddir, len(lost), ", ".join(lost))
        ddir = build_data(cfg, force=True)
        lost = missing_files(cfg)
        if lost:
            raise DataFilesMissing(f"item file(s) still missing after a rebuild at {ddir}: {', '.join(lost)}. The "
                                   f"results disk does not keep what is written to it (full Google Drive?).")
    return ddir
