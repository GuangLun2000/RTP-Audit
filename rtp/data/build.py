"""``build-data``: generate catalogue, choice items and probe items once per config (model-independent).

Output (``$RTP_WORK_DIR/runs/{config}/data/``)::

    catalogue.jsonl                     all categories, with a split field
    dev/choice_items.jsonl  dev/probe_items.jsonl
    test/choice_items.jsonl test/probe_items.jsonl      sealed: read only with --unseal
    cell_counts.csv  data_report.md  data_manifest.json  config.yaml  _DONE

Every file is a pure function of (seed, data config, seed files): two runs with the same
inputs produce byte-identical files.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from rtp import io
from rtp.config import Config
from rtp.data import schema
from rtp.data.catalogue import (all_real_names, catalogue_records, load_category_seeds, load_natural_negatives)
from rtp.data.choice_items import build_choice_items, cells
from rtp.data.placebo import load_placebo_file, validate_placebos
from rtp.data.probe_items import ATTRIBUTES, build_probe_items
from rtp.data.splits import SPLITS, check_split_access
from rtp.paths import data_dir

log = logging.getLogger(__name__)

CATALOGUE = "catalogue.jsonl"
CHOICE = "choice_items.jsonl"
PROBE = "probe_items.jsonl"


class DataNotBuiltError(RuntimeError):
    pass


def _categories(cfg: Config, available: list[str]) -> list[str]:
    cats = cfg.data.categories if cfg.data.categories is not None else available
    missing = [c for c in cats if c not in available]
    if missing:
        raise ValueError(f"categories not in {cfg.data.categories_file}: {missing}")
    bad_test = [c for c in cfg.data.test_categories if c not in cats]
    if bad_test:
        raise ValueError(f"test_categories not among the run's categories: {bad_test}")
    return list(cats)


def generate(cfg: Config) -> dict[str, Any]:
    """Pure generation step (no files written). Returns catalogue, items and probes."""
    d = cfg.data
    seeds = load_category_seeds(cfg.resolve(d.categories_file))
    nn = load_natural_negatives(cfg.resolve(d.natural_negatives_file))
    placebos = load_placebo_file(cfg.resolve(d.placebo_file))
    placebo_stats = validate_placebos(placebos, all_real_names(seeds, nn))
    cats = _categories(cfg, list(seeds.categories))
    choice_pool = {c: [e["name"] for e in placebos["choice"].get(c, [])] for c in cats}
    probe_pool = {c: [e["name"] for e in placebos["probe"].get(c, [])] for c in cats}
    catalogue = catalogue_records(cats, seeds, nn, choice_pool, d.n_real_per_category, d.n_placebo_per_category,
                                  d.test_categories)
    items = build_choice_items(cfg, seeds, catalogue)
    reals = {c: [r["entity"] for r in catalogue if r["category"] == c and not r["is_placebo"]] for c in cats}
    probes = build_probe_items(cfg, cats, probe_pool, reals)
    schema.validate_catalogue(catalogue)
    schema.validate_choice_items(items)
    schema.validate_probe_items(probes)
    review = [c for c in cats if seeds.categories[c].needs_review]
    return {"catalogue": catalogue, "choice": items, "probe": probes, "categories": cats,
            "needs_review": review, "placebo_stats": placebo_stats}


def cell_counts(items: list[dict[str, Any]]) -> pd.DataFrame:
    cnt = Counter((it["split"], it["category"], it["evidence_level"], it["handicap"],
                   "placebo" if it["target_is_placebo"] else "real") for it in items)
    rows = [dict(zip(("split", "category", "evidence_level", "handicap", "name_condition"), k), n_items=v)
            for k, v in sorted(cnt.items())]
    return pd.DataFrame(rows, columns=["split", "category", "evidence_level", "handicap", "name_condition", "n_items"])


def data_report(cfg: Config, out: dict[str, Any], counts: pd.DataFrame) -> str:
    items, probes, cat = out["choice"], out["probe"], out["catalogue"]
    lines = [f"# Data report: {cfg.name}", "",
             f"- seed: {cfg.seed}; data hash: `{cfg.data_hash}`",
             f"- categories: {', '.join(out['categories'])}",
             f"- test (sealed) categories: {', '.join(cfg.data.test_categories) or '(none)'}",
             f"- choice items: {len(items)} ({len({i['pair_id'] for i in items})} pairs x 2 names x 4 rotations)",
             f"- probe items: {len(probes)} ({len({(p['category'], p['entity']) for p in probes})} placebo entities)",
             f"- cells per entity: {len(cells(cfg.data.evidence_levels, cfg.data.handicaps))} "
             f"(handicap only at evidence >= 3), cards per cell: {cfg.data.cards_per_cell}",
             f"- placebo names still needing a web check: {out['placebo_stats']['n_needs_web_check']}"
             f" / {out['placebo_stats']['n_names']}"]
    if out["needs_review"]:
        lines.append(f"- **categories.yaml entries still marked needs_review: {', '.join(out['needs_review'])}**")
    lines += ["", "## Catalogue", "", "| split | category | real | placebo | natural negatives |", "|---|---|---|---|---|"]
    for c in out["categories"]:
        rows = [r for r in cat if r["category"] == c]
        reals = [r["entity"] for r in rows if not r["is_placebo"]]
        pls = [r["entity"] for r in rows if r["is_placebo"]]
        nn = [f"{r['entity']} ({r['natural_negative']})" for r in rows if r["natural_negative"]]
        lines.append(f"| {rows[0]['split']} | {c} | {', '.join(reals)} | {', '.join(pls)} | {', '.join(nn)} |")
    lines += ["", "## Choice items per cell (split x category x evidence x handicap x name)", ""]
    piv = counts.pivot_table(index=["split", "category", "evidence_level", "handicap"], columns="name_condition",
                             values="n_items", aggfunc="sum", fill_value=0).reset_index()
    lines.append("| " + " | ".join(piv.columns) + " |")
    lines.append("|" + "---|" * len(piv.columns))
    for row in piv.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Probe items: attribute level counts (entities)", ""]
    ents = {(p["split"], p["category"], p["entity"]): p["attributes"] for p in probes}
    for a in ATTRIBUTES:
        by = Counter((k[0], v[a]) for k, v in ents.items())
        lines.append(f"- {a}: " + ", ".join(f"{s}/level {lv}: {n}" for (s, lv), n in sorted(by.items())))
    return "\n".join(lines) + "\n"


SEED_SNAPSHOT_DIR = "seeds"


def data_config_yaml(cfg: Config) -> str:
    """The parts of the run config that determine the data (no default-filled measure sections)."""
    import yaml

    return yaml.safe_dump({"name": cfg.name, "seed": cfg.seed, "data": cfg.to_dict()["data"],
                           "seed_file_digests": cfg.seed_file_digests()}, sort_keys=False, allow_unicode=True)


def build_data(cfg: Config, force: bool = False) -> Path:
    ddir = data_dir(cfg.name)
    if io.finished_stage(ddir, cfg.data_hash, force):
        log.info("data already built at %s (hash %s); skipping", ddir, cfg.data_hash)
        return ddir
    io.prepare_stage_dir(ddir, data_config_yaml(cfg), cfg.data_hash, force=force)
    out = generate(cfg)
    files: dict[str, int] = {}
    files[CATALOGUE] = io.write_jsonl(ddir / CATALOGUE, out["catalogue"])
    for split in SPLITS:
        for name, recs in ((CHOICE, out["choice"]), (PROBE, out["probe"])):
            rel = f"{split}/{name}"
            files[rel] = io.write_jsonl(ddir / rel, [r for r in recs if r["split"] == split])
    # frozen copy of the seed files the data were built from (open-ended reads these)
    d = cfg.data
    snapshot_sha: dict[str, str] = {}
    for src in (d.categories_file, d.placebo_file, d.natural_negatives_file):
        path = cfg.resolve(src)
        io.atomic_write_bytes(ddir / SEED_SNAPSHOT_DIR / path.name, path.read_bytes())
        snapshot_sha[path.name] = io.file_sha256(ddir / SEED_SNAPSHOT_DIR / path.name)
    counts = cell_counts(out["choice"])
    io.atomic_write_text(ddir / "cell_counts.csv", counts.to_csv(index=False))
    report = data_report(cfg, out, counts)
    io.atomic_write_text(ddir / "data_report.md", report)
    manifest = {"config": cfg.name, "seed": cfg.seed, "data_hash": cfg.data_hash,
                "seed_file_digests": cfg.seed_file_digests(),
                "test_categories": list(cfg.data.test_categories), "categories": out["categories"],
                "n_records": files, "sha256": {k: io.file_sha256(ddir / k) for k in sorted(files)},
                "seed_snapshot_sha256": snapshot_sha,
                "needs_review": out["needs_review"], "placebo_stats": out["placebo_stats"]}
    io.write_json(ddir / "data_manifest.json", manifest)
    io.atomic_write_text(ddir / "summary.md", report)
    io.mark_done(ddir, {"data_hash": cfg.data_hash})
    log.info("built data in %s: %s", ddir, files)
    return ddir


# --------------------------------------------------------------------------- sealed readers
def require_built(cfg: Config) -> Path:
    ddir = data_dir(cfg.name)
    if not io.is_done(ddir):
        raise DataNotBuiltError(f"no finished data at {ddir}; run `python -m rtp.cli build-data --config ...` first")
    if io.stored_stage_hash(ddir) != cfg.data_hash:
        raise io.StageConfigMismatch(
            f"data at {ddir} were built from a different data config or different seed-file content "
            f"(categories.yaml / placebo_names.yaml / natural_negatives.csv). Rebuild with `build-data --force` "
            f"(model stages built on the old data will then refuse to resume; rerun them with --force).")
    return ddir


def load_seed_snapshot(cfg: Config):
    """(CategorySeeds, natural negatives) from the frozen copy stored with the built data."""
    ddir = require_built(cfg) / SEED_SNAPSHOT_DIR
    seeds = load_category_seeds(ddir / cfg.resolve(cfg.data.categories_file).name)
    nn = load_natural_negatives(ddir / cfg.resolve(cfg.data.natural_negatives_file).name)
    return seeds, nn


def load_choice_items(cfg: Config, split: str, unseal: bool = False) -> list[dict[str, Any]]:
    check_split_access(split, unseal)
    return io.read_jsonl(require_built(cfg) / split / CHOICE)


def load_probe_items(cfg: Config, split: str, unseal: bool = False) -> list[dict[str, Any]]:
    check_split_access(split, unseal)
    return io.read_jsonl(require_built(cfg) / split / PROBE)


def load_catalogue(cfg: Config, split: str, unseal: bool = False) -> list[dict[str, Any]]:
    check_split_access(split, unseal)
    return [r for r in io.read_jsonl(require_built(cfg) / CATALOGUE) if r["split"] == split]
