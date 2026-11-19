"""CPU analysis of finished ladder stages: per-model tables and stage-pair comparisons.

This is a rebuildable VIEW, not a stage: it reads ``scores.parquet`` of the finished ladder stages of a model,
calls the estimators (rtp.scale) and writes tables and a summary under ``runs/<config>/analysis/``. It is
recomputed whenever its inputs (the stages' hashes and their completion records), this module or the
estimator code, or the ladder settings change, and it never touches a model. Changing an estimator
therefore never invalidates a GPU stage.

    analysis/<model slug>/            units, curves, pulls, entities, fictional, raw, ... (.parquet);
                                      checks.json; summary.json; summary.md; ctx_* tables (four-option arm)
    analysis/_pairs/<A>__<B>/         scale, null, residuals, shape (.parquet); summary.json; summary.md
    analysis_test/...                 the same for the sealed test split (only with unseal)

Models are compared here, on stored per-model tables, and only on the tiers BOTH finished, measured with the
same items, the same readout code and the same letter form.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_ladder_items
from rtp.data.ladder_items import type_steps
from rtp.data.splits import check_split_access
from rtp.paths import model_slug, run_dir, stage_dir

log = logging.getLogger(__name__)

BINARY_TIERS = ("core", "ext")  # two-option tiers, analysed together
CTX_TIER = "ctx"  # four-option arm, analysed on its own and compared with the two-option result
TABLES = ("units", "curves", "pulls", "entities", "fictional", "raw", "anchor_clean", "pairs", "cells")
PAIR_TABLES = ("scale", "null", "residuals", "shape")
N_BOOT = 2000


def analysis_dir(cfg: Config, split: str = "dev") -> Path:
    return run_dir(cfg.name) / ("analysis" if split == "dev" else "analysis_test")


def _stage_name(tier: str, split: str) -> str:
    return f"ladder_{tier}" if split == "dev" else f"ladder_{tier}_test"


def finished_tiers(cfg: Config, model_id: str, split: str) -> dict[str, dict[str, Any]]:
    """Tiers whose ladder stage is finished AND current (built by the present code, items and settings).
    ``done_sha`` fingerprints the completion record, which changes when a stage is re-measured under the same
    identity (``--force``), so an analysis never keeps the numbers of scores that were replaced."""
    out: dict[str, dict[str, Any]] = {}
    for tier in (*BINARY_TIERS, CTX_TIER):
        sdir = stage_dir(cfg.name, model_id, _stage_name(tier, split))
        if not io.is_done(sdir):
            continue
        meta = io.read_json(sdir / io.STAGE_META)
        expected = stages.stage_hash(cfg, "ladder", model_id, split, meta.get("prompt_mode", "auto"), tier=tier)
        if meta.get("stage_hash") != expected:
            log.warning("%s: ladder-%s is stale (other code, items or settings); left out of the analysis", model_id,
                        tier)
            continue
        out[tier] = {"dir": sdir, "stage_hash": expected, "code_digest": meta.get("code_digest"),
                     "done_sha": io.file_sha256(sdir / io.DONE_FILE)[:16]}
    return out


def _code_digest() -> str:
    """This module and everything it imports, the estimators (rtp.scale) included."""
    from rtp.provenance import code_digest

    return code_digest("rtp.analyze")


def _load(cfg: Config, tiers: dict[str, dict[str, Any]], names: tuple[str, ...], split: str, unseal: bool):
    items: list[dict[str, Any]] = []
    scores = []
    for tier in names:
        if tier in tiers:
            items.extend(load_ladder_items(cfg, tier, split, unseal))
            scores.append(pd.read_parquet(tiers[tier]["dir"] / "scores.parquet"))
    return items, (pd.concat(scores, ignore_index=True) if scores else pd.DataFrame())


def _run_scale(cfg: Config, tiers: dict[str, dict[str, Any]], names: tuple[str, ...], split: str, unseal: bool,
               form: str | None = None) -> dict[str, Any]:
    from rtp import scale

    items, scores = _load(cfg, tiers, names, split, unseal)
    return scale.analyze_model(items, scores, type_steps(cfg), form=form)


def _write_tables(out_dir: Path, res: dict[str, Any], prefix: str = "") -> None:
    for name in TABLES:
        table = res.get(name)
        if isinstance(table, pd.DataFrame):
            io.write_parquet(out_dir / f"{prefix}{name}.parquet", table)
    io.write_json(out_dir / f"{prefix}checks.json", _jsonable(res.get("checks", {})))


def _read_tables(out_dir: Path, prefix: str = "") -> dict[str, Any]:
    res: dict[str, Any] = {}
    for name in TABLES:
        path = out_dir / f"{prefix}{name}.parquet"
        if path.exists():
            res[name] = pd.read_parquet(path)
    res["checks"] = io.read_json(out_dir / f"{prefix}checks.json")
    return res


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if obj != obj else round(float(obj), 6)
    return obj


def _r(x: Any, nd: int = 4) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else round(x, nd)


def _form(res: dict[str, Any]) -> str | None:
    units = res.get("units")
    return str(units["form"].iloc[0]) if isinstance(units, pd.DataFrame) and len(units) else None


def headline(res: dict[str, Any]) -> dict[str, Any]:
    """The few numbers of a model's analysis that the scorecard shows, per stratum and evidence type:
    the pooled anchor curve's raw value at the top step (``A_top``, logits: the model's decisiveness on that
    ladder), whether that curve rises at all (``curve_rises``) and how many category curves do not
    (``n_flat_category_curves``), the median name effect of real brands (``delta_median``, logits), the median
    preference level in natural units over the brands that have one (``theta_median``), and the shares of
    brands with a negative name effect, beyond the ladder (censored) or on a flat curve."""
    out: dict[str, Any] = {}
    curves, ent = res.get("curves"), res.get("entities")
    if isinstance(curves, pd.DataFrame) and len(curves):
        for (stratum, etype), g in curves.groupby(["stratum", "evidence_type"], sort=True):
            cell = out.setdefault(f"{stratum}/{etype}", {})
            pooled = g[g["category"] == "(all)"].sort_values("level")
            if len(pooled):
                cell["A_top"], cell["x_top"] = _r(pooled["A_raw"].iloc[-1]), _r(pooled["x"].iloc[-1])
                cell["curve_rises"] = bool(pooled["A_iso"].max() > 0)
            per = g[g["category"] != "(all)"].groupby("category")["A_iso"].max()
            cell["n_category_curves"] = int(len(per))
            cell["n_flat_category_curves"] = int((~(per > 0)).sum())
    if isinstance(ent, pd.DataFrame) and len(ent):
        for (stratum, etype), g in ent.groupby(["stratum", "evidence_type"], sort=True):
            cell = out.setdefault(f"{stratum}/{etype}", {})
            cell["n_brands"] = int(len(g))
            cell["delta_median"] = _r(g["delta"].median())
            cell["share_delta_negative"] = _r((g["delta"] < 0).mean())
            have = g[g["status"].isin(["ok", "censored"])]
            if len(have):
                cell["theta_median"] = _r(have["theta"].median())
                cell["share_censored"] = _r(have["censored"].astype(bool).mean())
            cell["share_on_flat_curve"] = _r(g["status"].eq("flat_curve").mean())
    return out


def compact_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """The checks without the long per-name table of the anchor check (``rows``): what the scorecard shows.
    The full tables stay in ``checks.json``."""
    out: dict[str, Any] = {}
    for name, block in checks.items():
        out[name] = {k: v for k, v in block.items() if k != "rows"} if isinstance(block, dict) else block
    return out


def choice_set_agreement(binary: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Two options vs four (target, control and two real competitors): do the name effects and preference
    levels of the same brands agree?

    The two arms measure a brand against different sets of fictional controls (all of them in the two-option
    arm, a few in the four-option arm), so their raw name effects have different zero points: each is the
    brand's effect minus the mean pull of ITS controls. Before comparing, the four-option name effect is
    re-centred on the average fictional name of its arm by adding back the mean pull of the controls it was
    measured against (the pulls come from the arm's own d = 0 anchors, where every pair of names appears).
    The two-option name effect is already centred: its controls are all names, whose pulls sum to zero.
    The re-centred effect is then converted with the four-option arm's own anchor curve.
    """
    from rtp.scale.theta import convert

    a, cells, pulls, curves = binary.get("entities"), ctx.get("cells"), ctx.get("pulls"), ctx.get("curves")
    if not all(isinstance(x, pd.DataFrame) and len(x) for x in (a, cells, pulls, curves)):
        return {}
    keys = ["arm", "category", "stratum", "evidence_type"]
    m = cells.merge(pulls[[*keys, "name", "pull"]].rename(columns={"name": "control"}), on=[*keys, "control"],
                    how="left")
    per = m.groupby([*keys, "entity"], sort=False).agg(delta_raw=("delta", "mean"), pull=("pull", "mean"),
                                                        n_controls=("control", "nunique"),
                                                        n_missing=("pull", lambda x: int(x.isna().sum()))).reset_index()
    per["delta"] = per["delta_raw"] + per["pull"]  # NaN where a control's pull is not identified
    four = convert(per, "delta", curves)
    two = a[a["arm"] == "pair"]
    on = ["category", "entity", "stratum", "evidence_type"]
    both = two[[*on, "delta", "theta", "status"]].merge(four[[*on, "delta", "theta", "status", "pull", "n_missing"]],
                                                        on=on, suffixes=("_2", "_4"))
    out: dict[str, Any] = {}
    for (stratum, etype), g in both.groupby(["stratum", "evidence_type"], sort=True):
        cell: dict[str, Any] = {"n_brands": int(len(g)), "n_brands_without_pull": int((g["n_missing"] > 0).sum()),
                                "mean_abs_pull_added": _r(g["pull"].abs().mean())}
        for col in ("delta", "theta"):
            x, y = g[f"{col}_2"], g[f"{col}_4"]
            ok = x.notna() & y.notna()
            if col == "theta":  # a level at the top of a ladder is a bound, not a value
                ok &= g["status_2"].eq("ok") & g["status_4"].eq("ok")
            cell[f"n_{col}"] = int(ok.sum())
            if ok.sum() >= 3:
                cell[f"{col}_corr"] = _r(np.corrcoef(x[ok], y[ok])[0, 1]) if x[ok].std() > 0 and y[ok].std() > 0 else None
                cell[f"{col}_median_diff_4_minus_2"] = _r((y[ok] - x[ok]).median())
        out[f"{stratum}/{etype}"] = cell
    return out


def analyze_model(cfg: Config, model_id: str, split: str = "dev", unseal: bool = False,
                  force: bool = False) -> dict[str, Any] | None:
    """Analyse one model. Returns the summary shown in the scorecard, or None when the core tier is not
    finished yet."""
    check_split_access(split, unseal)
    tiers = finished_tiers(cfg, model_id, split)
    if "core" not in tiers:
        return None
    meta = {"model_id": model_id, "split": split,
            "inputs": {t: [v["stage_hash"], v["done_sha"]] for t, v in sorted(tiers.items())},
            "code": _code_digest(), "steps": type_steps(cfg), "ladder_version": cfg.ladder.version,
            "readout_code": {t: v["code_digest"] for t, v in sorted(tiers.items())}}
    out_dir = analysis_dir(cfg, split) / model_slug(model_id)
    meta_path = out_dir / "analysis_meta.json"
    if not force and meta_path.exists() and io.read_json(meta_path) == _jsonable(meta) \
            and (out_dir / "summary.json").exists():
        return io.read_json(out_dir / "summary.json")
    res = _run_scale(cfg, tiers, BINARY_TIERS, split, unseal)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(out_dir, res)
    summary: dict[str, Any] = {"tiers": sorted(tiers), "letter_form": _form(res),
                               "checks": _jsonable(compact_checks(res.get("checks", {}))), "headline": headline(res)}
    if CTX_TIER in tiers:  # the same letter form as the two-option arm: one readout per model
        ctx = _run_scale(cfg, tiers, (CTX_TIER,), split, unseal, form=_form(res))
        _write_tables(out_dir, ctx, prefix="ctx_")
        summary["choice_set"] = choice_set_agreement(res, ctx)
    io.write_json(out_dir / "summary.json", _jsonable(summary))
    io.atomic_write_text(out_dir / "summary.md", model_markdown(model_id, split, summary))
    io.write_json(meta_path, _jsonable(meta))
    return io.read_json(out_dir / "summary.json")


def compare(cfg: Config, model_a: str, model_b: str, split: str = "dev", unseal: bool = False,
            force: bool = False) -> dict[str, Any] | None:
    """Stage pair A -> B: scale multipliers, the pooled null statistics, content residuals. None when one of
    the two models has no finished analysis yet.

    The comparison uses the two-option tiers BOTH models finished. A model that has more (say ``ext`` while
    the other has only ``core``) is re-analysed on the shared tiers for this comparison: an anchor curve from
    core + ext is a different estimate from a core-only one, so the stored tables cannot simply be filtered.
    """
    from rtp import scale

    check_split_access(split, unseal)
    if analyze_model(cfg, model_a, split, unseal) is None or analyze_model(cfg, model_b, split, unseal) is None:
        return None
    base = analysis_dir(cfg, split)
    side = {}
    for model in (model_a, model_b):
        tiers = finished_tiers(cfg, model, split)
        side[model] = {"tiers": tiers, "dir": base / model_slug(model),
                       "meta": io.read_json(base / model_slug(model) / "analysis_meta.json")}
    shared = tuple(t for t in BINARY_TIERS if all(t in side[m]["tiers"] for m in (model_a, model_b)))
    digests = [{t: side[m]["tiers"][t]["code_digest"] for t in shared} for m in (model_a, model_b)]
    if digests[0] != digests[1] or side[model_a]["meta"]["steps"] != side[model_b]["meta"]["steps"]:
        raise ValueError(f"'{model_a}' and '{model_b}' were measured with different readout code or ladder settings "
                         f"({digests[0]} vs {digests[1]}); re-measure one of them")
    out_dir = base / "_pairs" / f"{model_slug(model_a)}__{model_slug(model_b)}"
    meta = {"a": side[model_a]["meta"], "b": side[model_b]["meta"], "shared": list(shared), "n_boot": N_BOOT,
            "seed": cfg.seed, "code": _code_digest()}
    meta_path = out_dir / "analysis_meta.json"
    if not force and meta_path.exists() and io.read_json(meta_path) == _jsonable(meta) \
            and (out_dir / "summary.json").exists():
        return io.read_json(out_dir / "summary.json")
    tables, dropped = {}, {}
    for model in (model_a, model_b):
        own = tuple(t for t in BINARY_TIERS if t in side[model]["tiers"])
        dropped[model] = [t for t in own if t not in shared]
        tables[model] = _read_tables(side[model]["dir"]) if own == shared else \
            _run_scale(cfg, side[model]["tiers"], shared, split, unseal)
    a, b = tables[model_a], tables[model_b]
    if _form(a) != _form(b):
        raise ValueError(f"'{model_a}' is read in letter form '{_form(a)}' and '{model_b}' in '{_form(b)}': the two "
                         f"readouts are logits of different tokens and are not compared (see DECISIONS.md)")
    res = scale.compare_models(a, b, n_boot=N_BOOT, seed=cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in PAIR_TABLES:
        if isinstance(res.get(name), pd.DataFrame):
            io.write_parquet(out_dir / f"{name}.parquet", res[name])
    summary = {"tiers_compared": list(shared), "tiers_left_out": {m: v for m, v in dropped.items() if v},
               "letter_form": _form(a), "summary": _jsonable(res.get("summary", {})), "identity": identity(res)}
    io.write_json(out_dir / "summary.json", _jsonable(summary))
    io.atomic_write_text(out_dir / "summary.md", pair_markdown(model_a, model_b, split, summary))
    io.write_json(meta_path, _jsonable(meta))
    return io.read_json(out_dir / "summary.json")


def identity(res: dict[str, Any]) -> dict[str, Any]:
    """What a pure scale change must leave alone. ``c_min`` / ``c_max``: range of the pooled scale multipliers
    over the curves that rise in both models (``n_curves``). ``max_abs_theta_diff``: the largest change of a
    brand's preference level among the ``n_brand_cells`` cells that have a level in both models and are not
    censored. ``n_flag_changes``: cells whose censoring flag, or whether they have a level at all, differs
    between the two models. For a ``@scale=c`` variant the multipliers all equal c, no level moves and no
    flag changes; a result with ``n_curves`` or ``n_brand_cells`` of 0 proves nothing."""
    out: dict[str, Any] = {"c_min": None, "c_max": None, "n_curves": 0, "max_abs_theta_diff": None,
                           "n_brand_cells": 0, "n_flag_changes": None}
    sc = res.get("scale")
    if isinstance(sc, pd.DataFrame) and len(sc):
        vals = sc.loc[(sc["category"] == "(all)") & sc["valid"].astype(bool), "c"].dropna()
        out["n_curves"] = int(len(vals))
        if len(vals):
            out["c_min"], out["c_max"] = float(vals.min()), float(vals.max())
    rs = res.get("residuals")
    if isinstance(rs, pd.DataFrame) and len(rs):
        d = rs.loc[rs["d_theta_valid"].astype(bool), "d_theta"].abs()
        out["n_brand_cells"] = int(len(d))
        out["max_abs_theta_diff"] = float(d.max()) if len(d) else None
        out["n_flag_changes"] = int(((rs["censored_a"].astype(bool) != rs["censored_b"].astype(bool))
                                     | (rs["theta_a"].isna() != rs["theta_b"].isna())).sum())
    return out


# --------------------------------------------------------------------------- summaries
def _tree(obj: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                lines.append(f"{pad}- {k}:")
                lines.extend(_tree(v, indent + 1))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"{pad}- {k}:")
                lines.extend(f"{pad}  - " + ", ".join(f"{a}={b}" for a, b in row.items()) for row in v)
            else:
                lines.append(f"{pad}- {k}: {v}")
    return lines


def model_markdown(model_id: str, split: str, summary: dict[str, Any]) -> str:
    lines = [f"# Analysis: {model_id} ({split})", "",
             f"Tiers analysed: {', '.join(summary['tiers'])}; letter form `{summary.get('letter_form')}`. Units: logits "
             f"for name effects (delta) and anchor curves (A); natural evidence units for preference levels (theta: "
             f"stars, relative price, log review ratio). A check's `pass` is None when the loaded tiers hold no "
             f"cell that carries it.",
             "", "## Instrument checks", ""] + _tree(summary["checks"])
    lines += ["", "## Headline numbers per stratum / evidence type", ""] + _tree(summary["headline"])
    if summary.get("choice_set"):
        lines += ["", "## Two options vs four (choice-set robustness; four-option effects re-centred, see "
                      "rtp.analyze.choice_set_agreement)", ""] + _tree(summary["choice_set"])
    return "\n".join(lines) + "\n"


def pair_markdown(model_a: str, model_b: str, split: str, summary: dict[str, Any]) -> str:
    lines = [f"# Stage pair: {model_a} -> {model_b} ({split})", "",
             "c = scale multiplier (ratio of the two models' anchor curves, fictional names only); m = ratio of the "
             "pooled name effects of real brands; T1 = log m - log c (0 under 'uniform sharpening'). The primary "
             "interval of T1 is T1_t_lo..T1_t_hi (t over per-category values, centre T1_cat_mean).", "",
             f"Tiers compared (finished by both models): {', '.join(summary['tiers_compared'])}"]
    if summary.get("tiers_left_out"):
        lines.append(f"Tiers left out because only one model has them: {summary['tiers_left_out']}")
    lines += ["", "## Pure-scale identity", ""] + _tree(summary["identity"])
    lines += ["", "## Summary", ""] + _tree(summary["summary"])
    return "\n".join(lines) + "\n"


def measured_models(cfg: Config, split: str = "dev") -> list[str]:
    """Every model with a finished core ladder stage in this run."""
    out = []
    for done in sorted(run_dir(cfg.name).glob(f"*/{_stage_name('core', split)}/{io.DONE_FILE}")):
        mid = io.read_json(done.parent / io.STAGE_META).get("model_id")
        if mid:
            out.append(mid)
    return out


def run_analyze(cfg: Config, models: list[str] | None = None, pairs: list[tuple[str, str]] | None = None,
                split: str = "dev", unseal: bool = False) -> None:
    check_split_access(split, unseal)
    for m in models or measured_models(cfg, split):
        res = analyze_model(cfg, m, split, unseal, force=True)
        path = analysis_dir(cfg, split) / model_slug(m) / "summary.md"
        print(f"{m}: " + ("no finished core ladder stage" if res is None else str(path)))
    for a, b in pairs or []:
        res = compare(cfg, a, b, split, unseal, force=True)
        path = analysis_dir(cfg, split) / "_pairs" / f"{model_slug(a)}__{model_slug(b)}" / "summary.md"
        print(f"{a} -> {b}: " + ("one of the two models is not analysed yet" if res is None else str(path)))
