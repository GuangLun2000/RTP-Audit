"""Stage comparison A -> B: one scale multiplier, and what is left once it is taken out.

Null model ("post-training = uniform sharpening"): every logit difference of B is c times that of A, so
``A_B = c * A_A`` on the anchors and ``Delta_B(e) = c * Delta_A(e)`` for every brand, and no preference
level moves. Three numbers per (stratum, evidence_type):

* ``c``: the scale multiplier, from fictional-vs-fictional items ONLY (no real name enters it);
* ``m``: the multiplier of the name effects, pooled over brands;
* ``T1 = log m - log c``: the pooled content shift beyond scale, 0 under the null.

``T2`` asks whether the brand-level residuals ``r_e = Delta_B(e) - c * Delta_A(e)`` vary more than their
measurement error. A single pair of models gives no noise band for a training run; ``calibrate`` compares a
statistic with the same statistic on seed-replicate pairs of content-neutral runs.

Uncertainty treats the CATEGORY as the independent unit (fictional names, brands and base values are all
nested in categories). Within a category the brands share the same fictional controls, so brands are not
independent draws; the crossed mode also resamples the fictional names.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from rtp.data.constants import LADDER_TYPES
from rtp.rng import rng_for
from rtp.scale.anchor import ALL, KEYS
from rtp.scale.theta import NO_TYPE, SHARED_ZERO_STRATA, long_theta

log = logging.getLogger(__name__)

TEST_KEYS = ["arm", "stratum", "evidence_type", "cross_stratum"]
ENTITY_ID = ["arm", "category", "entity", "stratum", "evidence_type", "cross_stratum"]
CI = (2.5, 97.5)
MAX_REDRAWS = 100
# Two-sided 95% Student t quantiles t(df, 0.975), df = 1..30.
T_975 = (12.7062047, 4.3026527, 3.1824463, 2.7764451, 2.5705818, 2.4469119, 2.3646243, 2.3060041, 2.2621572,
         2.2281389, 2.2009852, 2.1788128, 2.1603687, 2.1447867, 2.1314495, 2.1199053, 2.1098156, 2.1009220,
         2.0930241, 2.0859634, 2.0796138, 2.0738731, 2.0686576, 2.0638986, 2.0595386, 2.0555294, 2.0518305,
         2.0484071, 2.0452296, 2.0422725)
Z_975 = 1.959963984540054


def t_quantile(df: int) -> float:
    """t(df, 0.975) without scipy: a table up to df = 30, the Cornish-Fisher expansion around the normal
    quantile beyond (error below 1e-5 there)."""
    if df < 1:
        raise ValueError("a t interval needs at least 1 degree of freedom (2 values)")
    if df <= len(T_975):
        return T_975[int(df) - 1]
    z = Z_975
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def _same_design(a: pd.DataFrame, b: pd.DataFrame, on: list[str], what: str) -> pd.DataFrame:
    """Inner view of two tables that must cover the same cells; a difference is an error, never a silent
    restriction to the overlap (the two models were then not measured with the same instrument)."""
    merged = a.merge(b, on=on, how="outer", suffixes=("_a", "_b"), indicator=True)
    odd = merged[merged["_merge"] != "both"]
    if len(odd):
        example = odd.iloc[0][on].to_dict()
        raise ValueError(f"the two models do not cover the same {what}: {len(odd)} cell(s) are in one only "
                         f"(first: {example}, {odd.iloc[0]['_merge']}). Measure both with the same items.")
    return merged.drop(columns="_merge")


def _log_ratio(num: Any, den: Any) -> np.ndarray:
    """log(num / den) where both are positive, NaN elsewhere."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    ok = (num > 0) & (den > 0)
    return np.where(ok, np.log(np.where(ok, num, 1.0)) - np.log(np.where(ok, den, 1.0)), np.nan)


# --------------------------------------------------------------------------- scale
def scale_multipliers(curves_a: pd.DataFrame, curves_b: pd.DataFrame) -> pd.DataFrame:
    """Scale multiplier of every anchor curve: ``c = sum over levels of A_B / sum over levels of A_A``.

    Estimand: by how much stage B multiplies the logit response to evidence between two made-up names. A
    ratio of sums of the RAW cell means (not a regression slope: no level gets extra weight for sitting far
    out, and no curve shape is assumed). One row per (arm, stratum, evidence_type, category) and a pooled
    row with category "(all)": the same ratio with the sums taken over categories too (equal weight per
    category and level). The composite ladder has its own rows.

    ``valid`` is False, and ``log_c`` NaN, when a sum is not positive (the curve does not rise: a ratio of
    responses means nothing then). Levels that are not measured in both models are left out and counted in
    ``n_levels_dropped``.
    """
    cols = [*KEYS, "level", "A_raw"]
    merged = _same_design(curves_a.loc[curves_a["category"] != ALL, cols],
                          curves_b.loc[curves_b["category"] != ALL, cols], [*KEYS, "level"], "anchor cells")
    ok = merged["A_raw_a"].notna() & merged["A_raw_b"].notna()
    merged["sum_a"] = merged["A_raw_a"].where(ok, 0.0)
    merged["sum_b"] = merged["A_raw_b"].where(ok, 0.0)
    merged["n_levels"] = ok.astype(int)
    merged["n_levels_dropped"] = (~ok).astype(int)
    per = merged.groupby(KEYS, sort=False)[["sum_a", "sum_b", "n_levels", "n_levels_dropped"]].sum().reset_index()
    per["n_categories"] = (per["n_levels"] > 0).astype(int)
    pooled = per.groupby(["arm", "stratum", "evidence_type"], sort=False)[
        ["sum_a", "sum_b", "n_levels", "n_levels_dropped", "n_categories"]].sum().reset_index()
    pooled["category"] = ALL
    out = pd.concat([pooled, per], ignore_index=True)
    out["valid"] = (out["sum_a"] > 0) & (out["sum_b"] > 0) & (out["n_levels"] > 0)
    out["log_c"] = _log_ratio(out["sum_b"], out["sum_a"])
    out["c"] = np.where(out["sum_a"] > 0, out["sum_b"] / out["sum_a"].where(out["sum_a"] > 0, 1.0), np.nan)
    return out[["arm", "stratum", "evidence_type", "category", "sum_a", "sum_b", "c", "log_c", "valid", "n_levels",
                "n_levels_dropped", "n_categories"]]


def shape_ratio(curves_a: pd.DataFrame, curves_b: pd.DataFrame) -> pd.DataFrame:
    """Per level: ``ratio = A_B(level) / A_A(level)`` (raw cell means), and ``ratio_over_c`` = that ratio
    divided by the curve's own multiplier c. Under a pure change of scale every level has the same ratio, so
    ``ratio_over_c`` is 1 along the ladder; a trend shows that the curve changed SHAPE, in which case one
    multiplier per curve is a summary, not a description. Category rows and the pooled "(all)" rows."""
    cols = [*KEYS, "level", "x", "A_raw"]
    merged = _same_design(curves_a[cols], curves_b[cols], [*KEYS, "level"], "anchor cells")
    a, b = merged["A_raw_a"], merged["A_raw_b"]
    ok = a.notna() & b.notna()
    merged["ratio"] = np.where(ok & (a > 0), b / a.where(a > 0, 1.0), np.nan)
    sums = merged.assign(sa=a.where(ok, 0.0), sb=b.where(ok, 0.0)).groupby(KEYS, sort=False)[["sa", "sb"]].transform("sum")
    merged["c"] = np.where(sums["sa"] > 0, sums["sb"] / sums["sa"].where(sums["sa"] > 0, 1.0), np.nan)
    merged["ratio_over_c"] = np.where(merged["c"] > 0, merged["ratio"] / merged["c"].where(merged["c"] > 0, 1.0), np.nan)
    merged = merged.rename(columns={"x_a": "x"})
    return merged[[*KEYS, "level", "x", "A_raw_a", "A_raw_b", "ratio", "c", "ratio_over_c"]]


# --------------------------------------------------------------------------- null model
def null_tests(entities_a: pd.DataFrame, entities_b: pd.DataFrame, mult: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tests of "B = A sharpened by one factor", per (arm, stratum, evidence_type, cross_stratum), and the
    brand-level residuals. Returns ``(tests, residuals)``.

    tests
        ``m`` = sum over brands of Delta_B / sum over brands of Delta_A (a ratio of sums, so brands with a
        name effect near 0 do not blow it up; ``n_negative_a`` counts brands with Delta_A < 0, which pull the
        denominator towards 0). ``T1 = log m - log c``, with the POOLED c of the same stratum and type.
        ``T2`` = variance of r_e over brands minus the mean measurement variance of r_e (se_B^2 + c^2 se_A^2
        from ``delta_se``); ``T2_ratio`` is their quotient. The measurement variance treats the two models'
        errors as independent although they share the fictional controls, so it is an upper bound and T2 a
        conservative excess. The error of c itself is in neither: use the bootstrap and ``calibrate``.
    residuals
        one row per brand: ``r`` = Delta_B - c * Delta_A (logits of B), ``r_se``, ``z`` = r / r_se,
        ``d_theta`` = theta_B - theta_A (evidence units; ``d_theta_valid`` False when either is censored or
        undefined).

    Name stratum: it has no ladder of its own, so its rows use c (and, for theta, the curve) of the
    ATTRIBUTE-ONLY stratum of each type and are marked ``cross_stratum``. Full stratum: one row per type,
    the same name effects against each type's full-card multiplier.
    """
    cols = [*ENTITY_ID, "delta", "delta_se", "theta", "censored", "below_floor"]
    res = _same_design(long_theta(entities_a)[cols], long_theta(entities_b)[cols], ENTITY_ID, "brand cells")
    res["curve_stratum"] = np.where(res["cross_stratum"], "attr", res["stratum"])
    pooled = mult.loc[mult["category"] == ALL, ["arm", "stratum", "evidence_type", "c", "log_c"]]
    res = res.merge(pooled.rename(columns={"stratum": "curve_stratum"}), on=["arm", "curve_stratum", "evidence_type"],
                    how="left")
    res["r"] = res["delta_b"] - res["c"] * res["delta_a"]
    res["r_se"] = np.sqrt(res["delta_se_b"] ** 2 + (res["c"] * res["delta_se_a"]) ** 2)
    res["z"] = res["r"] / res["r_se"].where(res["r_se"] > 0)
    res["d_theta"] = res["theta_b"] - res["theta_a"]
    res["d_theta_valid"] = res["d_theta"].notna() & ~res["censored_a"].astype(bool) & ~res["censored_b"].astype(bool)
    rows = []
    for key, g in res.groupby(TEST_KEYS, sort=False):
        ok = g[g["delta_a"].notna() & g["delta_b"].notna()]
        sum_a, sum_b = float(ok["delta_a"].sum()), float(ok["delta_b"].sum())
        log_m = float(_log_ratio(sum_b, sum_a))
        log_c = float(g["log_c"].iloc[0])
        t2 = ok[ok["r"].notna() & ok["r_se"].notna()]
        var_r = float(t2["r"].var(ddof=1)) if len(t2) > 1 else np.nan
        meas = float((t2["r_se"] ** 2).mean()) if len(t2) else np.nan
        moved = g[g["d_theta_valid"]]
        rows.append({
            **dict(zip(TEST_KEYS, key)), "n_entities": int(len(ok)), "n_entities_dropped": int(len(g) - len(ok)),
            "n_categories": int(ok["category"].nunique()), "n_negative_a": int((ok["delta_a"] < 0).sum()),
            "sum_delta_a": sum_a, "sum_delta_b": sum_b, "m": sum_b / sum_a if sum_a > 0 else np.nan, "log_m": log_m,
            "c": float(g["c"].iloc[0]), "log_c": log_c, "T1": log_m - log_c,
            "T2": var_r - meas, "T2_ratio": var_r / meas if meas and meas > 0 else np.nan, "n_T2": int(len(t2)),
            "mean_d_theta": float(moved["d_theta"].mean()) if len(moved) else np.nan,
            "median_d_theta": float(moved["d_theta"].median()) if len(moved) else np.nan,
            "n_d_theta": int(len(moved)), "n_d_theta_dropped": int(len(g) - len(moved)),
            # cells the ladder cannot tell from 0 in at least one model: their d_theta is within one first step
            "n_d_theta_below_floor": int((moved["below_floor_a"].astype(bool)
                                          | moved["below_floor_b"].astype(bool)).sum())})
    tests = pd.DataFrame(rows)
    residuals = res[[*ENTITY_ID, "delta_a", "delta_b", "delta_se_a", "delta_se_b", "c", "r", "r_se", "z", "theta_a",
                     "theta_b", "censored_a", "censored_b", "below_floor_a", "below_floor_b", "d_theta",
                     "d_theta_valid"]]
    return tests, residuals


# --------------------------------------------------------------------------- uncertainty
def pair_sums(mult: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    """Per test row and CATEGORY, the four sums every pooled statistic is a function of: ``S_a``, ``S_b``
    (anchor curve summed over levels) and ``D_a``, ``D_b`` (name effects summed over brands). Pooled
    c = sum S_b / sum S_a and m = sum D_b / sum D_a over categories, so resampling categories means
    resampling the rows of this table. NaN where a category lacks the cell."""
    ok = residuals[residuals["delta_a"].notna() & residuals["delta_b"].notna()]
    d = ok.groupby([*TEST_KEYS, "category"], sort=False).agg(
        D_a=("delta_a", "sum"), D_b=("delta_b", "sum"), n_entities=("delta_a", "size")).reset_index()
    d["curve_stratum"] = np.where(d["cross_stratum"], "attr", d["stratum"])
    s = mult.loc[(mult["category"] != ALL) & (mult["n_levels"] > 0),
                 ["arm", "stratum", "evidence_type", "category", "sum_a", "sum_b"]]
    s = s.rename(columns={"stratum": "curve_stratum", "sum_a": "S_a", "sum_b": "S_b"})
    out = d.merge(s, on=["arm", "curve_stratum", "evidence_type", "category"], how="left")
    covered = set(zip(out["arm"], out["curve_stratum"], out["evidence_type"], out["category"], out["cross_stratum"]))
    lone = s[[k not in covered for k in zip(s["arm"], s["curve_stratum"], s["evidence_type"], s["category"],
                                            [False] * len(s))]]
    if len(lone):  # anchor curves without any real brand in the same cell: scale only
        out = pd.concat([out, lone.assign(stratum=lone["curve_stratum"], cross_stratum=False, n_entities=0)],
                        ignore_index=True)
    out["n_entities"] = out["n_entities"].astype(int)
    return out[[*TEST_KEYS, "curve_stratum", "category", "S_a", "S_b", "D_a", "D_b", "n_entities"]]


def _count_matrix(index: np.ndarray, k: int) -> np.ndarray:
    """Rows of draws ``index`` (values in 0..k-1) -> how often each value was drawn, [n_draws, k]."""
    counts = np.zeros((index.shape[0], k))
    np.add.at(counts, (np.arange(index.shape[0])[:, None], index), 1.0)
    return counts


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero where either model lacks the cell, so a category contributes to both sums or to neither."""
    ok = np.isfinite(a) & np.isfinite(b)
    return np.where(ok, a, 0.0), np.where(ok, b, 0.0)


def _percentiles(draws: np.ndarray) -> tuple[float, float, int]:
    ok = draws[np.isfinite(draws)]
    if len(ok) == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(ok, CI)
    return float(lo), float(hi), int(len(ok))


def _t_interval(values: np.ndarray) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) of the 95% t interval of the mean of the finite ``values``; NaN bounds below 2."""
    ok = values[np.isfinite(values)]
    n = len(ok)
    if n < 2:
        return (float(ok.mean()) if n else float("nan")), float("nan"), float("nan"), n
    half = t_quantile(n - 1) * float(ok.std(ddof=1)) / math.sqrt(n)
    return float(ok.mean()), float(ok.mean() - half), float(ok.mean() + half), n


def _crossed_category(n_names: int, curve_cells: dict, delta_cells: dict, n_boot: int,
                      rng: np.random.Generator) -> tuple[dict, dict]:
    """Sums of ONE category recomputed under ``n_boot`` resamples of its fictional names.

    A resample draws ``n_names`` names with replacement; a name drawn twice counts twice. A pair of names is
    weighted by the product of its two counts, a control by its count, exactly as if the units of the drawn
    names had been stacked. Draws that leave a cell without any unit (no measured pair among the drawn
    names, or none of a brand's controls) are redrawn: the statistic is not defined on them.
    """
    def draw(n: int) -> np.ndarray:
        return _count_matrix(rng.integers(0, n_names, size=(n, n_names)), n_names)

    def compute(counts: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        valid = np.ones(len(counts), dtype=bool)
        s_out, d_out = {}, {}
        for key, levels in curve_cells.items():
            sa, sb = np.zeros(len(counts)), np.zeros(len(counts))
            for i, j, a_a, a_b in levels:
                w = counts[:, i] * counts[:, j]
                total = w.sum(axis=1)
                valid &= total > 0
                total = np.where(total > 0, total, 1.0)
                sa += (w @ a_a) / total
                sb += (w @ a_b) / total
            s_out[key] = (sa, sb)
        for key, blocks in delta_cells.items():
            da, db = np.zeros(len(counts)), np.zeros(len(counts))
            for k_idx, col_a, col_b in blocks:
                w = counts[:, k_idx]
                total = w.sum(axis=1)
                valid &= total > 0
                total = np.where(total > 0, total, 1.0)
                da += (w @ col_a) / total
                db += (w @ col_b) / total
            d_out[key] = (da, db)
        return s_out, d_out, valid

    counts = draw(n_boot)
    for _ in range(MAX_REDRAWS):
        s_out, d_out, valid = compute(counts)
        if valid.all():
            return s_out, d_out
        counts[~valid] = draw(int((~valid).sum()))
    raise RuntimeError("the crossed bootstrap could not draw fictional names that cover every cell")


def _crossed_arrays(arm: str, rows: pd.DataFrame, categories: list[str], crossed: dict[str, pd.DataFrame],
                    n_boot: int, seed: int) -> tuple[np.ndarray, ...]:
    """S_a, S_b, D_a, D_b as [rows, draws, categories] under resampling of the fictional names."""
    on_pair = [*KEYS, "level", "lo", "hi"]
    pa, pb = (crossed[k][crossed[k]["arm"] == arm] for k in ("pairs_a", "pairs_b"))
    pairs = _same_design(pa[[*on_pair, "a", "complete"]], pb[[*on_pair, "a", "complete"]], on_pair, "anchor pairs")
    pairs = pairs[pairs["complete_a"].astype(bool) & pairs["complete_b"].astype(bool)]
    on_cell = ["arm", "category", "stratum", "evidence_type", "entity", "control"]
    ca, cb = (crossed[k][crossed[k]["arm"] == arm] for k in ("cells_a", "cells_b"))
    cells = _same_design(ca[[*on_cell, "delta"]], cb[[*on_cell, "delta"]], on_cell, "brand x control cells")
    shape = (len(rows), n_boot, len(categories))
    out = [np.full(shape, np.nan) for _ in range(4)]
    curve_key = list(zip(rows["curve_stratum"], rows["evidence_type"]))
    delta_key = [(s, NO_TYPE if s in SHARED_ZERO_STRATA else t) for s, t in zip(rows["stratum"], rows["evidence_type"])]
    pairs_by = {c: g for c, g in pairs.groupby("category", sort=False)}
    cells_by = {c: g for c, g in cells.groupby("category", sort=False)}
    for ci, category in enumerate(categories):
        p, q = pairs_by.get(category), cells_by.get(category)
        found: set[str] = set()
        if p is not None:
            found |= set(p["lo"]) | set(p["hi"])
        if q is not None:
            found |= set(q["control"])
        if not found:
            continue
        names = sorted(found)
        pos = {n: i for i, n in enumerate(names)}
        curve_cells: dict = {}
        if p is not None:
            for (stratum, etype), g in p.groupby(["stratum", "evidence_type"], sort=False):
                curve_cells[(stratum, etype)] = [
                    (h["lo"].map(pos).to_numpy(), h["hi"].map(pos).to_numpy(), h["a_a"].to_numpy(dtype=float),
                     h["a_b"].to_numpy(dtype=float)) for _, h in g.groupby("level", sort=True)]
        delta_cells: dict = {}
        if q is not None:
            for (stratum, etype), g in q.groupby(["stratum", "evidence_type"], sort=False):
                wide_a = g.pivot(index="entity", columns="control", values="delta_a")
                wide_b = g.pivot(index="entity", columns="control", values="delta_b")
                pattern = wide_a.notna().apply(lambda r: tuple(r.index[r.to_numpy()]), axis=1)
                blocks = []
                for controls, members in pattern.groupby(pattern, sort=False):
                    k_idx = np.array([pos[c] for c in controls], dtype=int)
                    blocks.append((k_idx, wide_a.loc[members.index, list(controls)].to_numpy().sum(axis=0),
                                   wide_b.loc[members.index, list(controls)].to_numpy().sum(axis=0)))
                delta_cells[(stratum, etype)] = blocks
        rng = rng_for(seed, "scale-bootstrap", "names", arm, category)
        s_out, d_out = _crossed_category(len(names), curve_cells, delta_cells, n_boot, rng)
        for ri in range(len(rows)):
            if curve_key[ri] in s_out:
                out[0][ri, :, ci], out[1][ri, :, ci] = s_out[curve_key[ri]]
            if delta_key[ri] in d_out:
                out[2][ri, :, ci], out[3][ri, :, ci] = d_out[delta_key[ri]]
    return tuple(out)


def uncertainty(sums: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                crossed: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Intervals for the pooled c, m and T1 of every test row, and for H1. Returns ``(table, h1)``.

    Three intervals, all at 95%, all with the category as the independent unit:

    * category bootstrap (``*_lo`` / ``*_hi``): categories are resampled with replacement and the pooled
      ratios recomputed from the resampled category sums (percentile interval over ``n_boot`` draws).
    * crossed bootstrap (``*_lo_crossed`` / ``*_hi_crossed``; needs ``crossed`` = {pairs_a, pairs_b, cells_a,
      cells_b}): the same category draws, and inside every category the fictional names are resampled once
      per draw, with the anchor curves and name effects recomputed from the units of the drawn names. It
      adds the question "what if this category had been given other made-up names". Brands are NOT
      resampled: they are the population of interest, and they are not independent given shared controls.
    * t interval (``*_t_lo`` / ``*_t_hi``) of the mean of the PER-CATEGORY values, df = categories - 1. Its
      centre ``*_cat_mean`` weights categories equally and is not the pooled estimate. This is the PRIMARY
      interval for confirmatory claims: with 10 categories and a true null it covered 97% of the time against
      91% for the percentile bootstrap (simulation in tests/test_scale_estimators.py; DECISIONS.md).

    ``h1``: per arm, the range (max - min) of log c over the three evidence types of the attribute-only
    stratum, with both bootstrap intervals. A range is >= 0 by construction and biased upwards by noise, so
    read it against its interval and against replicate pairs (``calibrate``), not against 0.
    """
    tables: list[pd.DataFrame] = []
    h1: dict[str, Any] = {}
    for arm, part in sums.groupby("arm", sort=False):
        rows = part[[*TEST_KEYS, "curve_stratum"]].drop_duplicates().reset_index(drop=True)
        categories = sorted(part["category"].unique())
        row_of = {tuple(r): i for i, r in enumerate(rows[TEST_KEYS].itertuples(index=False, name=None))}
        col_of = {c: i for i, c in enumerate(categories)}
        base = [np.full((len(rows), len(categories)), np.nan) for _ in range(4)]
        ri = [row_of[k] for k in part[TEST_KEYS].itertuples(index=False, name=None)]
        cj = [col_of[c] for c in part["category"]]
        for arr, col in zip(base, ("S_a", "S_b", "D_a", "D_b")):
            arr[ri, cj] = part[col].to_numpy(dtype=float)
        sa, sb = _paired(base[0], base[1])
        da, db = _paired(base[2], base[3])
        n_cat = len(categories)
        counts = _count_matrix(rng_for(seed, "scale-bootstrap", "categories", arm).integers(
            0, n_cat, size=(n_boot, n_cat)), n_cat)  # [draws, categories]
        draws = {"": (_log_ratio(sb @ counts.T, sa @ counts.T), _log_ratio(db @ counts.T, da @ counts.T))}
        if crossed is not None:
            xs = _crossed_arrays(arm, rows, categories, crossed, n_boot, seed)
            xsa, xsb = _paired(xs[0], xs[1])
            xda, xdb = _paired(xs[2], xs[3])
            draws["_crossed"] = (
                _log_ratio(np.einsum("rbc,bc->rb", xsb, counts), np.einsum("rbc,bc->rb", xsa, counts)),
                _log_ratio(np.einsum("rbc,bc->rb", xdb, counts), np.einsum("rbc,bc->rb", xda, counts)))
        out = rows[TEST_KEYS].copy()
        per_cat = {"log_c": _log_ratio(base[1], base[0]), "log_m": _log_ratio(base[3], base[2])}
        per_cat["T1"] = per_cat["log_m"] - per_cat["log_c"]
        for name, values in per_cat.items():
            stats = [_t_interval(v) for v in values]
            out[f"{name}_cat_mean"] = [s[0] for s in stats]
            out[f"{name}_t_lo"], out[f"{name}_t_hi"] = [s[1] for s in stats], [s[2] for s in stats]
            out[f"n_cat_{name}"] = [s[3] for s in stats]
        for suffix, (log_c, log_m) in draws.items():
            for name, values in (("log_c", log_c), ("log_m", log_m), ("T1", log_m - log_c)):
                stats = [_percentiles(v) for v in values]
                out[f"{name}_lo{suffix}"], out[f"{name}_hi{suffix}"] = [s[0] for s in stats], [s[1] for s in stats]
                out[f"n_boot_{name}{suffix}"] = [s[2] for s in stats]
            for name in ("c", "m"):
                out[f"{name}_lo{suffix}"] = np.exp(out[f"log_{name}_lo{suffix}"])
                out[f"{name}_hi{suffix}"] = np.exp(out[f"log_{name}_hi{suffix}"])
        tables.append(out)
        attr = [row_of.get((arm, "attr", t, False)) for t in LADDER_TYPES]
        if all(i is not None for i in attr):
            point = _log_ratio(sb[attr].sum(axis=1), sa[attr].sum(axis=1))
            entry: dict[str, Any] = {"log_c": dict(zip(LADDER_TYPES, (float(v) for v in point))),
                                     "range": float(point.max() - point.min()), "n_categories": n_cat}
            for suffix, (log_c, _) in draws.items():
                spread = log_c[attr].max(axis=0) - log_c[attr].min(axis=0)  # NaN as soon as one type is undefined
                entry[f"lo{suffix}"], entry[f"hi{suffix}"], entry[f"n_boot{suffix}"] = _percentiles(spread)
            h1[str(arm)] = entry
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=TEST_KEYS)
    return table, h1


# --------------------------------------------------------------------------- noise band from replicates
def calibrate(stat_value: float, null_values: Any) -> dict[str, Any]:
    """Is ``stat_value`` outside what content-neutral training produces?

    ``null_values``: the same statistic on K seed-replicate pairs (runs whose training data do not touch the
    domain). Assuming those are draws from one normal distribution, a new draw falls in
    ``mean +/- t(K-1, 0.975) * sd * sqrt(1 + 1/K)`` with probability 95% (a PREDICTION interval for one new
    value, wider than a confidence interval of the mean). ``outside`` says whether the observed value does
    not; ``outside_2x`` repeats the test with the sd doubled, for the case where the replicates understate
    the run-to-run variation of the stage being tested (different recipe, more steps).
    """
    null = np.asarray(null_values, dtype=float).ravel()
    null = null[np.isfinite(null)]
    k = len(null)
    if k < 2:
        raise ValueError(f"calibrate needs at least 2 finite replicate values, got {k}")
    mean, sd = float(null.mean()), float(null.std(ddof=1))
    t = t_quantile(k - 1)
    half = t * sd * math.sqrt(1.0 + 1.0 / k)
    value = float(stat_value)
    defined = math.isfinite(value)
    return {"value": value, "k": k, "mean": mean, "sd": sd, "t": t,
            "lo": mean - half, "hi": mean + half, "outside": bool(defined and not mean - half <= value <= mean + half),
            "lo_2x": mean - 2 * half, "hi_2x": mean + 2 * half,
            "outside_2x": bool(defined and not mean - 2 * half <= value <= mean + 2 * half), "defined": defined}
