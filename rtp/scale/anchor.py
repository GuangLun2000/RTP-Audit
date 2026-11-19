"""Anchor units (fictional target vs fictional control): name pulls, anchor curves, and their inverse.

Model of one anchor unit at evidence level k against the target::

    L(i, j; k) = pull_i - pull_j - A(k) + noise

``pull`` is what a made-up name is worth by itself (ideally nothing), ``A(k)`` is the model's response to
evidence step k, in logits. Two design facts make both readable:

* d = 0: every unordered pair appears once, so ``L(i, j) = pull_i - pull_j + noise``. Which name is called
  the target is arbitrary there; least squares with sum-to-zero gives one pull per name.
* d = k > 0: each pair of the design appears in BOTH directions, so in the mean of the two directions the
  pulls cancel exactly and what is left is ``-A(k)``.

``A`` is in logits and is multiplied by c when every logit is multiplied by c. Its inverse turns a name
effect in logits into evidence units, which do not depend on c (rtp.scale.theta).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEYS = ["arm", "category", "stratum", "evidence_type"]
ALL = "(all)"  # category label of a curve pooled over categories
CARD_STRATA = ("full", "comp")  # strata that show two different card texts (counterbalanced by ``swap``)
COMPOSITE = "composite"
PULL_COLUMNS = [*KEYS, "name", "pull", "pull_se", "n", "n_names", "n_units", "pull_sd", "resid_sd", "identified"]
PAIR_COLUMNS = [*KEYS, "level", "lo", "hi", "a", "n_units", "complete"]
CURVE_COLUMNS = [*KEYS, "level", "x", "A_raw", "se", "n_units", "n_pairs", "n_pairs_dropped", "n_categories",
                 "A_iso", "monotone_raw"]


def level_x(evidence_type: str, level: int, steps: dict[str, list[float]]) -> float:
    """Natural unit of ladder level ``level`` (1-based) of a type: stars removed (rating), relative price
    increase (price), log of the review-count ratio (reviews). The composite ladder has no natural unit: its
    x is the level index. Level 0 is x = 0 for every type."""
    level = int(level)
    if level == 0:
        return 0.0
    if evidence_type == COMPOSITE:
        return float(level)
    if evidence_type not in steps:
        raise ValueError(f"no ladder steps for evidence type '{evidence_type}' (have {sorted(steps)})")
    if not 1 <= level <= len(steps[evidence_type]):
        raise ValueError(f"level {level} of '{evidence_type}' is outside its ladder of {len(steps[evidence_type])} steps")
    value = float(steps[evidence_type][level - 1])
    return math.log(value) if evidence_type == "reviews" else value


# --------------------------------------------------------------------------- name pulls (d = 0)
def name_pulls(units: pd.DataFrame) -> pd.DataFrame:
    """Pull of every fictional name, per (arm, category, stratum, evidence_type), from the d = 0 anchor units.

    Estimand: how many logits a made-up name is worth against the average made-up name of its category, with
    no evidence separating the options. Least squares on ``L(i, j) = pull_i - pull_j`` with sum-to-zero; it
    is identified when the pairs connect all names (the design's regular pair subsets do).

    Columns: ``pull`` (logits), ``pull_se`` (from the residual variance; NaN without residual degrees of
    freedom), ``n`` (d = 0 units the name is in), and per cell, repeated on its rows: ``n_names``,
    ``n_units``, ``pull_sd`` (SD of the pulls, ddof 1), ``resid_sd``, ``identified`` (False: the pairs do
    not connect the names; pulls are NaN). The d = 0 cards of the full stratum carry evidence_type "none".
    """
    zero = units[(units["family"] == "anchor") & (units["d"] == 0)]
    rows: list[dict[str, Any]] = []
    for key, g in zero.groupby(KEYS, sort=False):
        names = sorted(set(g["target_entity"]) | set(g["control_entity"]))
        pos = {n: i for i, n in enumerate(names)}
        k, n_obs = len(names), len(g)
        ti = g["target_entity"].map(pos).to_numpy()
        ci = g["control_entity"].map(pos).to_numpy()
        design = np.zeros((n_obs, k))
        design[np.arange(n_obs), ti] = 1.0
        design[np.arange(n_obs), ci] = -1.0
        y = g["L"].to_numpy(dtype=float)
        identified = bool(np.linalg.matrix_rank(design) == k - 1)
        pull = np.full(k, np.nan)
        se = np.full(k, np.nan)
        resid_sd = np.nan
        if identified:
            pinv = np.linalg.pinv(design)
            pull = pinv @ y  # minimum-norm solution: orthogonal to the constant vector, i.e. sum-to-zero
            dof = n_obs - (k - 1)
            if dof > 0:
                resid = y - design @ pull
                resid_sd = float(np.sqrt(resid @ resid / dof))
                se = resid_sd * np.sqrt(np.einsum("ij,ij->i", pinv, pinv))
        else:
            log.warning("name pulls of %s are not identified: the d = 0 pairs do not connect the names", key)
        n_per = np.bincount(ti, minlength=k) + np.bincount(ci, minlength=k)
        sd = float(np.std(pull, ddof=1)) if identified and k > 1 else np.nan
        for i, name in enumerate(names):
            rows.append({**dict(zip(KEYS, key)), "name": name, "pull": float(pull[i]), "pull_se": float(se[i]),
                         "n": int(n_per[i]), "n_names": k, "n_units": n_obs, "pull_sd": sd, "resid_sd": resid_sd,
                         "identified": identified})
    return pd.DataFrame(rows, columns=PULL_COLUMNS)


# --------------------------------------------------------------------------- anchor curve (d > 0)
def anchor_pairs(units: pd.DataFrame) -> pd.DataFrame:
    """Per unordered pair of fictional names and ladder level: ``a`` = minus the mean of L over the pair's two
    directions (each direction first averaged over its bases and swaps).

    The pulls of the two names enter the two directions with opposite signs, so ``a`` estimates A(level)
    whatever the pulls are. ``complete`` is False (and ``a`` NaN) when that cancellation is not guaranteed:
    a direction is missing, the two directions have different numbers of units, or, on full cards, the two
    card texts are not counterbalanced within each direction. ``lo`` < ``hi`` are the two names.
    """
    a = units[(units["family"] == "anchor") & (units["d"] != 0)]
    if (a["d"] < 0).any():
        raise ValueError("anchor units with d < 0: the anchor ladder is defined for d >= 0 only (A(-k) = -A(k))")
    if a.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    target = a["target_entity"].to_numpy(dtype=object)
    control = a["control_entity"].to_numpy(dtype=object)
    forward = target < control
    work = a[KEYS].copy()
    work["level"] = a["d"].to_numpy(dtype=int)
    work["lo"] = np.where(forward, target, control)
    work["hi"] = np.where(forward, control, target)
    work["forward"] = forward
    work["L"] = a["L"].to_numpy(dtype=float)
    work["swap"] = a["swap"].to_numpy(dtype=float)
    by = [*KEYS, "level", "lo", "hi"]
    side = work.groupby([*by, "forward"], sort=False).agg(L=("L", "mean"), n=("L", "size"), swap=("swap", "mean"))
    side = side.reset_index()
    both = side[side["forward"]].drop(columns="forward").merge(
        side[~side["forward"]].drop(columns="forward"), on=by, how="outer", suffixes=("_f", "_b"))
    cards = both["stratum"].isin(CARD_STRATA).to_numpy()
    complete = (both["L_f"].notna() & both["L_b"].notna() & (both["n_f"] == both["n_b"])
                & (both["swap_f"] == both["swap_b"])).to_numpy()
    complete &= ~cards | (both["swap_f"].to_numpy() == 0.5)
    both["a"] = np.where(complete, -(both["L_f"] + both["L_b"]) / 2.0, np.nan)
    both["n_units"] = (both["n_f"].fillna(0) + both["n_b"].fillna(0)).astype(int)
    both["complete"] = complete
    if not complete.all():
        log.warning("%d anchor pair cell(s) lack a counterbalanced direction and are left out of the curves",
                    int((~complete).sum()))
    return both[PAIR_COLUMNS].reset_index(drop=True)


def pava(y: Any, w: Any = None) -> np.ndarray:
    """Weighted isotonic (non-decreasing) least-squares fit by pool-adjacent-violators."""
    y = np.asarray(y, dtype=float).ravel()
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float).ravel()
    if len(w) != len(y) or np.isnan(y).any() or not np.all(w > 0):
        raise ValueError("pava needs finite values and one positive weight per value")
    value: list[float] = []
    weight: list[float] = []
    size: list[int] = []
    for yi, wi in zip(y, w):
        value.append(float(yi))
        weight.append(float(wi))
        size.append(1)
        while len(value) > 1 and value[-2] > value[-1]:
            v, u, s = value.pop(), weight.pop(), size.pop()
            total = weight[-1] + u
            value[-1] = (value[-1] * weight[-1] + v * u) / total
            weight[-1] = total
            size[-1] += s
    return np.repeat(value, size)


def isotonic_through_origin(y: Any, w: Any = None) -> np.ndarray:
    """Isotonic non-decreasing fit of the levels 1..K with the fixed point (level 0, A = 0): the least-squares
    fit under ``0 <= A(1) <= ... <= A(K)``, which is the unconstrained isotonic fit clipped at 0."""
    return np.maximum(pava(y, w), 0.0)


def _fit_curve(cell: pd.DataFrame) -> pd.DataFrame:
    """Add ``A_iso`` and ``monotone_raw`` to the rows of ONE curve (sorted by level here)."""
    cell = cell.sort_values("level").copy()
    raw = cell["A_raw"].to_numpy(dtype=float)
    ok = np.isfinite(raw) & (cell["n_units"].to_numpy() > 0)
    iso = np.full(len(cell), np.nan)
    if ok.any():
        iso[ok] = isotonic_through_origin(raw[ok], cell["n_units"].to_numpy(dtype=float)[ok])
    cell["A_iso"] = iso
    cell["monotone_raw"] = bool(ok.all() and np.all(np.diff(np.concatenate([[0.0], raw])) > 0))
    return cell


def anchor_curves(units: pd.DataFrame, steps: dict[str, list[float]], pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The anchor curve A(level) of every (arm, category, stratum, evidence_type), plus a pooled curve per
    (arm, stratum, evidence_type) with category "(all)".

    Estimand: the logit shift produced by evidence step ``level`` against one of two made-up names. Valid
    because both directions of every pair are present, so the names' own pulls cancel (``anchor_pairs``).

    Columns: ``x`` (natural unit of the level, see ``level_x``), ``A_raw`` (mean over the complete pairs; in
    the balanced design this is minus the cell mean of L), ``se`` (SD over unordered pairs / sqrt(n_pairs):
    pairs are the units, although pairs that share a name are not independent), ``n_units``, ``n_pairs``,
    ``n_pairs_dropped`` (incomplete pairs), ``A_iso`` (isotonic non-decreasing fit through (0, 0), weights
    ``n_units``; this is the curve that gets inverted), ``monotone_raw`` (per curve: the raw means increase
    strictly with the level, starting above 0).

    Pooled rows: ``A_raw`` is the mean over categories with equal weight per category, ``se`` the SD over
    categories / sqrt(n_categories), ``A_iso`` and ``monotone_raw`` are recomputed on the pooled means.
    """
    pairs = anchor_pairs(units) if pairs is None else pairs
    if pairs.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    work = pairs.assign(a_ok=pairs["a"].where(pairs["complete"]),
                        units_ok=np.where(pairs["complete"], pairs["n_units"], 0))
    cell = work.groupby([*KEYS, "level"], sort=False).agg(
        A_raw=("a_ok", "mean"), sd=("a_ok", "std"), n_pairs=("a_ok", "count"), n_all=("a", "size"),
        n_units=("units_ok", "sum")).reset_index()
    cell["se"] = cell["sd"] / np.sqrt(cell["n_pairs"].where(cell["n_pairs"] > 0))
    cell["n_pairs_dropped"] = cell["n_all"] - cell["n_pairs"]
    cell["n_categories"] = 1
    pooled = cell.groupby(["arm", "stratum", "evidence_type", "level"], sort=False).agg(
        A_mean=("A_raw", "mean"), sd=("A_raw", "std"), n_categories=("A_raw", "count"),
        n_pairs=("n_pairs", "sum"), n_pairs_dropped=("n_pairs_dropped", "sum"), n_units=("n_units", "sum")).reset_index()
    pooled = pooled.rename(columns={"A_mean": "A_raw"})
    pooled["se"] = pooled["sd"] / np.sqrt(pooled["n_categories"].where(pooled["n_categories"] > 0))
    pooled["category"] = ALL
    both = pd.concat([cell, pooled], ignore_index=True)
    both["x"] = [level_x(t, k, steps) for t, k in zip(both["evidence_type"], both["level"])]
    fitted = [_fit_curve(g) for _, g in both.groupby(KEYS, sort=False)]
    out = pd.concat(fitted, ignore_index=True)
    for col in ("level", "n_units", "n_pairs", "n_pairs_dropped", "n_categories"):
        out[col] = out[col].astype(int)
    return out[CURVE_COLUMNS]


# --------------------------------------------------------------------------- inverse
def invert(curve_x: Any, curve_A: Any, delta: Any) -> tuple[np.ndarray, np.ndarray]:
    """Evidence level worth ``delta`` logits: the inverse of the piecewise-linear curve through (0, 0) and
    the points ``(curve_x, curve_A)``, vectorised over ``delta``.

    ``curve_x`` must increase strictly from above 0 and ``curve_A`` must be non-negative and non-decreasing
    (pass ``A_iso``). A negative ``delta`` uses the odd extension A(-x) = -A(x), which holds by construction
    (evidence against the control is the same comparison with the roles exchanged). A value on a flat
    segment maps to the segment's LEFT end, the smallest evidence that reaches it. Beyond the last point the
    result is +/- x_max with ``censored`` True: the name is worth more than the whole ladder, and by how much
    is not measured. NaN in, NaN out (not censored).

    Returns ``(theta, censored)``, shaped like ``delta``.
    """
    x = np.asarray(curve_x, dtype=float).ravel()
    a = np.asarray(curve_A, dtype=float).ravel()
    if len(x) == 0 or len(x) != len(a) or not (np.isfinite(x).all() and np.isfinite(a).all()):
        raise ValueError("invert needs a non-empty curve of finite points, one A per x")
    if x[0] == 0.0:  # an explicit origin is accepted
        if a[0] != 0.0:
            raise ValueError("the curve must pass through (0, 0)")
        x, a = x[1:], a[1:]
    if len(x) == 0 or x[0] <= 0 or np.any(np.diff(x) <= 0):
        raise ValueError("curve_x must increase strictly from above 0")
    if a[0] < 0 or np.any(np.diff(a) < 0):
        raise ValueError("curve_A must be non-negative and non-decreasing (use the isotonic fit A_iso)")
    xs = np.concatenate([[0.0], x])
    grid = np.concatenate([[0.0], a])
    delta = np.asarray(delta, dtype=float)
    size = np.abs(delta)
    nan = np.isnan(size)
    safe = np.where(nan, 0.0, size)
    censored = safe > grid[-1]
    j = np.searchsorted(grid, safe, side="left")  # first point with A >= |delta|; the left end of a flat run
    hi = np.clip(j, 1, len(grid) - 1)
    span = grid[hi] - grid[hi - 1]
    inside = ~censored & (j > 0) & (grid[np.minimum(j, len(grid) - 1)] > safe)  # strictly inside a rising segment
    frac = np.divide(safe - grid[hi - 1], span, out=np.zeros_like(safe), where=inside & (span > 0))
    theta = np.where(inside, xs[hi - 1] + frac * (xs[hi] - xs[hi - 1]), xs[np.minimum(j, len(grid) - 1)])
    theta = np.where(censored, xs[-1], theta)
    theta = np.where(nan, np.nan, np.sign(delta) * theta)
    return theta, censored & ~nan


def curve_index(curves: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, np.ndarray]]:
    """(arm, category, stratum, evidence_type) -> the curve's invertible points: ``level``, ``x``, ``A_iso``
    (levels without a fitted value are left out; curves without any are absent)."""
    out: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for key, g in curves[curves["A_iso"].notna()].groupby(KEYS, sort=False):
        g = g.sort_values("level")
        out[tuple(key)] = {"level": g["level"].to_numpy(dtype=float), "x": g["x"].to_numpy(dtype=float),
                           "A_iso": g["A_iso"].to_numpy(dtype=float)}
    return out
