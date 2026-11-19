"""Preference levels: a name effect in logits, converted to evidence units by the model's own anchor curve.

For a real brand e against a fictional control at d = 0::

    L(e, control) = Delta_e - pull_control + noise

The mean over the controls is the NAME EFFECT ``delta`` (logits; the pulls of the fictional names sum to
zero over all controls). The PREFERENCE LEVEL is ``theta = A^{-1}(delta)``: how much adverse evidence, on
the ladder measured between two made-up names in the same category and stratum, offsets the brand's name.
Multiplying every logit by c multiplies ``delta`` and ``A`` by c and leaves ``theta`` where it is.

``theta`` is read a second way where the design allows it: a few brands per category are also shown at
d != 0, and ``theta_direct`` is the evidence at which the brand's own L crosses 0. The two readings agree
when name and evidence act on one logit axis (the "common channel" condition).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from rtp.data.constants import LADDER_TYPES
from rtp.scale.anchor import CARD_STRATA, KEYS, curve_index, invert, level_x, pava

log = logging.getLogger(__name__)

NO_TYPE = "none"
SHARED_ZERO_STRATA = ("name", "full")  # strata whose d = 0 units carry evidence_type "none"
CROSS_COLUMNS = [f"theta_cross_{t}" for t in LADDER_TYPES] + [f"censored_cross_{t}" for t in LADDER_TYPES]
CELL_COLUMNS = ["arm", "split", "category", "stratum", "evidence_type", "entity", "control", "delta", "n_units"]
ENTITY_KEYS = ["arm", "category", "stratum", "evidence_type", "entity"]
DIRECT_COLUMNS = ["has_direct", "direct_status", "direct_n_levels", "direct_n_controls", "direct_swap_balanced",
                  "direct_sign_changes", "delta_matched", "theta_matched", "theta_matched_level", "matched_censored",
                  "theta_direct", "theta_direct_level", "direct_censored"]


def zero_cells(units: pd.DataFrame) -> pd.DataFrame:
    """Per real brand and fictional control: mean of L over the real d = 0 units of the cell (bases x swaps;
    the swap average cancels the difference between the two card texts). ``evidence_type`` is the one the
    units carry: "none" in the name and full strata, the ladder's type in the attribute-only stratum."""
    real = units[(units["family"] == "real") & (units["d"] == 0)]
    if real.empty:
        return pd.DataFrame(columns=CELL_COLUMNS)
    by = ["arm", "split", "category", "stratum", "evidence_type", "target_entity", "control_entity"]
    cells = real.groupby(by, sort=False).agg(delta=("L", "mean"), n_units=("L", "size")).reset_index()
    return cells.rename(columns={"target_entity": "entity", "control_entity": "control"})[CELL_COLUMNS]


def convert(frame: pd.DataFrame, value: str, curves: pd.DataFrame) -> pd.DataFrame:
    """Convert the logit column ``value`` of ``frame`` (keys: arm, category, stratum, evidence_type) to
    evidence units with each row's own category curve.

    * attr, comp: the curve of the row's (stratum, evidence_type).
    * full: the d = 0 cards are shared by the three types, so a row with evidence_type "none" becomes one row
      per type, same value, each inverted against the full-card curve of that type.
    * name: there is no ladder without an evidence carrier, so ``theta`` is NaN (status ``cross_stratum``)
      and ``theta_cross_<type>`` holds the inversion against the category's ATTRIBUTE-ONLY curve of each
      type: a cross-stratum number, valid only if a name is worth the same next to an attribute as alone.

    Adds ``theta`` (natural units), ``theta_level`` (level-index units), ``censored`` (|value| beyond the
    top of the curve; theta is then +/- the top, a bound), ``below_floor``, ``status`` (ok / censored /
    flat_curve / no_curve / no_value / cross_stratum) and the cross columns.

    ``flat_curve``: the category's fitted curve never rises above 0 (the model does not respond to this
    evidence, or responds the wrong way). There is nothing to invert: ``theta`` is NaN and the cell is NOT
    counted as censored, which would read as "worth more than the whole ladder".

    ``below_floor`` is judged on the logit scale: |value| is smaller than the first strictly positive point
    of the fitted curve, so the ladder cannot tell the cell from 0. (When the first step's raw mean is not
    positive the fit is 0 up to that step, and the inverse places every small value at or beyond it.)
    """
    index = curve_index(curves)
    shared = (frame["stratum"] == "full") & (frame["evidence_type"] == NO_TYPE)
    parts = [frame[~shared]] + [frame[shared].assign(evidence_type=t) for t in LADDER_TYPES]
    out = pd.concat(parts, ignore_index=True)
    n = len(out)
    theta, theta_level = np.full(n, np.nan), np.full(n, np.nan)
    censored, below = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    status = np.full(n, "no_curve", dtype=object)
    cross = {t: (np.full(n, np.nan), np.zeros(n, dtype=bool)) for t in LADDER_TYPES}
    values = out[value].to_numpy(dtype=float)
    for key, idx in out.groupby(KEYS, sort=False).indices.items():
        arm, category, stratum, _ = key
        v = values[idx]
        if stratum == "name":
            status[idx] = "cross_stratum"
            for t in LADDER_TYPES:
                curve = index.get((arm, category, "attr", t))
                if curve is not None and rises(curve):
                    cross[t][0][idx], cross[t][1][idx] = invert(curve["x"], curve["A_iso"], v)
            continue
        curve = index.get(tuple(key))
        if curve is None:
            continue
        if not rises(curve):
            status[idx] = np.where(np.isnan(v), "no_value", "flat_curve")
            continue
        theta[idx], censored[idx] = invert(curve["x"], curve["A_iso"], v)
        theta_level[idx], _ = invert(curve["level"], curve["A_iso"], v)
        below[idx] = np.abs(v) < curve["A_iso"][curve["A_iso"] > 0][0]
        status[idx] = np.where(np.isnan(v), "no_value", np.where(censored[idx], "censored", "ok"))
    out["theta"], out["theta_level"], out["censored"], out["below_floor"] = theta, theta_level, censored, below
    out["status"] = status
    for t in LADDER_TYPES:
        out[f"theta_cross_{t}"], out[f"censored_cross_{t}"] = cross[t]
    return out


def rises(curve: dict[str, np.ndarray]) -> bool:
    """True if the fitted curve has a positive value somewhere, i.e. there is a response to invert."""
    return bool(len(curve["A_iso"]) and np.max(curve["A_iso"]) > 0)


def long_theta(table: pd.DataFrame) -> pd.DataFrame:
    """One ``theta`` per row: a name-stratum row becomes one row per ladder type with ``cross_stratum`` True,
    its ``theta`` / ``censored`` taken from the cross columns (``theta_level`` is not defined there). Rows of
    the other strata are kept with ``cross_stratum`` False. The cross columns are dropped."""
    name = table["stratum"] == "name"
    parts = [table[~name].assign(cross_stratum=False)]
    for t in LADDER_TYPES:
        part = table[name].assign(evidence_type=t, cross_stratum=True)
        part["theta"] = part[f"theta_cross_{t}"]
        part["censored"] = part[f"censored_cross_{t}"].astype(bool)
        part["theta_level"] = np.nan
        part["below_floor"] = False
        part["status"] = np.where(part["theta"].isna(), "no_curve", np.where(part["censored"], "censored", "ok"))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).drop(columns=CROSS_COLUMNS)
    out["cross_stratum"] = out["cross_stratum"].astype(bool)
    return out


# --------------------------------------------------------------------------- direct reading
def crossing(x: Any, y: Any, w: Any = None) -> tuple[float, int]:
    """Where a decreasing response crosses 0: ``(x*, 0)``, or ``(nan, +1)`` when it is still positive at the
    largest x and ``(nan, -1)`` when it is already negative at the smallest.

    ``y`` (mean L at evidence ``x``, x increasing) is first replaced by its isotonic NON-INCREASING fit
    (weights ``w``), then interpolated linearly between adjacent points. With monotone means this is plain
    interpolation of the means; with noisy, non-monotone means it still gives one crossing instead of several.
    """
    x = np.asarray(x, dtype=float)
    fit = -pava(-np.asarray(y, dtype=float), w)
    if fit[-1] > 0:
        return float("nan"), 1
    if fit[0] < 0:
        return float("nan"), -1
    j = int(np.argmax(fit <= 0))
    if j == 0 or fit[j] == 0:
        return float(x[j]), 0
    return float(x[j - 1] + fit[j - 1] / (fit[j - 1] - fit[j]) * (x[j] - x[j - 1])), 0


def direct_readings(units: pd.DataFrame, curves: pd.DataFrame, steps: dict[str, list[float]]) -> pd.DataFrame:
    """Direct reading of the preference level for the brands that are shown at d != 0.

    For one (arm, category, stratum, evidence_type, brand): the mean of L at each measured level, over the
    brand's d != 0 units and over the d = 0 units with the SAME controls, bases and swaps (so every level
    shows the same comparisons), and the evidence ``theta_direct`` at which that response crosses 0
    (``crossing``). Levels d < 0, where the control is the worse option, count as negative evidence.

    ``delta_matched`` is the d = 0 point of that response and ``theta_matched`` its conversion through the
    anchor curve: the like-for-like partner of ``theta_direct`` (same controls, hence the same offset from
    the controls' own pulls). ``direct_censored``: +1 the response is still positive at the top of the
    ladder, -1 it is already negative at the most favourable level measured, 0 otherwise.
    ``direct_swap_balanced`` is False when a card stratum shows the brand on one card text only: the direct
    reading then includes the difference between the two card texts.
    """
    real = units[units["family"] == "real"]
    moved = real[real["d"] != 0]
    columns = [*ENTITY_KEYS, *DIRECT_COLUMNS]
    if moved.empty:
        return pd.DataFrame(columns=columns)
    cols = [*KEYS, "target_entity", "control_entity", "base", "swap", "d", "L"]
    moved = moved[cols].copy()
    moved["zero_type"] = np.where(moved["stratum"].isin(SHARED_ZERO_STRATA), NO_TYPE, moved["evidence_type"])
    match = ["arm", "category", "stratum", "zero_type", "target_entity", "control_entity", "base", "swap"]
    wanted = moved[[*match, "evidence_type"]].drop_duplicates()
    zero = real[real["d"] == 0][cols].rename(columns={"evidence_type": "zero_type"})
    zero = wanted.merge(zero, on=match, how="inner")
    both = pd.concat([moved, zero], ignore_index=True)
    index = curve_index(curves)
    rows: list[dict[str, Any]] = []
    for key, g in both.groupby([*KEYS, "target_entity"], sort=False):
        arm, category, stratum, etype, entity = key
        lv = g.groupby("d").agg(L=("L", "mean"), n=("L", "size")).sort_index()
        d = lv.index.to_numpy(dtype=int)
        row: dict[str, Any] = {
            **dict(zip(ENTITY_KEYS, (arm, category, stratum, etype, entity))), "has_direct": True,
            "direct_status": "ok", "direct_n_levels": int(len(d)), "direct_n_controls": int(g["control_entity"].nunique()),
            "direct_swap_balanced": bool(stratum not in CARD_STRATA or g.loc[g["d"] != 0, "swap"].mean() == 0.5),
            "direct_sign_changes": 0, "delta_matched": np.nan, "theta_matched": np.nan,
            "theta_matched_level": np.nan, "matched_censored": False, "theta_direct": np.nan,
            "theta_direct_level": np.nan, "direct_censored": 0}
        if 0 not in d:
            row["direct_status"] = "no_zero"  # no d = 0 unit with the same control, base and swap
            rows.append(row)
            continue
        y = lv["L"].to_numpy(dtype=float)
        x = np.array([np.sign(k) * level_x(etype, abs(int(k)), steps) for k in d])
        signs = np.sign(y[y != 0])
        row["direct_sign_changes"] = int(np.sum(signs[1:] != signs[:-1]))
        row["delta_matched"] = float(y[d == 0][0])
        weights = lv["n"].to_numpy(dtype=float)
        row["theta_direct"], row["direct_censored"] = crossing(x, y, weights)
        row["theta_direct_level"], _ = crossing(d.astype(float), y, weights)
        if row["direct_censored"]:
            row["direct_status"] = "beyond_top" if row["direct_censored"] > 0 else "beyond_favourable"
        curve = index.get((arm, category, stratum, etype))
        if curve is not None and rises(curve):
            theta, cens = invert(curve["x"], curve["A_iso"], row["delta_matched"])
            row["theta_matched"], row["matched_censored"] = float(theta), bool(cens)
            row["theta_matched_level"] = float(invert(curve["level"], curve["A_iso"], row["delta_matched"])[0])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- tables
def entity_table(units: pd.DataFrame, curves: pd.DataFrame, steps: dict[str, list[float]]) -> pd.DataFrame:
    """Name effect and preference level of every real brand, per (arm, category, entity, stratum,
    evidence_type).

    ``delta``: mean of L over the brand's d = 0 units, first within each fictional control (bases x swaps),
        then over controls. Logits; multiplied by c under a pure change of scale.
    ``delta_se``: SD over controls / sqrt(n_controls). The fictional control is the unit of replication: all
        units against one control share that control's pull. Brands of one category share the SAME controls,
        so these errors are correlated across brands (rtp.scale.decompose resamples accordingly).
    ``theta``, ``theta_level``, ``censored``, ``below_floor``, ``status``, ``theta_cross_<type>``: see
        ``convert``. The curve is always the one of the brand's own category.
    ``theta_direct`` and the other direct columns: see ``direct_readings``; ``has_direct`` False elsewhere.

    A brand measured against only part of the category's fictional names (the context arm, the direct
    readings) has its zero at the mean pull of THOSE names, not of all names.
    """
    cells = zero_cells(units)
    by = ["arm", "split", "category", "stratum", "evidence_type", "entity"]
    base = cells.groupby(by, sort=False).agg(delta=("delta", "mean"), sd=("delta", "std"),
                                             n_controls=("delta", "size"), n_units=("n_units", "sum")).reset_index()
    base["delta_se"] = base["sd"] / np.sqrt(base["n_controls"])
    out = convert(base.drop(columns="sd"), "delta", curves)
    direct = direct_readings(units, curves, steps)
    out = out.merge(direct, on=ENTITY_KEYS, how="outer", sort=False)
    orphan = out["delta"].isna() & out["has_direct"].eq(True)
    if orphan.any():  # d != 0 units of a brand without any d = 0 unit in its cell (a tier was not loaded)
        log.warning("%d direct-reading cell(s) have no d = 0 units of their brand", int(orphan.sum()))
        out.loc[orphan, "status"] = "no_zero_units"
        split_of = units.drop_duplicates("category").set_index("category")["split"]
        out.loc[orphan, "split"] = out.loc[orphan, "category"].map(split_of)
    out["has_direct"] = out["has_direct"].eq(True)
    out["direct_status"] = out["direct_status"].where(out["has_direct"], "none")
    for col, fill in (("censored", False), ("below_floor", False), ("matched_censored", False),
                      ("direct_swap_balanced", True), ("direct_censored", 0), ("direct_sign_changes", 0),
                      ("direct_n_levels", 0), ("direct_n_controls", 0), ("n_controls", 0), ("n_units", 0),
                      *((f"censored_cross_{t}", False) for t in LADDER_TYPES)):
        out[col] = out[col].where(out[col].notna(), fill).astype(type(fill))
    columns = ["arm", "split", "category", "entity", "stratum", "evidence_type", "delta", "delta_se", "n_controls",
               "n_units", "theta", "theta_level", "censored", "below_floor", "status", *CROSS_COLUMNS, *DIRECT_COLUMNS]
    return out[columns].reset_index(drop=True)


def fictional_theta(pulls: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """The same conversion applied to the pull of each FICTIONAL name: how many evidence units a made-up
    name is worth against the average made-up name of its category. If the anchors are clean these values
    are noise around 0, small next to the preference levels of real brands (``anchor_clean``).

    One row per (arm, category, name, stratum, evidence_type), laid out like the entity table: ``pull``,
    ``pull_se``, ``theta``, ``theta_level``, ``censored``, ``below_floor``, ``status`` and the cross columns
    of the name stratum. The pulls of the full stratum come from its d = 0 cards (shared by the types).
    """
    frame = pulls[[*KEYS, "name", "pull", "pull_se", "n", "identified"]]
    out = convert(frame, "pull", curves)
    columns = ["arm", "category", "name", "stratum", "evidence_type", "pull", "pull_se", "n", "identified", "theta",
               "theta_level", "censored", "below_floor", "status", *CROSS_COLUMNS]
    return out[columns]


def _pooled_sd(frame: pd.DataFrame, column: str) -> tuple[float, int]:
    """Pooled within-category SD (root of the mean within-category variance, ddof 1) and the values used."""
    ok = frame[frame[column].notna()]
    var = ok.groupby("category", sort=False)[column].var(ddof=1).dropna()
    return (float(np.sqrt(var.mean())) if len(var) else float("nan")), int(len(ok))


def anchor_clean(fictional: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """The "anchor clean" condition as numbers, per (arm, stratum, evidence_type, cross_stratum).

    ``sd_fictional`` / ``sd_real``: pooled within-category SD of the preference level of the fictional names
    and of the real brands; ``ratio`` = sd_fictional / sd_real. A made-up name should be worth little next
    to the spread of real brands. Censored values enter at the top of the ladder (the SD is then a lower
    bound; ``n_censored_*`` say how often). ``rms_real`` is the root mean square of the real levels about 0,
    for the case where all brands of a category sit at a similar, non-zero level. ``sd_pull``, ``sd_delta``
    and ``ratio_logit`` are the same comparison in logits: no curve involved, nothing censored.
    Name-stratum rows are cross-stratum (their theta uses the attribute-only curve of each type).
    """
    f, e = long_theta(fictional), long_theta(entities)
    keys = ["arm", "stratum", "evidence_type", "cross_stratum"]
    groups_e = {k: g for k, g in e.groupby(keys, sort=False)}
    rows = []
    for key, gf in f.groupby(keys, sort=False):
        ge = groups_e.get(key)
        row = dict(zip(keys, key))
        row["sd_fictional"], row["n_fictional"] = _pooled_sd(gf, "theta")
        row["n_censored_fictional"] = int(gf["censored"].sum())
        row["sd_pull"], _ = _pooled_sd(gf, "pull")
        if ge is None:
            row.update(sd_real=np.nan, n_real=0, n_censored_real=0, rms_real=np.nan, sd_delta=np.nan)
        else:
            row["sd_real"], row["n_real"] = _pooled_sd(ge, "theta")
            row["n_censored_real"] = int(ge["censored"].sum())
            row["rms_real"] = float(np.sqrt(np.nanmean(ge["theta"].to_numpy(dtype=float) ** 2))) if row["n_real"] else np.nan
            row["sd_delta"], _ = _pooled_sd(ge, "delta")
        row["ratio"] = row["sd_fictional"] / row["sd_real"] if row["sd_real"] and row["sd_real"] > 0 else np.nan
        row["ratio_logit"] = row["sd_pull"] / row["sd_delta"] if row["sd_delta"] and row["sd_delta"] > 0 else np.nan
        rows.append(row)
    columns = [*keys, "sd_fictional", "sd_real", "ratio", "rms_real", "n_fictional", "n_real", "n_censored_fictional",
               "n_censored_real", "sd_pull", "sd_delta", "ratio_logit"]
    return pd.DataFrame(rows, columns=columns)
