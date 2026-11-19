"""The four conditions the preference level rests on (research objective, section 3.4), as numbers.

Each condition gets its numbers and one pass flag against a threshold below: True, False, or None when the
check could not be evaluated (no cell of the loaded tiers carries it; e.g. the direct readings live in the
``ext`` tier). A None is never read as a failure. THE THRESHOLDS ARE
PROVISIONAL: they were fixed before any confirmation model was measured and exist so that a report cannot
pass a condition by looking at the result. Changing one is a design decision (one line in DECISIONS.md),
never a tuning knob. A failed flag does not hide anything: the numbers are always reported.

Common scale (H1) is a property of a PAIR of models and lives in rtp.scale.decompose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from rtp.scale.anchor import ALL, KEYS
from rtp.scale.theta import anchor_clean

COMMON_CHANNEL_MAX_MEDIAN_DIFF = 0.5
"""Common channel passes when the median |theta_direct - theta_matched| is at most this many ladder LEVELS
(level-index units, so the three evidence types share one tolerance)."""
ANCHOR_SD_RATIO_MAX = 0.5
"""Anchor clean passes when the SD of the fictional names' preference levels is below this fraction of the
SD of the real brands' levels, in every (arm, stratum, evidence_type) where the ratio is a VALUE: defined,
and with at most ``CENSORED_SHARE_MAX`` of the real brands beyond the ladder. Censored brands enter at the
top of the ladder, which squeezes their SD: the ratio of such a row is an upper bound. It cannot fail the check
(a row that fails only because the ladder is too short would send the next cycle off to replace names that are
fine); a bound below the limit is still a pass."""
CENSORED_SHARE_MAX = 0.10
"""Ladder passes when at most this share of brand cells is worth more than the whole ladder."""
COVERAGE_MIN = 0.9
"""Readout passes when the mean letter mass at the forced position is at least this (same value as
rtp.measure.ladder.LOW_COVERAGE)."""


def _num(value: Any) -> Any:
    """JSON-safe scalar: NaN and infinities become None."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows of a table as JSON-safe dicts."""
    return [{k: _num(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _common_channel(entities: pd.DataFrame) -> dict[str, Any]:
    direct = entities[entities["has_direct"]]
    reasons = {
        "no_zero_units": direct["direct_status"].eq("no_zero") | direct["status"].eq("no_zero_units"),
        "direct_beyond_top": direct["direct_censored"] > 0,
        "direct_beyond_favourable": direct["direct_censored"] < 0,
        "conversion_censored": direct["matched_censored"],
        "no_or_flat_curve": direct["theta_matched"].isna() & direct["delta_matched"].notna(),
    }
    both = direct[direct["theta_direct"].notna() & direct["theta_matched"].notna() & ~direct["matched_censored"]]
    rows = []
    for key, g in both.groupby(["arm", "stratum", "evidence_type"], sort=False):
        diff = (g["theta_direct"] - g["theta_matched"]).to_numpy(dtype=float)
        diff_level = (g["theta_direct_level"] - g["theta_matched_level"]).to_numpy(dtype=float)
        rows.append({**dict(zip(("arm", "stratum", "evidence_type"), key)), "n_cells": len(g),
                     "median_abs_diff": float(np.median(np.abs(diff))),
                     "median_abs_diff_level": float(np.median(np.abs(diff_level))),
                     "mean_diff_level": float(np.mean(diff_level)),
                     "corr": _corr(g["theta_direct_level"].to_numpy(dtype=float),
                                   g["theta_matched_level"].to_numpy(dtype=float)),
                     "n_swap_unbalanced": int((~g["direct_swap_balanced"]).sum())})
    diff_level = (both["theta_direct_level"] - both["theta_matched_level"]).to_numpy(dtype=float)
    unmatched = (both["theta_direct_level"] - both["theta_level"]).to_numpy(dtype=float)
    median = float(np.median(np.abs(diff_level))) if len(both) else float("nan")
    return {
        "n_direct_cells": int(len(direct)), "n_cells": int(len(both)),
        "dropped": {k: int(v.sum()) for k, v in reasons.items()},
        "median_abs_diff_level": _num(median),
        "corr": _num(_corr(both["theta_direct_level"].to_numpy(dtype=float),
                           both["theta_matched_level"].to_numpy(dtype=float))),
        # against the conversion that uses ALL controls of the brand (not like-for-like: other controls)
        "median_abs_diff_level_all_controls": _num(float(np.nanmedian(np.abs(unmatched))) if len(both) else float("nan")),
        "by_cell": records(pd.DataFrame(rows)),
        "pass": bool(median <= COMMON_CHANNEL_MAX_MEDIAN_DIFF) if len(both) else None,
    }


def _anchor_clean(fictional: pd.DataFrame, entities: pd.DataFrame) -> dict[str, Any]:
    table = anchor_clean(fictional, entities)
    share = table["n_censored_real"] / table["n_real"].where(table["n_real"] > 0)
    table = table.assign(bound_only=table["ratio"].notna() & ~(share <= CENSORED_SHARE_MAX))
    defined = table[table["ratio"].notna()]
    usable, bound = defined[~defined["bound_only"]], defined[defined["bound_only"]]
    unidentified = fictional[~fictional["identified"].astype(bool)]
    worst = float(usable["ratio"].max()) if len(usable) else float("nan")
    proven = bool(len(bound)) and bool((bound["ratio"] < ANCHOR_SD_RATIO_MAX).all())  # a bound below the limit passes
    flag = bool(worst < ANCHOR_SD_RATIO_MAX) if len(usable) else (True if proven else None)
    return {
        "rows": records(table), "n_rows": int(len(table)), "n_rows_undefined": int(len(table) - len(defined)),
        # rows whose real brands are censored: the ratio is an upper bound and does not decide the flag
        "n_rows_bound_only": int(len(bound)),
        "max_ratio_bound_only": _num(float(bound["ratio"].max()) if len(bound) else float("nan")),
        "n_names_unidentified": int(len(unidentified)), "max_ratio": _num(worst),
        # the same comparison in logits (SD of pulls / SD of name effects): no curve, nothing censored
        "max_ratio_logit": _num(float(table["ratio_logit"].max()) if table["ratio_logit"].notna().any() else float("nan")),
        "pass": flag,
    }


def _ladder(curves: pd.DataFrame, entities: pd.DataFrame) -> dict[str, Any]:
    per_curve = curves[curves["category"] != ALL].sort_values("level").groupby(KEYS, sort=False).agg(
        monotone_raw=("monotone_raw", "first"), top=("A_iso", "max"), first=("A_iso", "first"),
        dropped=("n_pairs_dropped", "sum")).reset_index()
    own = entities[entities["status"].isin(["ok", "censored"])]  # brand cells converted with their own curve
    flat = entities[entities["status"].eq("flat_curve")]
    rows = []
    for key, g in entities[entities["status"].isin(["ok", "censored", "flat_curve"])].groupby(
            ["arm", "stratum", "evidence_type"], sort=False):
        conv = g[g["status"] != "flat_curve"]
        curve_key = per_curve[(per_curve["arm"] == key[0]) & (per_curve["stratum"] == key[1])
                              & (per_curve["evidence_type"] == key[2])]
        rows.append({**dict(zip(("arm", "stratum", "evidence_type"), key)), "n_entities": len(conv),
                     "n_entities_on_flat_curves": int(len(g) - len(conv)),
                     "n_curves": int(len(curve_key)), "n_curves_flat": int((~(curve_key["top"] > 0)).sum()),
                     "share_censored": float(conv["censored"].mean()) if len(conv) else float("nan"),
                     "share_below_floor": float(conv["below_floor"].mean()) if len(conv) else float("nan")})
    share = float(own["censored"].mean()) if len(own) else float("nan")
    return {
        "n_curves": int(len(per_curve)),
        "share_monotone_raw": _num(float(per_curve["monotone_raw"].mean()) if len(per_curve) else float("nan")),
        "n_curves_flat": int((~(per_curve["top"] > 0)).sum()),  # no positive response anywhere: nothing to invert
        # the finest step is not resolved (its fitted value is 0) although the curve rises later
        "n_curves_first_step_flat": int(((per_curve["top"] > 0) & ~(per_curve["first"] > 0)).sum()),
        "n_entities_on_flat_curves": int(len(flat)),
        "n_pairs_dropped": int(per_curve["dropped"].sum()),
        "n_entities": int(len(own)), "share_censored": _num(share),
        "share_below_floor": _num(float(own["below_floor"].mean()) if len(own) else float("nan")),
        "dropped": {"flat_curve": int(len(flat)), "no_curve": int(entities["status"].eq("no_curve").sum()),
                    "no_value": int(entities["status"].eq("no_value").sum()),
                    "no_zero_units": int(entities["status"].eq("no_zero_units").sum()),
                    "cross_stratum_only": int(entities["status"].eq("cross_stratum").sum())},
        "by_cell": records(pd.DataFrame(rows)),
        "pass": bool(share <= CENSORED_SHARE_MAX) if len(own) else None,
    }


def _readout(units: pd.DataFrame) -> dict[str, Any]:
    mass = units[units["L_mass"].notna()]
    coverage = float(units["c_forced"].mean()) if len(units) else float("nan")
    diff = (mass["L"] - mass["L_mass"]).to_numpy(dtype=float)
    return {
        "n_units": int(len(units)), "mean_c_forced": _num(coverage),
        "mean_c_free": _num(float(units["c_free"].mean()) if len(units) else float("nan")),
        "share_low_coverage": _num(float((units["c_forced"] < COVERAGE_MIN).mean()) if len(units) else float("nan")),
        "n_mass_units": int(len(mass)), "n_units_without_mass": int(len(units) - len(mass)),
        "corr_L_mass": _num(_corr(mass["L"].to_numpy(dtype=float), mass["L_mass"].to_numpy(dtype=float))),
        "median_abs_diff_L_mass": _num(float(np.median(np.abs(diff))) if len(mass) else float("nan")),
        "share_letter_collision": _num(float(units["letter_collision"].mean()) if len(units) else float("nan")),
        "median_abs_order_bias": _num(float(units["order_bias"].abs().median()) if units["order_bias"].notna().any()
                                      else float("nan")),
        "pass": bool(coverage >= COVERAGE_MIN) if len(units) else None,
    }


def condition_checks(units: pd.DataFrame, curves: pd.DataFrame, entities: pd.DataFrame,
                     fictional: pd.DataFrame) -> dict[str, Any]:
    """The four single-model conditions as a JSON-serialisable dict (undefined numbers are None; every block
    says how many cells it could not use, and why).

    ``common_channel``: do the two readings of a preference level agree? Median |theta_direct -
        theta_matched| in ladder levels and their correlation, over the brand cells that have both and are
        not censored (``dropped`` counts the others by reason). ``theta_matched`` is the conversion of the
        d = 0 point of the SAME units the direct reading uses.
    ``anchor_clean``: is a made-up name worth little? The SD ratio of ``rtp.scale.theta.anchor_clean``;
        passes when every ratio that is a value is below ``ANCHOR_SD_RATIO_MAX``. Rows with more than
        ``CENSORED_SHARE_MAX`` of the real brands beyond the ladder give only an upper bound
        (``n_rows_bound_only``, ``max_ratio_bound_only``) and cannot fail the check; with no usable row the
        check passes if every bound is below the limit and is otherwise not evaluated: the ladder has to be
        extended first.
    ``ladder``: does the ladder work as a ruler? Share of category curves whose raw means rise strictly,
        share of brand cells beyond the top of their curve (pass at most ``CENSORED_SHARE_MAX``), share
        below the first resolved step (the ladder cannot tell them from 0). Curves that never rise are
        counted apart (``n_curves_flat``; their brand cells have no preference level and are NOT censored):
        a flat curve means the evidence does not move the model, which a longer ladder would not cure.
    ``readout``: is the letter readout the model's answer? Mean letter mass at the forced position (pass at
        least ``COVERAGE_MIN``), agreement of ``L`` with the answer-path readout ``L_mass`` on the diagnostic
        sample, share of units with a letter-colliding name.
    """
    return {
        "thresholds": {"COMMON_CHANNEL_MAX_MEDIAN_DIFF": COMMON_CHANNEL_MAX_MEDIAN_DIFF,
                       "ANCHOR_SD_RATIO_MAX": ANCHOR_SD_RATIO_MAX, "CENSORED_SHARE_MAX": CENSORED_SHARE_MAX,
                       "COVERAGE_MIN": COVERAGE_MIN, "provisional": True},
        "common_channel": _common_channel(entities),
        "anchor_clean": _anchor_clean(fictional, entities),
        "ladder": _ladder(curves, entities),
        "readout": _readout(units),
    }
