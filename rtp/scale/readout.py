"""Unit table: one order-averaged letter-logit difference per comparison.

A UNIT is one comparison (target vs control, one stratum, one evidence level) shown at every option order.
For one item the readout is ``L_item = z[target letter] - z[control letter]``, two logits of ONE letter form
at the forced position. The unit value ``L`` is the mean of ``L_item`` over the unit's orders.

Why the mean over orders is the right unit value: an additive advantage of an option POSITION adds
``b[position of target] - b[position of control]`` to ``L_item``. A two-option unit shows the target at A
and at B, so the two terms are ``+(b_A - b_B)`` and ``-(b_A - b_B)``; a four-option unit rotates cyclically,
the target at position o and the control at o + 1, so the terms telescope to zero. In both cases the
position bias cancels exactly, whatever its size.

Everything here is a linear function of the stored logits, so multiplying every logit by c multiplies ``L``
by c. The probability columns (``p_target``, ``pick_target``) are NOT linear: they are kept only for the
scale-dependent metrics of prior work (rtp.scale.raw).

This module must stay importable without torch: it reimplements the six-line canonical-form rule of
``rtp.measure.ladder.canonical_form`` (that module imports torch); a test asserts that the two agree.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
MAX_OPTIONS = len(LETTERS)
FORMS = ("sp", "ns")  # " A" and "A"
ARM_PAIR, ARM_CTX = "pair", "ctx"  # two-option comparisons / the same comparison among four options
ARM_OF_N_OPTIONS = {2: ARM_PAIR, 4: ARM_CTX}
UNIT_META = ("family", "tier", "split", "category", "stratum", "evidence_type", "d", "target_entity",
             "target_is_fictional", "control_entity", "base", "swap")
UNIT_COLUMNS = ("unit_id", "arm", *UNIT_META, "n_options", "form", "L", "order_bias", "c_forced", "c_free",
                "p_target", "pick_target", "letter_collision", "L_mass")


class ReadoutError(ValueError):
    """Items and scores do not form complete units (a missing order, a missing score, mixed models, ...)."""


def canonical_form(scores: pd.DataFrame) -> str:
    """The letter form (" A" = ``sp``, "A" = ``ns``) the analysis reads: the one whose logit is higher on
    average over items and options, i.e. the likelier one at the forced position. A fixed rule, never chosen
    by looking at results. Same rule as ``rtp.measure.ladder.canonical_form``."""
    sp = np.stack([scores[f"z_sp_{j}"].to_numpy(dtype=float) for j in range(MAX_OPTIONS)], axis=1)
    ns = np.stack([scores[f"z_ns_{j}"].to_numpy(dtype=float) for j in range(MAX_OPTIONS)], axis=1)
    if np.all(np.isnan(sp)):
        return "ns"
    if np.all(np.isnan(ns)):
        return "sp"
    return "sp" if np.nanmean(sp - ns) >= 0 else "ns"


def _some(values: Sequence[Any], k: int = 5) -> str:
    values = list(values)
    return ", ".join(str(v) for v in values[:k]) + (f", ... ({len(values)} in total)" if len(values) > k else "")


def _item_frame(items: Sequence[dict[str, Any]]) -> pd.DataFrame:
    if not len(items):
        raise ReadoutError("no ladder items given")
    frame = pd.DataFrame({
        "item_id": [it["item_id"] for it in items],
        "unit_id": [it["unit_id"] for it in items],
        "order": [int(it["order"]) for it in items],
        "n_shown": [len(it["options"]) for it in items],
        "target_pos": [LETTERS.index(it["target_label"]) for it in items],
        "control_pos": [LETTERS.index(it["control_label"]) for it in items],
        **{k: [it[k] for it in items] for k in UNIT_META},
    })
    dup = frame["item_id"][frame["item_id"].duplicated()]
    if len(dup):
        raise ReadoutError(f"duplicate item_id among the items: {_some(dup.unique())}")
    bad = sorted(set(frame["n_shown"]) - set(ARM_OF_N_OPTIONS))
    if bad:
        raise ReadoutError(f"items with {bad} options: only {sorted(ARM_OF_N_OPTIONS)} are part of the design")
    return frame


def unit_table(items: Sequence[dict[str, Any]], scores: pd.DataFrame, form: str | None = None) -> pd.DataFrame:
    """One row per unit: its metadata and the order-averaged readout of ONE model.

    Columns
    -------
    ``arm``: ``pair`` (two options; tiers core and ext) or ``ctx`` (four options). The arms show different
        prompts, so every later table keeps them apart.
    ``L``: mean over the unit's orders of z[target] - z[control] in letter form ``form`` (default: the
        canonical form of these scores). Logits; position bias cancels (module docstring).
    ``order_bias``: two-option units only: half the difference between the order with the target at A and the
        order with the target at B, i.e. the logit advantage of position A. NaN for four-option units.
    ``c_forced``, ``c_free``: mean letter mass at the forced / free position (readout coverage).
    ``p_target``: mean over orders of the forced two-way choice probability exp(lp_t) / (exp(lp_t) +
        exp(lp_c)) from the merged letter log-probs. Scale-dependent.
    ``pick_target``: share of orders in which the target letter has the higher merged log-prob (a tie counts
        one half). Scale-dependent across items, not within one.
    ``letter_collision``: an option name of the unit starts with an option-letter token.
    ``L_mass``: mean over orders of log q_forced[target] - log q_forced[control] (answer-path masses, letters
        AND names) when every order of the unit was in the diagnostic sample; NaN otherwise.

    Raises ``ReadoutError`` when an item has no score, a unit lacks an order, the scores belong to more than
    one model, or the scored positions contradict the items.
    """
    meta = _item_frame(items)
    need = ["item_id", "n_options", "target_idx", "control_idx", "c_forced", "c_free",
            *(f"lp_forced_{j}" for j in range(MAX_OPTIONS)),
            *(f"z_{f}_{j}" for f in FORMS for j in range(MAX_OPTIONS))]
    missing_cols = [c for c in need if c not in scores.columns]
    if missing_cols:
        raise ReadoutError(f"scores lack column(s) {missing_cols}")
    dup = scores["item_id"][scores["item_id"].duplicated()]
    if len(dup):
        raise ReadoutError(f"duplicate item_id among the scores: {_some(dup.unique())}")
    if "model_id" in scores.columns and scores["model_id"].nunique(dropna=False) > 1:
        raise ReadoutError(f"scores of more than one model: {_some(scores['model_id'].unique())}; "
                           f"a unit table describes one model")
    extra = len(scores) - int(scores["item_id"].isin(meta["item_id"]).sum())
    if extra:
        log.info("%d scored items are not among the given items and are ignored", extra)
    df = meta.merge(scores, on="item_id", how="left", indicator=True, validate="one_to_one")
    lacking = df.loc[df["_merge"] != "both", "item_id"]
    if len(lacking):
        raise ReadoutError(f"{len(lacking)} item(s) have no score: {_some(lacking)}")
    wrong = df[(df["target_idx"] != df["target_pos"]) | (df["control_idx"] != df["control_pos"])
               | (df["n_options"] != df["n_shown"])]
    if len(wrong):
        raise ReadoutError(f"scores and items disagree on the target/control position or the number of options "
                           f"(different data builds?): {_some(wrong['item_id'])}")
    if form is None:
        form = canonical_form(df)
    if form not in FORMS:
        raise ReadoutError(f"unknown letter form '{form}', expected one of {FORMS}")

    rows = np.arange(len(df))
    t = df["target_idx"].to_numpy(dtype=int)
    c = df["control_idx"].to_numpy(dtype=int)
    z = df[[f"z_{form}_{j}" for j in range(MAX_OPTIONS)]].to_numpy(dtype=float)
    l_item = z[rows, t] - z[rows, c]
    if np.isnan(l_item).any():
        raise ReadoutError(f"letter form '{form}' has no logit for the target or control letter of "
                           f"{int(np.isnan(l_item).sum())} item(s): {_some(df['item_id'][np.isnan(l_item)])}")
    lp = df[[f"lp_forced_{j}" for j in range(MAX_OPTIONS)]].to_numpy(dtype=float)
    gap = lp[rows, t] - lp[rows, c]
    if np.isnan(gap).any():
        raise ReadoutError(f"{int(np.isnan(gap).sum())} item(s) lack merged letter log-probs: "
                           f"{_some(df['item_id'][np.isnan(gap)])}")
    p_item = 1.0 / (1.0 + np.exp(-gap))
    pick_item = (gap > 0).astype(float) + 0.5 * (gap == 0)
    mass_item = np.full(len(df), np.nan)
    if all(f"q_forced_{j}" in df.columns for j in range(MAX_OPTIONS)):
        q = df[[f"q_forced_{j}" for j in range(MAX_OPTIONS)]].to_numpy(dtype=float)
        qt, qc = q[rows, t], q[rows, c]
        ok = (qt > 0) & (qc > 0)  # NaN (not sampled) and zero masses: no mass readout for the unit
        mass_item[ok] = np.log(qt[ok]) - np.log(qc[ok])
    collide = (df["letter_collision"].fillna(False).to_numpy(dtype=bool) if "letter_collision" in df.columns
               else np.zeros(len(df), dtype=bool))

    codes, unit_ids = pd.factorize(df["unit_id"], sort=False)
    n_units = len(unit_ids)
    first = np.full(n_units, -1, dtype=int)
    first[codes[::-1]] = rows[::-1]  # first item of each unit
    count = np.bincount(codes, minlength=n_units)
    order = df["order"].to_numpy(dtype=int)
    n_opt = df["n_shown"].to_numpy(dtype=int)[first]
    if (order < 0).any() or (order >= MAX_OPTIONS).any():
        raise ReadoutError(f"option order outside 0..{MAX_OPTIONS - 1}: {_some(df['item_id'][(order < 0) | (order > 3)])}")
    seen = np.bincount(codes, weights=np.power(2.0, order), minlength=n_units)
    incomplete = (count != n_opt) | (seen != np.power(2.0, n_opt) - 1.0)
    if incomplete.any():
        raise ReadoutError(f"{int(incomplete.sum())} unit(s) lack an option order (every unit needs one item per "
                           f"position, or the position bias does not cancel): {_some(unit_ids[incomplete])}")
    group = df.groupby(list(UNIT_META), sort=False, dropna=False).ngroup().to_numpy()
    mixed = group != group[first][codes]
    if mixed.any():
        raise ReadoutError(f"unit metadata differs between option orders: {_some(df['unit_id'][mixed].unique())}")

    def mean(values: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=values, minlength=n_units) / count

    at_a = np.bincount(codes, weights=np.where(order == 0, l_item, 0.0), minlength=n_units)
    at_b = np.bincount(codes, weights=np.where(order == 1, l_item, 0.0), minlength=n_units)
    out = df.iloc[first][list(UNIT_META)].reset_index(drop=True)
    out.insert(0, "unit_id", np.asarray(unit_ids))
    out.insert(1, "arm", [ARM_OF_N_OPTIONS[int(n)] for n in n_opt])
    out["n_options"] = n_opt
    out["form"] = form
    out["L"] = mean(l_item)
    out["order_bias"] = np.where(n_opt == 2, (at_a - at_b) / 2.0, np.nan)
    out["c_forced"] = mean(df["c_forced"].to_numpy(dtype=float))
    out["c_free"] = mean(df["c_free"].to_numpy(dtype=float))
    out["p_target"] = mean(p_item)
    out["pick_target"] = mean(pick_item)
    out["letter_collision"] = np.bincount(codes, weights=collide, minlength=n_units) > 0
    out["L_mass"] = mean(mass_item)  # NaN as soon as one order has no mass readout
    for col in ("d", "base", "swap"):
        out[col] = out[col].astype(int)
    out["target_is_fictional"] = out["target_is_fictional"].astype(bool)
    return out[list(UNIT_COLUMNS)]
