"""Content vs scale of a choice preference (research objective v1118, sections 3, 4 and 6).

Choice model for a target e against a control at adverse-evidence level d::

    P(choose e | d) = sigmoid(gamma * (p_e - w(d)))

Observable are the name effect ``Delta_e = gamma * p_e`` (logits), the anchor curve ``A(d) = gamma * w(d)``
(the response to evidence between two FICTIONAL names) and the preference level ``theta_e = A^{-1}(Delta_e)``
in evidence units. Multiplying every logit by c multiplies Delta and A by c and leaves theta unchanged.

Pure CPU, pandas in and out; no model is loaded and torch is never imported.

    readout    items + letter logits  ->  one order-averaged logit difference per unit
    anchor     fictional vs fictional ->  name pulls, anchor curves, their inverse
    theta      real vs fictional      ->  name effects and preference levels (converted and direct)
    raw        the scale-dependent metrics of prior work, on the same units
    checks     the four single-model conditions as numbers
    decompose  model A -> model B: scale multiplier, null model "uniform sharpening", residuals

``analyze_model`` runs the single-model part; ``compare_models`` takes two of its results.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from rtp.scale.anchor import anchor_curves, anchor_pairs, invert, name_pulls
from rtp.scale.checks import _num, condition_checks, records
from rtp.scale.decompose import (calibrate, null_tests, pair_sums, scale_multipliers, shape_ratio, t_quantile,
                                 uncertainty)
from rtp.scale.raw import raw_metrics
from rtp.scale.readout import canonical_form, unit_table
from rtp.scale.theta import anchor_clean, entity_table, fictional_theta, zero_cells

__all__ = ["analyze_model", "analyze_units", "anchor_clean", "anchor_curves", "anchor_pairs", "calibrate",
           "canonical_form", "compare_models", "condition_checks", "entity_table", "fictional_theta", "invert",
           "name_pulls", "null_tests", "pair_sums", "raw_metrics", "scale_multipliers", "shape_ratio", "t_quantile",
           "uncertainty", "unit_table", "zero_cells"]

# T1_t_lo / T1_t_hi is the primary interval (t over per-category values); T1_cat_mean is its centre (categories
# weighted equally, not the pooled T1) and n_cat_T1 the number of categories it could use.
SUMMARY_FIELDS = ("n_entities", "n_categories", "n_negative_a", "c", "c_lo", "c_hi", "log_c", "log_c_cat_mean",
                  "log_c_t_lo", "log_c_t_hi", "n_cat_log_c", "m", "m_lo", "m_hi", "T1", "T1_cat_mean", "T1_t_lo",
                  "T1_t_hi", "n_cat_T1", "T1_lo", "T1_hi", "T2", "T2_ratio", "median_d_theta", "mean_d_theta",
                  "n_d_theta", "n_d_theta_below_floor", "c_lo_crossed", "c_hi_crossed", "T1_lo_crossed",
                  "T1_hi_crossed")


def analyze_units(units: pd.DataFrame, steps: dict[str, list[float]]) -> dict[str, Any]:
    """Everything one model's unit table gives (see ``analyze_model``)."""
    pulls = name_pulls(units)
    pairs = anchor_pairs(units)
    curves = anchor_curves(units, steps, pairs=pairs)
    entities = entity_table(units, curves, steps)
    fictional = fictional_theta(pulls, curves)
    return {"units": units, "curves": curves, "pulls": pulls, "entities": entities, "fictional": fictional,
            "raw": raw_metrics(units), "checks": condition_checks(units, curves, entities, fictional),
            "anchor_clean": anchor_clean(fictional, entities), "pairs": pairs, "cells": zero_cells(units)}


def analyze_model(items: Sequence[dict[str, Any]], scores: pd.DataFrame, steps: dict[str, list[float]],
                  form: str | None = None) -> dict[str, Any]:
    """Single-model analysis of the ladder stage's ``scores`` on ``items``.

    ``steps``: ``rtp.data.ladder_items.type_steps(cfg)``, the natural unit of every ladder level.

    Returns ``units`` (rtp.scale.readout.unit_table), ``curves`` and ``pulls`` (rtp.scale.anchor),
    ``entities`` and ``fictional`` (rtp.scale.theta), ``raw`` (rtp.scale.raw), ``checks`` (a dict,
    rtp.scale.checks), and three tables the pair comparison reuses: ``anchor_clean``, ``pairs`` (anchor value
    per pair of fictional names) and ``cells`` (name effect per brand and control).
    """
    return analyze_units(unit_table(items, scores, form=form), steps)


def compare_models(a: dict[str, Any], b: dict[str, Any], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Stage comparison A -> B from two ``analyze_model`` results measured on the same items.

    ``scale``: scale multipliers per curve and pooled (``scale_multipliers``), the pooled rows with their
        intervals. ``null``: c, m, T1, T2 per (arm, stratum, evidence_type) with the category bootstrap, the
        crossed bootstrap and the t interval (``null_tests`` + ``uncertainty``). ``residuals``: per brand.
        ``shape``: per-level ratio of the two curves. ``summary``: the headline numbers as a JSON-serialisable
        dict, including H1 (the spread of log c over the evidence types of the attribute-only stratum).
    """
    mult = scale_multipliers(a["curves"], b["curves"])
    tests, residuals = null_tests(a["entities"], b["entities"], mult)
    sums = pair_sums(mult, residuals)
    have = all(k in x for x in (a, b) for k in ("pairs", "cells"))
    crossed = {"pairs_a": a["pairs"], "pairs_b": b["pairs"], "cells_a": a["cells"], "cells_b": b["cells"]} if have else None
    intervals, h1 = uncertainty(sums, n_boot=n_boot, seed=seed, crossed=crossed)
    keys = ["arm", "stratum", "evidence_type", "cross_stratum"]
    null = tests.merge(intervals, on=keys, how="left") if len(tests) else tests
    own = intervals[~intervals["cross_stratum"].astype(bool)] if len(intervals) else intervals
    ci_cols = [c for c in own.columns if c.startswith(("c_lo", "c_hi", "log_c_"))]
    scale = mult.merge(own[["arm", "stratum", "evidence_type", *ci_cols]].assign(category="(all)"),
                       on=["arm", "stratum", "evidence_type", "category"], how="left") if len(own) else mult
    summary = {
        "n_boot": int(n_boot), "seed": int(seed), "crossed_bootstrap": bool(have),
        "cells": [{**{k: _num(row[k]) for k in keys}, **{f: _num(row[f]) for f in SUMMARY_FIELDS if f in row}}
                  for row in null.to_dict(orient="records")],
        "scale": records(scale.loc[scale["category"] == "(all)", ["arm", "stratum", "evidence_type", "c", "log_c",
                                                                  "valid", "n_categories", "n_levels"]]),
        "H1": {arm: {k: ({t: _num(v) for t, v in val.items()} if isinstance(val, dict) else _num(val))
                     for k, val in entry.items()} for arm, entry in h1.items()},
    }
    return {"scale": scale, "null": null, "residuals": residuals, "shape": shape_ratio(a["curves"], b["curves"]),
            "summary": summary}
