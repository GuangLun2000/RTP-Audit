"""The scale-DEPENDENT metrics that prior work reports, computed on the same units.

They are here to be compared with the preference level, not to be interpreted by themselves: each of them
moves when every logit is multiplied by a constant, with no change in what the model prefers. Two of them
come in an argmax variant. Under an EXACT rescaling of all logits the argmax of every single item is
unchanged, so the argmax variants do not move in that idealised case; they still depend on scale in real
comparisons, where the share of items decided by noise changes with decisiveness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rtp.scale.theta import NO_TYPE, SHARED_ZERO_STRATA

RAW_COLUMNS = ["arm", "stratum", "evidence_type", "metric", "d", "value", "n"]
CELL = ["arm", "category", "stratum", "target_entity", "control_entity", "base", "swap"]


def _n_eff(share: pd.Series) -> float:
    """exp(entropy) of the distribution proportional to ``share``: the effective number of brands."""
    p = share.to_numpy(dtype=float)
    total = p.sum()
    if not total > 0:
        return float("nan")
    p = p[p > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def raw_metrics(units: pd.DataFrame) -> pd.DataFrame:
    """Long table of raw metrics on the real-brand units, per (arm, stratum, evidence_type).

    At d = 0 (``d`` = 0; the name and full strata carry evidence_type "none"):

    * ``choice_share_prob``: mean ``p_target``, the brand's share of the forced two-way probability.
    * ``choice_share_argmax``: mean ``pick_target``, the share of items in which the brand's letter wins.
    * ``n_eff``: concentration over the real brands of a category: exp(entropy) of the distribution
      proportional to each brand's mean ``p_target``; computed per category, then averaged (``n`` =
      categories). Lower = more concentrated.

    At each level d != 0, over the cells (brand x control x base x swap) measured at d = 0 AND at d:

    * ``flip_rate_prob``: sum p0 * (1 - pd) / sum p0: the probability that a brand drawn at d = 0 is not
      drawn at level d, under independent sampling from the two-way probabilities.
    * ``flip_rate_argmax``: among cells where the brand is picked at d = 0 (order-averaged ``p_target`` above
      one half), the share where it is no longer picked at level d (``n`` = cells picked at d = 0).
    """
    real = units[units["family"] == "real"]
    rows: list[dict] = []
    zero = real[real["d"] == 0]
    for key, g in zero.groupby(["arm", "stratum", "evidence_type"], sort=False):
        head = dict(zip(("arm", "stratum", "evidence_type"), key), d=0)
        rows.append({**head, "metric": "choice_share_prob", "value": float(g["p_target"].mean()), "n": len(g)})
        rows.append({**head, "metric": "choice_share_argmax", "value": float(g["pick_target"].mean()), "n": len(g)})
        share = g.groupby(["category", "target_entity"], sort=False)["p_target"].mean()
        per_category = share.groupby(level="category", sort=False).apply(_n_eff).dropna()
        rows.append({**head, "metric": "n_eff", "value": float(per_category.mean()) if len(per_category) else np.nan,
                     "n": int(len(per_category))})
    moved = real[real["d"] != 0]
    if len(moved):
        moved = moved[[*CELL, "evidence_type", "d", "p_target"]].copy()
        moved["zero_type"] = np.where(moved["stratum"].isin(SHARED_ZERO_STRATA), NO_TYPE, moved["evidence_type"])
        at_zero = zero[[*CELL, "evidence_type", "p_target"]].rename(
            columns={"evidence_type": "zero_type", "p_target": "p0"})
        paired = moved.merge(at_zero, on=[*CELL, "zero_type"], how="inner")
        for key, g in paired.groupby(["arm", "stratum", "evidence_type", "d"], sort=True):
            head = dict(zip(("arm", "stratum", "evidence_type", "d"), key))
            head["d"] = int(head["d"])
            p0, pd_ = g["p0"].to_numpy(dtype=float), g["p_target"].to_numpy(dtype=float)
            rows.append({**head, "metric": "flip_rate_prob", "n": len(g),
                         "value": float(np.sum(p0 * (1.0 - pd_)) / np.sum(p0)) if np.sum(p0) > 0 else np.nan})
            picked = p0 > 0.5
            rows.append({**head, "metric": "flip_rate_argmax", "n": int(picked.sum()),
                         "value": float(np.mean(pd_[picked] <= 0.5)) if picked.any() else np.nan})
    return pd.DataFrame(rows, columns=RAW_COLUMNS)
