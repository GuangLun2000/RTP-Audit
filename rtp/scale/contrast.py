"""Is a stage change ONE multiplier? Contrast families, references for the null statistic, raw-share counterfactual.

Under "B = A with every logit multiplied by c" EVERY logit contrast of the two-option items grows by the same
factor: the advantage of an option position, the pull of one made-up name over another, the response to each
evidence step, the name effect of a real brand. Cycle C0 showed that these families do not share one factor
on Tulu 3 SFT -> DPO, so the answer to "did real names gain more than the scale?" depends on which family is
called the scale. This module therefore reports

* ``family_rows``: the multiplier of every family, with a t interval over categories;
* ``reference_rows``: the null statistic T1 = log(multiplier of real names) - log(multiplier of a reference),
  against every reference the design offers (the whole anchor curve, its first step, its top step, the
  made-up names of the same cell);
* ``share_counterfactual``: how much of the change of a raw choice share a uniform sharpening of A explains.

For a pure-scale variant (``A@scale=c``) every multiplier is c, every T1 is 0 and the sharpening explains all
of the change: the same wiring identity as for the preference level.

All statistics are per-category values first (categories are the unit of replication throughout the
project), then a mean of logs over categories: a geometric mean, NOT the ratio of pooled sums that ``m`` and
``c`` of rtp.scale.decompose are. Descriptive until the seed replicates give a noise band for the same numbers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from rtp.data.constants import LADDER_TYPES
from rtp.scale.anchor import ALL
from rtp.scale.decompose import _t_interval

ARM = "pair"  # two-option items only: the position bias of a four-option item is not one number
POSITION, FICTIONAL, FIRST, TOP, CURVE, REAL = ("option position", "fictional names", "evidence, first step",
                                                "evidence, top step", "evidence, whole curve", "real brand names")
FAMILIES = (POSITION, FICTIONAL, FIRST, TOP, CURVE, REAL)
GRID = np.exp(np.linspace(math.log(0.25), math.log(32.0), 141))  # candidate uniform sharpening factors
MIN_SHARE_CHANGE = 0.01  # below this change of a choice share, "explained share" is a ratio of noise


def _log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """log(b / a) per category, over the categories where both magnitudes are positive."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    ok = both[(both["a"] > 0) & (both["b"] > 0)]
    return np.log(ok["b"] / ok["a"])


def _rms(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def family_logs(a: dict[str, Any], b: dict[str, Any]) -> dict[tuple[str, str, str], pd.Series]:
    """Per (family, stratum, evidence_type): log(B / A) of the family's magnitude, one value per category.

    option position   median |order_bias| of the two-option units (evidence types pooled: ``all``)
    fictional names   root mean square of the names' pulls at d = 0 (no evidence separates the options)
    evidence, ...     the category's anchor response at the first step, at the top step, and summed over the
                      steps measured in both models (the last is c of rtp.scale.decompose, per category)
    real brand names  sum of the name effects of the brands measured in both models
    A category enters a family only when its magnitude is positive in both models.
    """
    out: dict[tuple[str, str, str], pd.Series] = {}

    def units(res: dict[str, Any]) -> pd.DataFrame:
        u = res["units"]
        return u[(u["arm"] == ARM) & u["order_bias"].notna()]

    ua, ub = units(a), units(b)
    for stratum in sorted(set(ua["stratum"]) & set(ub["stratum"])):
        mag = [g[g["stratum"] == stratum].groupby("category")["order_bias"].apply(lambda s: float(s.abs().median()))
               for g in (ua, ub)]
        out[(POSITION, stratum, "all")] = _log_ratio(*mag)

    def pulls(res: dict[str, Any]) -> pd.DataFrame:
        p = res["pulls"]
        return p[(p["arm"] == ARM) & p["identified"].astype(bool)]

    pa, pb = pulls(a), pulls(b)
    for stratum, etype in sorted(set(zip(pa["stratum"], pa["evidence_type"])) & set(zip(pb["stratum"], pb["evidence_type"]))):
        mag = [g[(g["stratum"] == stratum) & (g["evidence_type"] == etype)].groupby("category")["pull"].apply(_rms)
               for g in (pa, pb)]
        out[(FICTIONAL, stratum, etype)] = _log_ratio(*mag)

    on = ["stratum", "evidence_type", "category", "level"]

    def curves(res: dict[str, Any]) -> pd.DataFrame:
        c = res["curves"]
        return c.loc[(c["arm"] == ARM) & (c["category"] != ALL), [*on, "A_raw"]]

    both = curves(a).merge(curves(b), on=on, suffixes=("_a", "_b")).dropna(subset=["A_raw_a", "A_raw_b"])
    for (stratum, etype), g in both.groupby(["stratum", "evidence_type"], sort=True):
        lo, hi = g["level"].min(), g["level"].max()
        for family, part in ((FIRST, g[g["level"] == lo]), (TOP, g[g["level"] == hi]), (CURVE, g)):
            if family == TOP and hi == lo:
                continue
            sums = part.groupby("category")[["A_raw_a", "A_raw_b"]].sum()
            out[(family, stratum, etype)] = _log_ratio(sums["A_raw_a"], sums["A_raw_b"])

    keys = ["stratum", "evidence_type", "category", "entity"]

    def brands(res: dict[str, Any]) -> pd.DataFrame:
        e = res["entities"]
        return e.loc[(e["arm"] == ARM) & e["delta"].notna(), [*keys, "delta"]]

    both = brands(a).merge(brands(b), on=keys, suffixes=("_a", "_b"))
    for (stratum, etype), g in both.groupby(["stratum", "evidence_type"], sort=True):
        sums = g.groupby("category")[["delta_a", "delta_b"]].sum()
        out[(REAL, stratum, etype)] = _log_ratio(sums["delta_a"], sums["delta_b"])
    return out


def _exp(x: float) -> float:
    return math.exp(x) if x == x else float("nan")


def family_rows(logs: dict[tuple[str, str, str], pd.Series]) -> list[dict[str, Any]]:
    """One row per family and cell: ``multiplier`` (geometric mean over categories of B / A), its t interval
    over categories (``lo``, ``hi``) and ``n_categories``. Under uniform sharpening all rows show one number."""
    rows = []
    for (family, stratum, etype), s in sorted(logs.items(), key=lambda kv: (kv[0][1], FAMILIES.index(kv[0][0]), kv[0][2])):
        mean, lo, hi, n = _t_interval(s.to_numpy(dtype=float))
        if not n:  # no category with a positive magnitude in both models (e.g. no position bias at all)
            continue
        rows.append({"family": family, "stratum": stratum, "evidence_type": etype, "multiplier": _exp(mean),
                     "lo": _exp(lo), "hi": _exp(hi), "n_categories": n})
    return rows


def reference_rows(logs: dict[tuple[str, str, str], pd.Series]) -> list[dict[str, Any]]:
    """T1 of every real-name cell against every reference: mean over categories of log(multiplier of the real
    names) - log(multiplier of the reference), with its t interval. 0 = the real names grew like the
    reference. References: the made-up names of the same cell (no evidence carrier, no assumption across
    strata), and of each anchor curve the cell is converted on (its own stratum; the attribute-only curves
    for the name-only stratum): the whole curve (the project's c), the first step, the top step."""
    rows = []
    for (family, stratum, etype), real in sorted(logs.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if family != REAL:
            continue
        curve_stratum = "attr" if stratum == "name" else stratum
        types = [etype] if (CURVE, curve_stratum, etype) in logs else \
            [t for t in LADDER_TYPES if (CURVE, curve_stratum, t) in logs]
        refs = [("fictional names", stratum, etype, (FICTIONAL, stratum, etype))]
        for t in types:
            refs += [(label, curve_stratum, t, (fam, curve_stratum, t))
                     for label, fam in (("whole curve (c)", CURVE), ("first step", FIRST), ("top step", TOP))]
        for label, ref_stratum, ref_type, key in refs:
            if key not in logs:
                continue
            diff = (real - logs[key]).dropna()  # aligned on the category
            mean, lo, hi, n = _t_interval(diff.to_numpy(dtype=float))
            if not n:
                continue
            rows.append({"stratum": stratum, "evidence_type": etype, "reference": label, "ref_stratum": ref_stratum,
                         "ref_evidence_type": ref_type, "T1": mean, "lo": lo, "hi": hi, "n_categories": n})
    return rows


def share_counterfactual(units_a: pd.DataFrame, units_b: pd.DataFrame,
                         scale_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The raw choice share of real brands at d = 0 (two options, the single letter form of the readout), in
    A, in B, and in "A with every logit multiplied by c": how much of the change a raw-metric study would
    report is reproduced by sharpening alone.

    Per item the share is sigmoid(L_item); the two items of a unit are ``L + order_bias`` and
    ``L - order_bias``, so the position bias is sharpened together with everything else, as a real
    multiplication of the logits would do. ``share_cf`` uses the measured scale multiplier ``c``: of the cell's
    own curve when it rises in both models, otherwise the geometric mean over the curves of its stratum that do
    (the name-only stratum has no curve: the attribute-only curves). ``share_best``
    is the counterfactual share closest to B's over a grid of factors (``c_best``): what the most favourable
    uniform sharpening reaches. ``explained`` / ``explained_best`` = (counterfactual - A) / (B - A); None when
    the share moved by less than ``MIN_SHARE_CHANGE``.
    """
    def items(units: pd.DataFrame) -> pd.DataFrame:
        return units[(units["arm"] == ARM) & (units["family"] == "real") & (units["d"] == 0)
                     & units["order_bias"].notna()]

    def logits(g: pd.DataFrame) -> np.ndarray:
        level, bias = g["L"].to_numpy(dtype=float), g["order_bias"].to_numpy(dtype=float)
        return np.concatenate([level + bias, level - bias])

    def share(values: np.ndarray, c: float = 1.0) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-c * values))))

    c_of = {(r["stratum"], r["evidence_type"]): float(r["c"]) for r in scale_rows
            if r.get("arm", ARM) == ARM and r.get("valid") and r.get("c")}
    ua, ub = items(units_a), items(units_b)
    rows = []
    for (stratum, etype), ga in ua.groupby(["stratum", "evidence_type"], sort=True):
        gb = ub[(ub["stratum"] == stratum) & (ub["evidence_type"] == etype)]
        if not len(gb):
            continue
        la, lb = logits(ga), logits(gb)
        s_a, s_b = share(la), share(lb)
        curve_stratum = "attr" if stratum == "name" else stratum
        cs = [c_of[(curve_stratum, t)] for t in ([etype] if (curve_stratum, etype) in c_of else LADDER_TYPES)
              if (curve_stratum, t) in c_of]
        c = math.exp(float(np.mean(np.log(cs)))) if cs else None
        grid = np.array([share(la, float(g)) for g in GRID])
        k = int(np.argmin(np.abs(grid - s_b)))
        moved = abs(s_b - s_a) >= MIN_SHARE_CHANGE
        cf = share(la, c) if c else None
        rows.append({"stratum": stratum, "evidence_type": etype, "n_units": int(len(ga)), "share_a": s_a,
                     "share_b": s_b, "c": c, "share_cf": cf,
                     "explained": (cf - s_a) / (s_b - s_a) if moved and cf is not None else None,
                     "share_best": float(grid[k]), "c_best": float(GRID[k]),
                     "explained_best": (float(grid[k]) - s_a) / (s_b - s_a) if moved else None})
    return rows
