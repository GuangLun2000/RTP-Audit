"""Name effects and the evidence ladder (spec §7.2), plus the measure-stage summary.

    name_effect(pair) = mean_rot s_target(real name) - mean_rot s_target(placebo name)

computed strictly within a ``pair_id`` (identical cards, only the target name differs).
Per entity: mean name effect per evidence level, with pair counts and standard errors. Per
model: a robust (Huber) regression through the origin of the evidence-4 name effect on the
evidence-0 name effect (spec), plus two variants that separate evidence from prompt format
(evidence-4 at handicap 0 only; evidence 1 as the format baseline), each with a bootstrap CI
over entities. Checks that make the ladder interpretable: is the evidence manipulation strong
enough (within-card-set handicap response, scale-free ratio to the name effects), is the answer
format stable across evidence levels (coverage and name share vs evidence 0), and do the
estimators O5 / O3 / O4 and the imputation-free bounds agree?
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from rtp import io
from rtp.config import Config
from rtp.data.choice_items import LETTERS

log = logging.getLogger(__name__)

N_ROTATIONS = 4
ITEM_META = ["item_id", "pair_id", "card_set", "placebo_substitute", "category", "pair_entity", "evidence_level", "handicap", "target_is_placebo",
             "rotation", "target_label"]


class IncompletePairError(RuntimeError):
    pass


def items_meta(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: it[k] for k in ITEM_META} for it in items])


def rotation_average(scores: pd.DataFrame, meta: pd.DataFrame, col: str = "s_target") -> pd.DataFrame:
    """Mean ``col`` (default s_target) over the 4 rotations of each (pair, name version), returned as column
    ``s_target``. Incomplete groups are dropped."""
    df = scores[["item_id", col]].rename(columns={col: "s_target"}).merge(meta, on="item_id", how="inner",
                                                                         validate="one_to_one")
    keys = ["pair_id", "target_is_placebo"]
    g = df.groupby(keys, sort=True)
    extra = {c: (c, "first") for c in ("card_set", "placebo_substitute") if c in df}
    out = g.agg(category=("category", "first"), entity=("pair_entity", "first"), **extra,
                evidence_level=("evidence_level", "first"), handicap=("handicap", "first"),
                s_target=("s_target", "mean"), n_rotations=("rotation", "nunique"), n_items=("item_id", "size"))
    out = out.reset_index()
    bad = out[(out["n_rotations"] != N_ROTATIONS) | (out["n_items"] != N_ROTATIONS)]
    if len(bad):
        log.warning("dropping %d (pair, name) groups without exactly %d rotations", len(bad), N_ROTATIONS)
        out = out.drop(bad.index)
    return out.reset_index(drop=True)


def paired_name_effect(pair_scores: pd.DataFrame) -> pd.DataFrame:
    """Per pair: s_real - s_placebo. Only pairs with both versions contribute; nothing crosses pair_ids."""
    real = pair_scores[~pair_scores["target_is_placebo"]].set_index("pair_id")
    plac = pair_scores[pair_scores["target_is_placebo"]].set_index("pair_id")
    common = real.index.intersection(plac.index)
    dropped = len(real.index.union(plac.index)) - len(common)
    if dropped:
        log.warning("dropping %d pairs that lack one of the two name versions", dropped)
    r, p = real.loc[common], plac.loc[common]
    for col in ("category", "entity", "evidence_level", "handicap"):
        if not (r[col] == p[col]).all():
            raise IncompletePairError(f"pair metadata '{col}' differs between name versions")
    out = pd.DataFrame({
        "pair_id": common, "category": r["category"].values, "entity": r["entity"].values,
        "evidence_level": r["evidence_level"].values, "handicap": r["handicap"].values,
        "s_real": r["s_target"].values, "s_placebo": p["s_target"].values,
    })
    for k, c in enumerate(("card_set", "placebo_substitute")):
        if c in r:
            out.insert(3 + k, c, r[c].values)
    out["name_effect"] = out["s_real"] - out["s_placebo"]
    return out.sort_values("pair_id").reset_index(drop=True)


def entity_evidence_table(ne: pd.DataFrame) -> pd.DataFrame:
    """Rows (category, entity). ``ne_e{k}`` = mean name effect at evidence k (all handicaps), ``se_e{k}`` its
    standard error and ``n_e{k}`` the number of independent units. The unit is the placebo substitute: pairs
    that share a substitute share that name's own pull (at evidence 0, 24 pairs share 6 substitutes), and at
    evidence >= 1 each card set has its own substitute, so the handicap cells of one card set (same cards)
    are averaged first. Without substitute/card-set columns the unit is the pair. ``ne_e4_h0`` = evidence 4
    at handicap 0 only (same card quality as evidence 0-2, so only the evidence differs)."""
    keys = ["category", "entity"]
    unit = next((c for c in ("placebo_substitute", "card_set") if c in ne), "pair_id")
    per_unit = ne.groupby(keys + ["evidence_level", unit])["name_effect"].mean().reset_index()
    g = per_unit.groupby(keys + ["evidence_level"])["name_effect"]
    stats = g.agg(["mean", "std", "size"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["size"])
    parts = []
    for name, col in (("ne", "mean"), ("se", "se"), ("n", "size")):
        w = stats.pivot_table(index=keys, columns="evidence_level", values=col, aggfunc="first")
        w.columns = [f"{name}_e{int(c)}" for c in w.columns]
        parts.append(w)
    out = pd.concat(parts, axis=1)
    h0 = ne[(ne["evidence_level"] == 4) & (ne["handicap"] == 0)].groupby(keys)["name_effect"]
    if len(h0.size()):
        out = out.join(h0.mean().rename("ne_e4_h0")).join((h0.std() / np.sqrt(h0.size())).rename("se_e4_h0"))
    out = out.join(ne.groupby(keys).size().rename("n_pairs"))
    ordered = [c for c in out.columns if c.startswith("ne_")] + [c for c in out.columns if not c.startswith("ne_")]
    return out[ordered].reset_index()


def handicap_table(ne: pd.DataFrame, min_evidence: int = 3) -> pd.DataFrame:
    """Mean name effect per (category, entity, evidence, handicap) for evidence >= ``min_evidence``."""
    sub = ne[ne["evidence_level"] >= min_evidence]
    return (sub.groupby(["category", "entity", "evidence_level", "handicap"])["name_effect"]
            .agg(["mean", "size"]).rename(columns={"mean": "name_effect", "size": "n_pairs"}).reset_index())


def robust_origin_regression(x: np.ndarray, y: np.ndarray, epsilon: float = 1.35) -> dict[str, Any]:
    """Huber regression through the origin, y ~ slope * x. Returns slope and residuals y - slope * x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    resid = np.full(len(x), np.nan)
    if ok.sum() < 2 or np.allclose(x[ok], 0):
        return {"slope": float("nan"), "residuals": resid, "n": int(ok.sum()), "method": "huber-origin"}
    from sklearn.linear_model import HuberRegressor

    reg = HuberRegressor(epsilon=epsilon, alpha=0.0, fit_intercept=False, max_iter=1000)
    reg.fit(x[ok].reshape(-1, 1), y[ok])
    slope = float(reg.coef_[0])
    resid[ok] = y[ok] - slope * x[ok]
    return {"slope": slope, "residuals": resid, "n": int(ok.sum()), "method": "huber-origin",
            "huber_scale": float(reg.scale_), "outliers": int(reg.outliers_.sum())}


def ladder_regression(table: pd.DataFrame, lo: str = "ne_e0", hi: str = "ne_e4") -> tuple[pd.DataFrame, dict]:
    if lo not in table or hi not in table:
        return table.assign(ladder_fit=np.nan, ladder_residual=np.nan), {"slope": float("nan"), "n": 0}
    fit = robust_origin_regression(table[lo].values, table[hi].values)
    out = table.copy()
    out["ladder_fit"] = fit["slope"] * out[lo]
    out["ladder_residual"] = fit["residuals"]
    info = {k: v for k, v in fit.items() if k != "residuals"}
    return out, info


LADDER_VARIANTS = (
    ("spec: ne_e4 (all handicaps) ~ ne_e0", "ne_e0", "ne_e4"),
    ("ne_e4 (handicap 0) ~ ne_e0", "ne_e0", "ne_e4_h0"),
    ("format baseline: ne_e4 (handicap 0) ~ ne_e1", "ne_e1", "ne_e4_h0"),
)


def bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 1000,
                       level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the Huber-origin slope, resampling entities with replacement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(len(x), size=len(x))
        s_ = robust_origin_regression(x[idx], y[idx])["slope"]
        if np.isfinite(s_):
            slopes.append(s_)
    if len(slopes) < n_boot // 2:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(slopes, a)), float(np.quantile(slopes, 1 - a))


def ladder_fits(table: pd.DataFrame, seed: int, n_boot: int = 1000) -> list[dict[str, Any]]:
    out = []
    for label, lo, hi in LADDER_VARIANTS:
        if lo not in table or hi not in table:
            out.append({"variant": label, "slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                        "n": 0})
            continue
        fit = robust_origin_regression(table[lo].values, table[hi].values)
        lo_ci, hi_ci = bootstrap_slope_ci(table[lo].values, table[hi].values, seed, n_boot)
        out.append({"variant": label, "slope": fit["slope"], "ci_low": lo_ci, "ci_high": hi_ci, "n": fit["n"]})
    return out


def pair_covariates(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Per (pair, version): how the target's card compares with the other three options.

    rating_gap = target rating - mean other rating (<= 0; = -delta[handicap]); log_price_ratio and
    log_review_ratio = log(target / mean of the others). NaN where the field is hidden.
    """
    rows = []
    for it in items:
        if it["rotation"] != 0:
            continue
        tgt = next(o for o in it["options"] if o["role"] == "target")
        oth = [o for o in it["options"] if o["role"] != "target"]

        def gap(field, log=False):
            if tgt[field] is None:
                return float("nan")
            m = float(np.mean([o[field] for o in oth]))
            return float(np.log(tgt[field] / m)) if log else float(tgt[field] - m)

        rows.append({"pair_id": it["pair_id"], "target_is_placebo": it["target_is_placebo"],
                     "rating_gap": gap("rating"), "log_price_ratio": gap("price", log=True),
                     "log_review_ratio": gap("n_reviews", log=True)})
    return pd.DataFrame(rows)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and classical standard errors."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if n <= k:
        return beta, np.full(k, np.nan)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def evidence_sensitivity(pairs: pd.DataFrame, cov: pd.DataFrame) -> list[dict[str, Any]]:
    """Manipulation check: does the model's score for the target respond to the card evidence?

    For each name version separately, OLS of the rotation-averaged s_target on the target's rating gap,
    log price ratio and log review ratio (evidence >= 3), and on the log price ratio alone (evidence 2).
    A model that weighs the evidence has a clearly positive rating coefficient. If it does not, a flat
    evidence ladder means "evidence ignored", not "reputation overridden".
    """
    df = pairs.merge(cov, on=["pair_id", "target_is_placebo"], how="inner")
    out = []
    for version, sub in df.groupby("target_is_placebo"):
        vname = "placebo" if version else "real"
        e3 = sub[sub["evidence_level"] >= 3].dropna(subset=["rating_gap", "log_price_ratio", "log_review_ratio"])
        if len(e3) > 4:
            X = np.column_stack([np.ones(len(e3)), e3["rating_gap"], e3["log_price_ratio"], e3["log_review_ratio"]])
            b, se = ols(X, e3["s_target"].values)
            for name, bi, si in zip(("rating_gap", "log_price_ratio", "log_review_ratio"), b[1:], se[1:]):
                out.append({"version": vname, "evidence": ">=3", "covariate": name, "coef": float(bi), "se": float(si),
                            "n_pairs": int(len(e3))})
        e2 = sub[sub["evidence_level"] == 2].dropna(subset=["log_price_ratio"])
        if len(e2) > 2:
            b, se = ols(np.column_stack([np.ones(len(e2)), e2["log_price_ratio"]]), e2["s_target"].values)
            out.append({"version": vname, "evidence": "2", "covariate": "log_price_ratio", "coef": float(b[1]),
                        "se": float(se[1]), "n_pairs": int(len(e2))})
    return out


def responds_to_ratings(sens: list[dict[str, Any]]) -> bool | None:
    """True if the rating coefficient is > 2 SE above zero in BOTH versions (None if not estimable)."""
    rows = [r for r in sens if r["covariate"] == "rating_gap"]
    if len(rows) < 2 or any(not np.isfinite(r["se"]) for r in rows):
        return None
    return all(r["coef"] - 2 * r["se"] > 0 for r in rows)


# --------------------------------------------------------------------------- answer-format diagnostics
FORMAT_SHIFT_WARN = 0.2  # |change vs evidence 0| of the free-position name share or coverage
MANIPULATION_RATIO_MIN = 0.25  # provisional (PI to confirm): placebo per-star handicap response / median |ne_e0|
DIAG_COLS = ["c_free", "c_forced", "name_share_free", "name_share_forced", "leak_free", "leak_forced",
             "pruned_free", "pruned_forced"]


def _with_meta(scores: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = scores.merge(meta[["item_id"] + cols], on="item_id")
    if "target_is_placebo" in df:
        df["version"] = np.where(df["target_is_placebo"], "placebo", "real")
    return df


def coverage_by_level(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mean coverage, name share, identity leak and pruned mass per (evidence level, version)."""
    df = _with_meta(scores, meta, ["evidence_level", "target_is_placebo"])
    cols = [c for c in DIAG_COLS if c in df]
    return df.groupby(["evidence_level", "version"])[cols].mean().reset_index()


def format_stability(cov: pd.DataFrame, max_shift: float = FORMAT_SHIFT_WARN) -> list[dict[str, Any]]:
    """Levels whose free-position answer FORMAT differs from evidence 0 (name share or coverage shifted by
    more than ``max_shift``): there the ladder compares readouts of different answer formats."""
    out = []
    for version, sub in cov.groupby("version"):
        base = sub[sub["evidence_level"] == 0]
        if base.empty:
            continue
        for _, r in sub[sub["evidence_level"] != 0].iterrows():
            for col in ("name_share_free", "c_free"):
                d = float(r[col] - base[col].iloc[0])
                if np.isfinite(d) and abs(d) > max_shift:
                    out.append({"version": version, "evidence_level": int(r["evidence_level"]), "column": col,
                                "e0": float(base[col].iloc[0]), "value": float(r[col]), "shift": d})
    return out


def channel_by_entity(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per entity: share of the target's free answers given by NAME, and identity leak, per version."""
    df = _with_meta(scores, meta, ["category", "pair_entity", "target_is_placebo"])
    g = df.groupby(["category", "pair_entity", "version"])
    agg = g.agg(name=("target_name_free", "sum"), mass=("target_mass_free", "sum"),
                leak=("leak_free", "mean")).reset_index()
    agg["name_share"] = agg["name"] / agg["mass"].where(agg["mass"] > 0)
    w = agg.pivot_table(index=["category", "pair_entity"], columns="version", values=["name_share", "leak"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index().rename(columns={"pair_entity": "entity"})
    if {"name_share_real", "name_share_placebo"} <= set(w.columns):
        w["name_share_gap"] = w["name_share_real"] - w["name_share_placebo"]
    return w


def estimator_table(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per entity ne_e0 / ne_e4 under O5 (primary), O3 (free only) and O4 (forced only)."""
    parts = []
    for tag, col in (("o5", "s_target"), ("o3", "s_target_free"), ("o4", "s_target_forced")):
        if col not in scores:
            continue
        t = entity_evidence_table(paired_name_effect(rotation_average(scores, meta, col=col)))
        keep = [c for c in ("ne_e0", "ne_e4") if c in t]
        parts.append(t.set_index(["category", "entity"])[keep].add_suffix(f"_{tag}"))
    return pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame()


def estimator_agreement(est: pd.DataFrame) -> list[dict[str, Any]]:
    """Spearman correlation of entity name effects between O5 and O3 / O4 (evidence 0 and 4)."""
    out = []
    for lvl in ("ne_e0", "ne_e4"):
        for other in ("o3", "o4"):
            a, b = f"{lvl}_o5", f"{lvl}_{other}"
            if a in est and b in est:
                ok = est[[a, b]].dropna()
                rho = float(ok[a].rank().corr(ok[b].rank())) if len(ok) >= 3 else float("nan")
                out.append({"level": lvl, "vs": other, "spearman": rho, "n": int(len(ok))})
    return out


def name_effect_bounds(scores: pd.DataFrame, meta: pd.DataFrame, level: int = 0) -> dict[str, Any]:
    """Imputation-free bounds of each entity's name effect at ``level``: P_target lies in
    [q_free_t, q_free_t + 1 - c_free] per item (the deferred mass could go anywhere), so the rotation-averaged
    pair effect lies in [lo_real - hi_placebo, hi_real - lo_placebo]; averaged per entity."""
    if "s_target_lo" not in scores:
        return {"n_entities": 0}
    lo = rotation_average(scores, meta, col="s_target_lo").rename(columns={"s_target": "lo"})
    hi = rotation_average(scores, meta, col="s_target_hi").rename(columns={"s_target": "hi"})
    df = lo.merge(hi[["pair_id", "target_is_placebo", "hi"]], on=["pair_id", "target_is_placebo"])
    df = df[df["evidence_level"] == level]
    r = df[~df["target_is_placebo"]].set_index("pair_id")
    p = df[df["target_is_placebo"]].set_index("pair_id")
    common = r.index.intersection(p.index)
    pair = pd.DataFrame({"entity": r.loc[common, "entity"], "category": r.loc[common, "category"],
                         "lo": r.loc[common, "lo"] - p.loc[common, "hi"], "hi": r.loc[common, "hi"] - p.loc[common, "lo"]})
    ent = pair.groupby(["category", "entity"])[["lo", "hi"]].mean()
    if ent.empty:
        return {"n_entities": 0}
    excl = (ent["lo"] > 0) | (ent["hi"] < 0)
    return {"level": level, "n_entities": int(len(ent)), "share_excluding_zero": float(excl.mean()),
            "median_width": float((ent["hi"] - ent["lo"]).median())}


def handicap_response(pairs: pd.DataFrame, deltas: list[float], seed: int = 0) -> list[dict[str, Any]]:
    """Within-card-set response of s_target to the handicap (evidence >= 3), per name version: least squares
    through the origin of s(h) - s(h=0) on the stars removed, delta[h]; reported as the drop in s_target
    per star (positive = the model penalises a worse rating). Matched cards make this a clean manipulation
    effect: the two cells differ only in the target's rating. SE: cluster bootstrap over card sets."""
    if "card_set" not in pairs:
        return []
    out = []
    df = pairs[pairs["evidence_level"] >= 3]
    for version, sub in df.groupby("target_is_placebo"):
        base = sub[sub["handicap"] == 0].set_index(["card_set", "evidence_level"])["s_target"]
        rest = sub[sub["handicap"] > 0].copy()
        rest["d"] = rest["s_target"].values - base.reindex(list(zip(rest["card_set"], rest["evidence_level"]))).values
        rest["x"] = [deltas[int(h)] for h in rest["handicap"]]
        rest = rest.dropna(subset=["d"])
        if rest.empty or float((rest["x"] ** 2).sum()) == 0:
            continue

        def slope(r: pd.DataFrame) -> float:
            return float(-(r["x"] * r["d"]).sum() / (r["x"] ** 2).sum())

        b = slope(rest)
        sets = rest["card_set"].unique()
        groups = {k: g for k, g in rest.groupby("card_set")}
        rng = np.random.default_rng(seed)
        boots = [slope(pd.concat([groups[k] for k in rng.choice(sets, size=len(sets))])) for _ in range(500)]
        out.append({"version": "placebo" if version else "real", "per_star_drop": b,
                    "se": float(np.std(boots, ddof=1)) if len(sets) > 1 else float("nan"),
                    "n_card_sets": int(len(sets)), "n_cells": int(len(rest))})
    return out


def manipulation_ratio(resp: list[dict[str, Any]], table: pd.DataFrame) -> float:
    """Scale-free strength of the evidence manipulation: the placebo version's per-star handicap response
    divided by the median |ne_e0| over entities (how far one star moves the score, in units of a typical
    name effect). NaN if not estimable."""
    plac = [r for r in resp if r["version"] == "placebo"]
    if not plac or "ne_e0" not in table:
        return float("nan")
    scale = float(np.nanmedian(np.abs(table["ne_e0"].values))) if len(table) else float("nan")
    return float(plac[0]["per_star_drop"] / scale) if scale and np.isfinite(scale) and scale > 0 else float("nan")


def _chance(a: pd.Series, b: pd.Series) -> float:
    """Agreement expected from the two marginal letter distributions alone (unparsed answers never match)."""
    n = len(a)
    if n == 0:
        return float("nan")
    fa, fb = a.value_counts() / n, b.value_counts() / n
    return float(sum(fa.get(L, 0.0) * fb.get(L, 0.0) for L in LETTERS))


def consistency(scores: pd.DataFrame) -> dict[str, Any]:
    """Spec §7.1 check on the 2% sample: the FREE greedy answer vs the scored argmax.

    Overall agreement is against the primary (O5) argmax; unparsed answers count as disagreement. Per mode:
    immediate answers vs the free-position argmax (O3), deferred answers vs the forced-position argmax (O4,
    a direct test of the O5 imputation). Each comes with its chance level from the marginal frequencies."""
    chk = scores[scores["gen_text"].notna()]
    n = len(chk)
    out: dict[str, Any] = {"n_checked": int(n), "agreement": float("nan"), "chance": float("nan"), "n_unparsed": 0}
    if n == 0:
        return out
    out.update(agreement=float((chk["gen_letter"] == chk["argmax_letter"]).mean()),
               chance=_chance(chk["gen_letter"], chk["argmax_letter"]), n_unparsed=int(chk["gen_letter"].isna().sum()))
    if "gen_mode" in chk:
        for mode, ref in (("immediate", "argmax_free"), ("deferred", "argmax_forced")):
            sub = chk[chk["gen_mode"] == mode]
            out[f"n_{mode}"] = int(len(sub))
            if len(sub) and ref in sub:
                out[f"agreement_{mode}"] = float((sub["gen_letter"] == sub[ref]).mean())
                out[f"chance_{mode}"] = _chance(sub["gen_letter"], sub[ref])
                out[f"n_unparsed_{mode}"] = int(sub["gen_letter"].isna().sum())
    return out


def positional_bias(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mean letter probability per position, and mean target probability by target position."""
    df = scores.merge(meta[["item_id", "target_label"]], on="item_id")
    rows = []
    for L in LETTERS:
        sub = df[df["target_label"] == L]
        tp = sub.apply(lambda r: r[f"p_{L}"], axis=1).mean() if len(sub) else float("nan")
        rows.append({"letter": L, "mean_p": float(df[f"p_{L}"].mean()), "mean_p_target_when_here": float(tp)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summary
_SIGNED_MARKERS = ("ne_e", "name_effect", "residual", "ladder_fit", "s_real", "s_placebo", "slope", "coef", "gap",
                   "ci_")


def _pct(v: Any) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.1%}"


def _fmt(v: Any, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:.0f}"
        return f"{v:+.3f}" if signed else f"{v:.3f}"
    return str(v)


def _md_table(df: pd.DataFrame) -> list[str]:
    """Markdown table; effect-like columns (name effects, residuals) get an explicit sign."""
    signed = [any(m in str(c) for m in _SIGNED_MARKERS) for c in df.columns]
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, s) for v, s in zip(row, signed)) + " |")
    return lines


def summarize_measure(cfg: Config, model_id: str, split: str, items: list[dict[str, Any]],
                      catalogue: list[dict[str, Any]], scores: pd.DataFrame, sdir, elapsed: float) -> dict[str, Any]:
    from rtp.measure.choice import DEFERRED_MAX_TOKENS, LOW_COVERAGE

    meta = items_meta(items)
    pairs = rotation_average(scores, meta)
    ne = paired_name_effect(pairs)
    table = entity_evidence_table(ne)
    roles = {(r["category"], r["entity"]): r["natural_negative"] for r in catalogue if not r["is_placebo"]}
    table.insert(2, "natural_negative", [roles.get((c, e)) for c, e in zip(table["category"], table["entity"])])
    table, reg = ladder_regression(table)
    fits = ladder_fits(table, cfg.seed)
    hcap = handicap_table(ne)
    cons = consistency(scores)
    bias = positional_bias(scores, meta)
    sens = evidence_sensitivity(pairs, pair_covariates(items))
    responds = responds_to_ratings(sens)
    resp = handicap_response(pairs, cfg.data.handicap_rating_deltas, cfg.seed)
    ratio = manipulation_ratio(resp, table)
    cov = coverage_by_level(scores, meta)
    shifts = format_stability(cov)
    chan = channel_by_entity(scores, meta)
    est = estimator_table(scores, meta)
    est_agree = estimator_agreement(est)
    bounds = name_effect_bounds(scores, meta, level=0)

    io.write_parquet(sdir / "pair_scores.parquet", pairs)
    io.write_parquet(sdir / "name_effects.parquet", ne)
    io.write_parquet(sdir / "entity_evidence.parquet", table)
    io.write_parquet(sdir / "estimators_by_entity.parquet", est)
    io.write_parquet(sdir / "handicap_effects.parquet", hcap)
    io.write_parquet(sdir / "coverage_by_level.parquet", cov)
    io.write_parquet(sdir / "answer_channel_by_entity.parquet", chan)
    io.write_json(sdir / "ladder_regression.json", {"spec": reg, "variants": fits})
    io.write_json(sdir / "evidence_sensitivity.json", {"rows": sens, "responds_to_ratings": responds,
                                                      "handicap_response": resp, "manipulation_ratio": ratio})
    io.write_json(sdir / "consistency.json", cons)
    io.write_json(sdir / "answer_format.json", {"format_shifts": shifts, "estimator_agreement": est_agree,
                                               "bounds_e0": bounds})

    warn = cons["n_checked"] > 0 and cons["agreement"] < cfg.measure.consistency_warn_below
    c_free, c_forced = float(scores["c_free"].mean()), float(scores["c_forced"].mean())
    cov_warn = c_forced < LOW_COVERAGE
    manip_warn = bool(np.isfinite(ratio) and ratio < MANIPULATION_RATIO_MIN)
    lines = [f"# Measure summary: {model_id} ({split})", "",
             f"- config: {cfg.name} (data hash `{cfg.data_hash}`); items: {len(scores)}; pairs: {len(ne)}; "
             f"elapsed: {elapsed:.0f}s",
             "- scores: answer-path estimator O5 (free answers at the spec position; the deferred rest imputed by "
             "the committed answer after \"Answer:\"), counting letter AND displayed-name answers.",
             "", "## Answer coverage and format", "",
             f"Coverage c = probability that the next tokens start an answer naming one of the four options (letter "
             f"or name). Mean c_free (spec position) = {c_free:.3f}; mean c_forced (after \"Answer:\") = "
             f"{c_forced:.3f}. 1 - c_free is the deferred (imputed) share. name_share = fraction of the option "
             f"answers given by brand NAME. leak = mass on the target slot's ABSENT identity (credited to no "
             f"option). pruned = upper bound on answer mass lost to trie pruning.", ""]
    lines += _md_table(cov)
    if cov_warn:
        lines.append(f"- **WARNING: mean c_forced {c_forced:.3f} < {LOW_COVERAGE}: even after \"Answer:\" the model "
                     f"mostly does not name an option (wrong prompt mode or template?).**")
    if c_free < LOW_COVERAGE:
        lines.append(f"- note: c_free {c_free:.3f} < {LOW_COVERAGE}: {1 - c_free:.0%} of the free-position mass "
                     f"defers the answer, so O5 relies on the forced readout for that share; compare O3/O4 and the "
                     f"bounds below.")
    if shifts:
        lines.append(f"- **WARNING: the free-position answer format changes with the evidence level (shift > "
                     f"{FORMAT_SHIFT_WARN} vs evidence 0); the ladder then compares different readouts:**")
        lines += [""] + _md_table(pd.DataFrame(shifts))
    lines += ["", "## Consistency check (spec §7.1: free greedy answer vs the scored argmax)", "",
              f"- checked {cons['n_checked']} items ({cfg.measure.gen_check_fraction:.0%} sample), generated from the "
              f"free position; agreement with the O5 argmax {_pct(cons['agreement'])} (chance "
              f"{_pct(cons['chance'])}); answers naming no option: {cons['n_unparsed']}",
              f"- immediate answers (option named within {cfg.measure.gen_max_new_tokens} tokens): "
              f"{cons.get('n_immediate', 0)}, agreement with the free argmax "
              f"{_pct(cons.get('agreement_immediate'))} (chance {_pct(cons.get('chance_immediate'))})",
              f"- deferred answers (continued to {DEFERRED_MAX_TOKENS} tokens, concluding answer parsed): "
              f"{cons.get('n_deferred', 0)}, agreement with the forced argmax "
              f"{_pct(cons.get('agreement_deferred'))} (chance {_pct(cons.get('chance_deferred'))}; unparsed "
              f"{cons.get('n_unparsed_deferred', 0)}). This tests the O5 imputation (deferred answer = committed "
              f"answer after \"Answer:\")."]
    if warn:
        lines.append(f"- **WARNING: agreement {cons['agreement']:.1%} is below "
                     f"{cfg.measure.consistency_warn_below:.0%}; the scores may not reflect what the model would "
                     f"answer.**")
    lines += ["", "## Estimators and bounds", "",
              "Entity name effects under O5 (primary), O3 (free position only, renormalised over immediate "
              "answers) and O4 (forced position only). Spearman correlation across entities:", ""]
    lines += _md_table(pd.DataFrame(est_agree)) if est_agree else ["(not estimable)"]
    if bounds.get("n_entities"):
        lines.append(f"- imputation-free bounds at evidence 0 (deferred mass assigned adversarially): "
                     f"{bounds['share_excluding_zero']:.0%} of {bounds['n_entities']} entities have bounds excluding "
                     f"0; median width {bounds['median_width']:.2f} logits.")
    lines += ["", "## Does the model use the card evidence? (manipulation check)", "",
              "Within matched card sets, the drop in s_target per star removed from the target's rating (handicap "
              "cells vs handicap 0, evidence >= 3; cluster-bootstrap SE over card sets). The manipulation ratio "
              "divides the placebo version's per-star drop by the median |ne_e0|: below "
              f"{MANIPULATION_RATIO_MIN} (provisional), one star moves the score by less than a quarter of a typical "
              "name effect, and a flat ladder means the evidence was too weak, not that reputation resisted it.", ""]
    lines += _md_table(pd.DataFrame(resp)) if resp else ["(not estimable: no matched handicap cells)"]
    lines.append(f"- manipulation ratio: {_fmt(ratio)}")
    if manip_warn:
        lines.append(f"- **WARNING: manipulation ratio {ratio:.3f} < {MANIPULATION_RATIO_MIN}: the evidence "
                     f"manipulation is weak relative to the name effects; do not read the ladder as 'evidence "
                     f"overrides reputation'.**")
    lines += ["", "Across pairs (OLS of the rotation-averaged s_target on the target's card gaps):", ""]
    lines += _md_table(pd.DataFrame(sens)) if sens else ["(not estimable: no evidence >= 2 pairs)"]
    if responds is False:
        lines.append("- **WARNING: the rating coefficient is not clearly positive in both versions: the model does "
                     "not respond to the rating evidence, so the evidence ladder below cannot be read as "
                     "'evidence overrides reputation'.**")
    if len(chan):
        lines += ["", "## Answer channel and identity leak per entity (free position)", "",
                  "name_share = share of the target's answer mass given by NAME; leak = mass on the absent identity "
                  "(placebo version: the model names the real brand that is not on the list). Name answers are "
                  "credited to their option, so a real-vs-placebo gap here is information, not a bias in O5.", ""]
        sort_col = "name_share_gap" if "name_share_gap" in chan else chan.columns[-1]
        lines += _md_table(chan.sort_values(sort_col, ascending=False).head(10))
    lines += ["", "## Position bias (before rotation averaging)", ""] + _md_table(bias)
    ne_cols = ["category", "entity", "natural_negative"] + [c for c in table.columns if c.startswith("ne_")] + \
        ["ladder_residual", "n_pairs"]
    lines += ["", "## Name effect per entity and evidence level", "",
              "Name effect = s_target(real name) - s_target(placebo name), same cards, averaged over 4 rotations. "
              "`ne_e4` averages handicaps 0-4; `ne_e4_h0` uses handicap 0 only. Positive = the real name helps.", ""]
    lines += _md_table(table[[c for c in ne_cols if c in table.columns]])
    se_cols = ["category", "entity"] + [c for c in table.columns if c.startswith(("se_", "n_e"))]
    lines += ["", "Standard errors and independent units (placebo substitutes; at evidence >= 1 one card set each):", ""]
    lines += _md_table(table[se_cols])
    lines += ["", "## Evidence ladder regressions (Huber, through the origin; 95% bootstrap CI over entities)", ""]
    lines += _md_table(pd.DataFrame(fits))
    lines += ["- `ladder_residual` above = ne_e4 - slope * ne_e0 for the spec variant (behaviour-only "
              "decomposition; comparison method).",
              "- The format-baseline variant uses evidence 1 (description only, no quality information) as the "
              "reference, so a drop caused merely by the longer prompt format does not look like evidence "
              "overriding reputation.",
              "", "## Name effect by handicap (evidence >= 3, mean over entities)", ""]
    hpiv = hcap.groupby(["evidence_level", "handicap"])["name_effect"].mean().reset_index()
    lines += _md_table(hpiv)
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
    rnd = lambda v: round(float(v), 4) if v is not None and np.isfinite(v) else None  # noqa: E731
    return {"model_id": model_id, "split": split, "n_items": int(len(scores)), "n_pairs": int(len(ne)),
            "consistency": cons, "consistency_warning": bool(warn), "ladder_slope": reg.get("slope"),
            "ladder_variants": fits, "responds_to_ratings": responds,
            "c_free": rnd(c_free), "c_forced": rnd(c_forced), "coverage_warning": bool(cov_warn),
            "format_shifts": len(shifts), "format_warning": bool(shifts),
            "manipulation_ratio": rnd(ratio), "manipulation_warning": manip_warn,
            "bounds_e0": bounds, "estimator_agreement": est_agree,
            "elapsed_s": round(elapsed, 1)}
