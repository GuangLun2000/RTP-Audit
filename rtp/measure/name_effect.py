"""Name effects and the evidence ladder (spec §7.2), plus the measure-stage summary.

    name_effect(pair) = mean_rot s_target(real name) - mean_rot s_target(placebo name)

computed strictly within a ``pair_id`` (identical cards, only the target name differs).
Per entity: mean name effect per evidence level, with pair counts and standard errors. Per
model: a robust (Huber) regression through the origin of the evidence-4 name effect on the
evidence-0 name effect (spec), plus two variants that separate evidence from prompt format
(evidence-4 at handicap 0 only; evidence 1 as the format baseline), each with a bootstrap CI
over entities. Two checks make the ladder interpretable: does the model respond to the card
evidence at all (slope of s_target on rating/price/review gaps), and does it answer with the
brand NAME instead of a letter (name-first-token mass)?
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
ITEM_META = ["item_id", "pair_id", "category", "pair_entity", "evidence_level", "handicap", "target_is_placebo",
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
    out = g.agg(category=("category", "first"), entity=("pair_entity", "first"),
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
    out["name_effect"] = out["s_real"] - out["s_placebo"]
    return out.sort_values("pair_id").reset_index(drop=True)


def entity_evidence_table(ne: pd.DataFrame) -> pd.DataFrame:
    """Rows (category, entity). ``ne_e{k}`` = mean name effect at evidence k (all handicaps), ``se_e{k}`` its
    standard error (pairs are independent draws), ``n_e{k}`` the pair count; ``ne_e4_h0`` = evidence 4 at
    handicap 0 only (same card quality as evidence 0-2, so only the evidence differs)."""
    keys = ["category", "entity"]
    g = ne.groupby(keys + ["evidence_level"])["name_effect"]
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


NAME_MASS_GAP_WARN = 0.02  # mean real-minus-placebo name-first-token mass of the target, per entity


def mass_by_level(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(meta[["item_id", "evidence_level", "target_is_placebo"]], on="item_id")
    df["version"] = np.where(df["target_is_placebo"], "placebo", "real")
    return (df.groupby(["evidence_level", "version"])[["letter_mass", "name_mass_target", "name_mass_other"]]
            .mean().reset_index())


def name_mass_by_entity(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per entity: target name-first-token mass in the real vs placebo version, and letter collisions."""
    df = scores.merge(meta[["item_id", "category", "pair_entity", "target_is_placebo"]], on="item_id")
    g = df.groupby(["category", "pair_entity", "target_is_placebo"])
    agg = g.agg(name_mass=("name_mass_target", "mean"), collision=("target_name_collision", "any")).reset_index()
    real = agg[~agg["target_is_placebo"]].set_index(["category", "pair_entity"])
    plac = agg[agg["target_is_placebo"]].set_index(["category", "pair_entity"])
    out = pd.DataFrame({"name_mass_real": real["name_mass"], "name_mass_placebo": plac["name_mass"],
                        "real_name_collides_with_letter": real["collision"],
                        "placebo_collides_with_letter": plac["collision"]})
    out["gap"] = out["name_mass_real"] - out["name_mass_placebo"]
    return out.reset_index().rename(columns={"pair_entity": "entity"}).sort_values("gap", ascending=False)


def consistency(scores: pd.DataFrame) -> dict[str, Any]:
    """Agreement between letter-probability argmax and the greedy generation on the checked sample."""
    chk = scores[scores["gen_text"].notna()]
    n = len(chk)
    if n == 0:
        return {"n_checked": 0, "agreement": float("nan"), "n_unparsed": 0}
    agree = (chk["gen_letter"] == chk["argmax_letter"]).mean()
    return {"n_checked": int(n), "agreement": float(agree), "n_unparsed": int(chk["gen_letter"].isna().sum())}


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
    from rtp.measure.choice import LOW_LETTER_MASS

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
    by_level = mass_by_level(scores, meta)
    nm = name_mass_by_entity(scores, meta)
    max_gap = float(nm["gap"].max()) if len(nm) else float("nan")
    gap_warn = bool(np.isfinite(max_gap) and max_gap > NAME_MASS_GAP_WARN)
    ne_credit = paired_name_effect(rotation_average(scores, meta, col="s_target_credit"))
    credit_table = entity_evidence_table(ne_credit)

    io.write_parquet(sdir / "pair_scores.parquet", pairs)
    io.write_parquet(sdir / "name_effects.parquet", ne)
    io.write_parquet(sdir / "entity_evidence.parquet", table)
    io.write_parquet(sdir / "entity_evidence_credit.parquet", credit_table)
    io.write_parquet(sdir / "handicap_effects.parquet", hcap)
    io.write_parquet(sdir / "name_mass_by_entity.parquet", nm)
    io.write_json(sdir / "ladder_regression.json", {"spec": reg, "variants": fits})
    io.write_json(sdir / "evidence_sensitivity.json", {"rows": sens, "responds_to_ratings": responds})
    io.write_json(sdir / "consistency.json", cons)

    warn = cons["n_checked"] > 0 and cons["agreement"] < cfg.measure.consistency_warn_below
    mass = float(scores["letter_mass"].mean())
    mass_warn = mass < LOW_LETTER_MASS
    lines = [f"# Measure summary: {model_id} ({split})", "",
             f"- config: {cfg.name} (data hash `{cfg.data_hash}`); items: {len(scores)}; pairs: {len(ne)}; "
             f"elapsed: {elapsed:.0f}s",
             f"- mean probability mass on letter tokens A-D at the scored position (after the \"Answer:\" prefill, "
             f"before renormalisation): {mass:.3f}"]
    if mass_warn:
        lines.append(f"- **WARNING: mean letter mass {mass:.3f} < {LOW_LETTER_MASS}: the model is not answering with "
                     f"a letter at the scored position (wrong prompt mode or template?); letter scores are mostly "
                     f"noise.**")
    lines += ["", "## Consistency check (letter argmax vs greedy generation)", "",
              f"- checked {cons['n_checked']} items ({cfg.measure.gen_check_fraction:.0%} sample); agreement "
              f"{cons['agreement']:.1%}; generations without a parsable letter: {cons['n_unparsed']}"]
    if warn:
        lines.append(f"- **WARNING: agreement {cons['agreement']:.1%} is below "
                     f"{cfg.measure.consistency_warn_below:.0%}; first-token letter scores may not reflect what "
                     f"the model would answer.**")
    lines += ["", "## Does the model use the card evidence? (manipulation check)", "",
              "OLS of the rotation-averaged s_target on how the target's card compares with the other options. "
              "A model that weighs evidence has a clearly positive rating_gap coefficient (and usually a negative "
              "price coefficient). If it does not, a flat evidence ladder means the evidence was ignored, not that "
              "it overrode reputation.", ""]
    lines += _md_table(pd.DataFrame(sens)) if sens else ["(not estimable: no evidence >= 2 pairs)"]
    if responds is False:
        lines.append("- **WARNING: the rating coefficient is not clearly positive in both versions: the model does "
                     "not respond to the rating evidence, so the evidence ladder below cannot be read as "
                     "'evidence overrides reputation'.**")
    lines += ["", "## Does the model answer with the brand NAME instead of a letter?", "",
              "Mass of each option's name-first token at the scored position (an upper bound on answering with the "
              "name). Letter scores ignore this mass; if it differs between the real and placebo versions it can "
              "hide or fake a name effect. `s_target_credit` adds it to the option's letter mass (sensitivity "
              "estimator, last table).", ""]
    lines += _md_table(by_level)
    lines += ["", "Entities with the largest real-minus-placebo target name mass:", ""]
    lines += _md_table(nm.head(10))
    if gap_warn:
        lines.append(f"- **WARNING: the target's name-first-token mass is up to {max_gap:.3f} higher in the real "
                     f"version than in the placebo version (> {NAME_MASS_GAP_WARN}); compare the letter-based name "
                     f"effects with the credited ones below before interpreting them.**")
    coll = nm[nm["real_name_collides_with_letter"]]["entity"].tolist()
    if coll:
        lines.append(f"- real brand names whose first token is an option letter (a name answer counts as that "
                     f"letter): {', '.join(coll)}")
    lines += ["", "## Position bias (before rotation averaging)", ""] + _md_table(bias)
    ne_cols = ["category", "entity", "natural_negative"] + [c for c in table.columns if c.startswith("ne_")] + \
        ["ladder_residual", "n_pairs"]
    lines += ["", "## Name effect per entity and evidence level", "",
              "Name effect = s_target(real name) - s_target(placebo name), same cards, averaged over 4 rotations. "
              "`ne_e4` averages handicaps 0-4; `ne_e4_h0` uses handicap 0 only. Positive = the real name helps.", ""]
    lines += _md_table(table[[c for c in ne_cols if c in table.columns]])
    se_cols = ["category", "entity"] + [c for c in table.columns if c.startswith(("se_", "n_e"))]
    lines += ["", "Standard errors and pair counts:", ""] + _md_table(table[se_cols])
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
    lines += ["", "## Sensitivity: name effects with name-answer mass credited to its option", ""]
    lines += _md_table(credit_table[["category", "entity"] + [c for c in credit_table.columns if c.startswith("ne_")]])
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
    return {"model_id": model_id, "split": split, "n_items": int(len(scores)), "n_pairs": int(len(ne)),
            "consistency": cons, "consistency_warning": bool(warn), "ladder_slope": reg.get("slope"),
            "ladder_variants": fits, "responds_to_ratings": responds,
            "name_mass_max_gap": round(max_gap, 4) if np.isfinite(max_gap) else None,
            "name_mass_warning": gap_warn,
            "mean_letter_mass": round(mass, 4), "letter_mass_warning": bool(mass_warn),
            "elapsed_s": round(elapsed, 1)}
