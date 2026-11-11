"""Name effects and the evidence ladder (spec §7.2), plus the measure-stage summary.

    name_effect(pair) = mean_rot s_target(real name) - mean_rot s_target(placebo name)

computed strictly within a ``pair_id`` (identical cards, only the target name differs).
Per entity: mean name effect per evidence level. Per model: a robust (Huber) regression
through the origin of the evidence-4 name effect on the evidence-0 name effect, with
per-entity residuals (behaviour-only decomposition, a later comparison method).
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


def rotation_average(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mean ``s_target`` over the 4 rotations of each (pair, name version). Incomplete groups are dropped."""
    df = scores[["item_id", "s_target"]].merge(meta, on="item_id", how="inner", validate="one_to_one")
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
    """Rows (category, entity), columns ``ne_e{k}`` = mean name effect at evidence level k (all handicaps)."""
    piv = ne.pivot_table(index=["category", "entity"], columns="evidence_level", values="name_effect", aggfunc="mean")
    piv.columns = [f"ne_e{int(c)}" for c in piv.columns]
    counts = ne.groupby(["category", "entity"]).size().rename("n_pairs")
    return piv.join(counts).reset_index()


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
_SIGNED_MARKERS = ("ne_e", "name_effect", "residual", "ladder_fit", "s_real", "s_placebo", "slope")


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
    meta = items_meta(items)
    pairs = rotation_average(scores, meta)
    ne = paired_name_effect(pairs)
    table = entity_evidence_table(ne)
    roles = {(r["category"], r["entity"]): r["natural_negative"] for r in catalogue if not r["is_placebo"]}
    table.insert(2, "natural_negative", [roles.get((c, e)) for c, e in zip(table["category"], table["entity"])])
    table, reg = ladder_regression(table)
    hcap = handicap_table(ne)
    cons = consistency(scores)
    bias = positional_bias(scores, meta)

    io.write_parquet(sdir / "pair_scores.parquet", pairs)
    io.write_parquet(sdir / "name_effects.parquet", ne)
    io.write_parquet(sdir / "entity_evidence.parquet", table)
    io.write_parquet(sdir / "handicap_effects.parquet", hcap)
    io.write_json(sdir / "ladder_regression.json", reg)
    io.write_json(sdir / "consistency.json", cons)

    from rtp.measure.choice import LOW_LETTER_MASS

    warn = cons["n_checked"] > 0 and cons["agreement"] < cfg.measure.consistency_warn_below
    mass = float(scores["letter_mass"].mean())
    mass_warn = mass < LOW_LETTER_MASS
    lines = [f"# Measure summary: {model_id} ({split})", "",
             f"- config: {cfg.name} (data hash `{cfg.data_hash}`); items: {len(scores)}; pairs: {len(ne)}; "
             f"elapsed: {elapsed:.0f}s",
             f"- mean probability mass on letter tokens A-D (before renormalisation): {mass:.3f}"]
    if mass_warn:
        lines.append(f"- **WARNING: mean letter mass {mass:.3f} < {LOW_LETTER_MASS}: the model is not answering with "
                     f"a letter at the scored position (wrong prompt mode or template?); letter scores are mostly "
                     f"noise.**")
    lines += [
             "", "## Consistency check (letter argmax vs greedy generation)", "",
             f"- checked {cons['n_checked']} items ({cfg.measure.gen_check_fraction:.0%} sample); agreement "
             f"{cons['agreement']:.1%}; generations without a parsable letter: {cons['n_unparsed']}"]
    if warn:
        lines.append(f"- **WARNING: agreement {cons['agreement']:.1%} is below "
                     f"{cfg.measure.consistency_warn_below:.0%}; first-token letter scores may not reflect what "
                     f"the model would answer.**")
    lines += ["", "## Position bias (before rotation averaging)", ""] + _md_table(bias)
    lines += ["", "## Name effect per entity and evidence level", "",
              "Name effect = s_target(real name) - s_target(placebo name), same cards, averaged over 4 rotations "
              "and all cards/handicaps in the level. Positive = the real name helps.", ""]
    lines += _md_table(table.drop(columns=["ladder_fit"]))
    lines += ["", "## Evidence ladder regression", "",
              f"- robust (Huber) regression through the origin of ne_e4 on ne_e0 over {reg.get('n', 0)} entities: "
              f"slope = {_fmt(reg.get('slope'))}",
              "- `ladder_residual` above = ne_e4 - slope * ne_e0 (behaviour-only decomposition; comparison method).",
              "", "## Name effect by handicap (evidence >= 3, mean over entities)", ""]
    hpiv = hcap.groupby(["evidence_level", "handicap"])["name_effect"].mean().reset_index()
    lines += _md_table(hpiv)
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
    return {"model_id": model_id, "split": split, "n_items": int(len(scores)), "n_pairs": int(len(ne)),
            "consistency": cons, "consistency_warning": bool(warn), "ladder_slope": reg.get("slope"),
            "mean_letter_mass": round(mass, 4), "letter_mass_warning": bool(mass_warn),
            "elapsed_s": round(elapsed, 1)}
