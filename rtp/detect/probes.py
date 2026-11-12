"""M3 linear probes, control task and the knowledge subspace K (spec §7.5), plus the diagnostics that say
whether K can do its M4 job. Hard constraint 1: reads the activations of ONE model; no base model.

Probes: per frame ("background" = spec text, "choice" = background + choice prompt, read at the option
line), per extracted layer, per attribute, a ridge classifier on standardised activations (one SVD per
training set serves every attribute, every alpha and the control task). Evaluation:

* grouped CV by entity on the training (dev, not held-out) categories, alpha chosen by that CV;
* control task (spec): attribute vectors permuted ACROSS entities, same CV -> selectivity = real - control;
* held-out dev categories (``probe.heldout_categories``): accuracy on unseen categories and entities, also
  across frames (trained on one frame, tested on the other);
* binding controls (held-out categories): two-brand texts (read X, also describes Y) and undescribed
  names (describes X, reads Z). A probe that reads the entity should match X, and be at chance for Z.

K (spec §7.5): per frame and layer, the top principal components of the attribute-level mean activations
followed by the probe weight vectors (mapped back to activation space), orthonormalised by QR and
truncated to ``k_rank``. K is exported for every extracted layer; M4 selects the layer.

Diagnostics for M4 (computed on the evidence-0 choice activations of real vs placebo targets):
* real-brand transfer: the price-tier probe applied to real brands vs the catalogue tier (Spearman);
* familiarity: does the popularity probe rank real brands above placebos (AUC), and does that readout
  track the measured evidence-0 name effect;
* K capture: share of ||h_real - h_placebo||^2 that lies in K, against random subspaces of the same rank.
If K captures no more than a random subspace, delta_k is ~0 by construction and the M4 identity residual
would equal the whole name effect; the summary says so instead of letting M4 proceed silently.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_catalogue
from rtp.data.probe_items import ATTRIBUTES, N_LEVELS
from rtp.log import add_file_log
from rtp.paths import stage_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

TIER_LEVEL = {"budget": 0, "mid": 1, "premium": 2}  # catalogue tier -> price_tier level
SELECTIVITY_WARN = 0.05
BINDING_WARN = 0.05
CAPTURE_RATIO_WARN = 2.0
AUC_WARN = 0.60


# --------------------------------------------------------------------------- estimators
class RidgeSVD:
    """Ridge regression on standardised features via one thin SVD; ``fit`` is then cheap for any target/alpha."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        self.sd = sd
        self.U, self.S, self.Vt = np.linalg.svd((X - self.mu) / sd, full_matrices=False)

    def fit(self, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        Y = np.asarray(Y, dtype=np.float64)
        b = Y.mean(axis=0)
        f = self.S / (self.S ** 2 + alpha)
        W = self.Vt.T @ (f[:, None] * (self.U.T @ (Y - b)))
        return W, b

    def scores(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd) @ W + b

    def raw_directions(self, W: np.ndarray) -> np.ndarray:
        """Columns of W as directions in activation space (w . (x - mu) / sd = (w / sd) . x + const)."""
        return W / self.sd[:, None]


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return out


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that never split an entity; entities are assigned to folds at random (deterministic)."""
    uniq = np.array(sorted(set(groups.tolist())))
    k = max(2, min(n_folds, len(uniq)))
    perm = rng_for(seed, "probe-folds").permutation(len(uniq))
    fold_of = {g: i % k for i, g in zip(range(len(uniq)), uniq[perm])}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def permuted_labels(entities: np.ndarray, labels: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Control task: assign each entity the attribute vector of another entity (all attributes jointly)."""
    uniq = sorted(set(entities.tolist()))
    first = {e: int(np.where(entities == e)[0][0]) for e in uniq}
    perm = rng_for(seed, "probe-control").permutation(len(uniq))
    source = {e: uniq[j] for e, j in zip(uniq, perm)}
    return {a: np.array([y[first[source[e]]] for e in entities]) for a, y in labels.items()}


def subspace_basis(directions: list[np.ndarray], k_rank: int, tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (d x <=k_rank) of the given direction columns, in order, via QR."""
    C = np.column_stack([d / (np.linalg.norm(d) + 1e-12) for d in directions])
    Q, R = np.linalg.qr(C)
    diag = np.abs(np.diag(R))
    keep = diag > tol * diag.max() if len(diag) else np.zeros(0, dtype=bool)
    return Q[:, keep][:, :k_rank]


def class_mean_pcs(X: np.ndarray, y: np.ndarray, k: int) -> list[np.ndarray]:
    """Top principal directions of the per-level mean activations (centred), at most n_levels - 1."""
    means = np.stack([X[y == lv].mean(axis=0) for lv in sorted(set(y.tolist()))])
    if len(means) < 2:
        return []
    _, _, vt = np.linalg.svd(means - means.mean(axis=0), full_matrices=False)
    return [vt[i] for i in range(min(k, len(means) - 1))]


def build_k(X: np.ndarray, labels: dict[str, np.ndarray], fitted: dict[str, Any], k_rank: int) -> np.ndarray:
    """K = orthonormal basis of [level-mean PCs of every attribute, then the probes' weight directions].

    The PCs come first: they are where attribute information is ENCODED (what an M4 transplant moves);
    ridge weights are discriminative read-out directions and only fill the remaining rank.
    """
    directions: list[np.ndarray] = []
    for a in ATTRIBUTES:
        directions.extend(class_mean_pcs(X, labels[a], N_LEVELS[a] - 1))
    for a in ATTRIBUTES:
        directions.extend(fitted["full"].raw_directions(fitted["attrs"][a]["W"]).T)
    return subspace_basis(directions, k_rank)


def capture_ratio(D: np.ndarray, K: np.ndarray) -> float:
    """Share of sum ||d||^2 that lies in span(K) (K orthonormal)."""
    tot = float((D ** 2).sum())
    return float(((D @ K) ** 2).sum() / tot) if tot > 0 else float("nan")


def random_capture(D: np.ndarray, rank: int, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        Q, _ = np.linalg.qr(rng.standard_normal((D.shape[1], rank)))
        vals.append(capture_ratio(D, Q))
    return float(np.mean(vals)), float(np.std(vals))


# --------------------------------------------------------------------------- stage
def _labels(index: pd.DataFrame, prefix: str = "y_") -> dict[str, np.ndarray]:
    return {a: index[f"{prefix}{a}"].to_numpy() for a in ATTRIBUTES}


def _acc(scores: np.ndarray, y: np.ndarray) -> float:
    return float((scores.argmax(axis=1) == y).mean()) if len(y) else float("nan")


def fit_frame_layer(X: np.ndarray, ents: np.ndarray, labels: dict[str, np.ndarray], alphas: list[float],
                    n_folds: int, seed: int) -> dict[str, Any]:
    """Grouped-CV alpha selection for real and permuted labels, then a full fit. Returns models and metrics."""
    control = permuted_labels(ents, labels, seed)
    folds = grouped_folds(ents, n_folds, seed)
    correct = {(kind, a, al): 0 for kind in ("real", "control") for a in ATTRIBUTES for al in alphas}
    total = 0
    for tr, va in folds:
        m = RidgeSVD(X[tr])
        total += len(va)
        for kind, lab in (("real", labels), ("control", control)):
            for a in ATTRIBUTES:
                Y = one_hot(lab[a][tr], N_LEVELS[a])
                for al in alphas:
                    W, b = m.fit(Y, al)
                    correct[(kind, a, al)] += int((m.scores(X[va], W, b).argmax(axis=1) == lab[a][va]).sum())
    full = RidgeSVD(X)
    out: dict[str, Any] = {"full": full, "attrs": {}}
    for a in ATTRIBUTES:
        best = max(alphas, key=lambda al: correct[("real", a, al)])
        best_c = max(alphas, key=lambda al: correct[("control", a, al)])
        W, b = full.fit(one_hot(labels[a], N_LEVELS[a]), best)
        out["attrs"][a] = {"alpha": best, "alpha_control": best_c, "W": W, "b": b,
                           "cv_acc": correct[("real", a, best)] / total,
                           "control_acc": correct[("control", a, best_c)] / total,
                           "chance": float(np.bincount(labels[a]).max() / len(labels[a]))}
    return out


def run_probe(cfg: Config, model_id: str, force: bool = False, prompt_mode: str = "auto") -> Path:
    """Train probes on the dev activations of ``model_id``, export K, write diagnostics. Dev split only."""
    from rtp.detect.activations import extract_stage_name, load_activations

    split = "dev"
    pr = cfg.probe
    edir = stage_dir(cfg.name, model_id, extract_stage_name(split))
    sdir = stage_dir(cfg.name, model_id, "probe")
    shash = stages.stage_hash(cfg, "probe", model_id, split, prompt_mode)
    if io.finished_stage(sdir, shash, force):
        log.info("probe already finished for %s with this config and code (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta=stages.stage_meta("probe", model_id, split, prompt_mode))
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        meta = io.read_json(edir / "layers.json")
        layers = meta["layers"]
        p_idx, p_acts = load_activations(edir, "probe", "name")
        try:
            c_idx, c_acts = load_activations(edir, "choice", "name")
        except FileNotFoundError:
            log.warning("probe: no choice activations; transfer, familiarity and K-capture diagnostics are skipped")
            c_idx, c_acts = None, None
        log.info("probe: %d probe items, %d choice items, %d layers, d=%d", len(p_idx),
                 0 if c_idx is None else len(c_idx), len(layers), p_acts.shape[-1])
        heldout = set(pr.heldout_categories)
        main = p_idx["control"].isna()
        tiers = {(r["category"], r["entity"]): TIER_LEVEL.get(r["tier"]) for r in load_catalogue(cfg, split)
                 if not r["is_placebo"]}
        ne0 = _measured_ne0(cfg, model_id)
        frames = [f for f in cfg.data.probe_frames if (p_idx["frame"] == f).any()]
        metrics, binding, transfer = [], [], []
        k_meta: dict[str, Any] = {"layers": layers, "frames": frames, "k_rank": pr.k_rank, "files": {}}
        models: dict[tuple[str, int], dict[str, Any]] = {}
        for frame in frames:
            tr = (main & (p_idx["frame"] == frame) & ~p_idx["category"].isin(heldout)).to_numpy()
            if tr.sum() == 0:
                raise RuntimeError(f"no training probe items for frame '{frame}'")
            ents = p_idx.loc[tr, "entity"].to_numpy()
            labels = {a: v[tr] for a, v in _labels(p_idx).items()}
            for li, layer in enumerate(layers):
                X = p_acts[tr, li, :].astype(np.float64)
                fitted = fit_frame_layer(X, ents, labels, list(pr.alphas), pr.cv_folds, cfg.seed)
                models[(frame, li)] = fitted
                full = fitted["full"]
                for a in ATTRIBUTES:
                    st = fitted["attrs"][a]
                    row = {"frame": frame, "layer": layer, "attribute": a, "alpha": st["alpha"],
                           "cv_acc": st["cv_acc"], "control_acc": st["control_acc"],
                           "selectivity": st["cv_acc"] - st["control_acc"], "chance": st["chance"],
                           "n_train": int(tr.sum())}
                    for target_frame in frames:  # held-out categories, same and other frame
                        te = (main & (p_idx["frame"] == target_frame) & p_idx["category"].isin(heldout)).to_numpy()
                        acc = _acc(full.scores(p_acts[te, li, :], st["W"], st["b"]), p_idx.loc[te, f"y_{a}"].to_numpy())
                        key = "heldout_acc" if target_frame == frame else f"heldout_acc_on_{target_frame}"
                        row[key] = acc
                        if target_frame == frame:
                            row["n_heldout"] = int(te.sum())
                    metrics.append(row)
                K = build_k(X, labels, fitted, pr.k_rank)
                kfile = f"K_{frame}_L{layer}.npy"
                np.save(sdir / kfile, K.astype(np.float32))
                k_meta["files"][f"{frame}:{layer}"] = {"file": kfile, "rank": int(K.shape[1])}
                if c_idx is not None:
                    transfer.append(_transfer_row(frame, layer, li, fitted, K, c_idx, c_acts, tiers, ne0, pr, cfg.seed))
                if frame == "background":
                    binding.extend(_binding_rows(layer, li, fitted, p_idx, p_acts, heldout))
                log.info("probe: frame=%s layer=%d done (heldout acc %s)", frame, layer,
                         {a: round(r["heldout_acc"], 2) for r in metrics[-len(ATTRIBUTES):] for a in [r["attribute"]]})
        mdf, bdf, tdf = pd.DataFrame(metrics), pd.DataFrame(binding), pd.DataFrame(transfer)
        io.write_parquet(sdir / "probe_metrics.parquet", mdf)
        io.write_parquet(sdir / "binding.parquet", bdf)
        io.write_parquet(sdir / "transfer.parquet", tdf)
        io.write_json(sdir / "k_meta.json", k_meta)
        warnings = readiness_warnings(mdf, bdf, tdf, pr.k_frame)
        write_probe_summary(sdir, model_id, cfg, mdf, bdf, tdf, warnings, time.time() - t0)
        from rtp.provenance import environment

        io.mark_done(sdir, {"model_id": model_id, "layers": layers, "frames": frames, "warnings": warnings,
                            "finished_with": environment(), "elapsed_s": round(time.time() - t0, 1)})
        log.info("probe finished: %s (%d readiness warnings)", sdir, len(warnings))
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir


def _measured_ne0(cfg: Config, model_id: str) -> dict[tuple[str, str], float]:
    """Evidence-0 name effect per real entity from this model's finished dev measure stage (if any)."""
    mdir = stage_dir(cfg.name, model_id, "measure")
    path = mdir / "entity_evidence.parquet"
    if not (io.is_done(mdir) and path.exists()):
        return {}
    t = pd.read_parquet(path)
    return {(c, e): float(v) for c, e, v in zip(t["category"], t["entity"], t.get("ne_e0", pd.Series(dtype=float)))}


def _transfer_row(frame, layer, li, fitted, K, c_idx, c_acts, tiers, ne0, pr, seed) -> dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    full = fitted["full"]
    X = c_acts[:, li, :].astype(np.float64)
    real = ~c_idx["target_is_placebo"].to_numpy()
    row: dict[str, Any] = {"frame": frame, "layer": layer, "k_rank": int(K.shape[1])}
    st = fitted["attrs"]["price_tier"]
    pred = full.scores(X, st["W"], st["b"]).argmax(axis=1)
    df = pd.DataFrame({"category": c_idx["category"], "entity": c_idx["pair_entity"], "pred": pred, "real": real})
    per = df[df["real"]].groupby(["category", "entity"])["pred"].mean()
    tier = np.array([tiers.get(k, np.nan) for k in per.index], dtype=float)
    ok = np.isfinite(tier)
    if ok.sum() >= 3 and len(set(tier[ok])) > 1 and np.ptp(per.to_numpy()[ok]) > 0:  # constant -> undefined
        rho, p = spearmanr(per.to_numpy()[ok], tier[ok])
        row.update(tier_spearman=float(rho), tier_p=float(p), n_real_entities=int(ok.sum()))
    st = fitted["attrs"]["popularity"]
    pop = full.scores(X, st["W"], st["b"]).argmax(axis=1).astype(float)
    if real.any() and (~real).any():
        row["popularity_auc"] = float(roc_auc_score(real.astype(int), pop))
    if ne0:
        dfp = pd.DataFrame({"category": c_idx["category"], "entity": c_idx["pair_entity"], "pop": pop, "real": real})
        per_pop = dfp[dfp["real"]].groupby(["category", "entity"])["pop"].mean()
        y = np.array([ne0.get(k, np.nan) for k in per_pop.index])
        okk = np.isfinite(y)
        if okk.sum() >= 3 and np.ptp(per_pop.to_numpy()[okk]) > 0 and np.ptp(y[okk]) > 0:
            r2, p2 = spearmanr(per_pop.to_numpy()[okk], y[okk])
            row.update(familiarity_rho_vs_ne_e0=float(r2), familiarity_p=float(p2))
    pairs = c_idx.assign(row_i=np.arange(len(c_idx))).pivot_table(index=["pair_id", "rotation"],
                                                                   columns="target_is_placebo", values="row_i",
                                                                   aggfunc="first").dropna()
    if len(pairs) and K.shape[1]:
        D = X[pairs[False].astype(int).to_numpy()] - X[pairs[True].astype(int).to_numpy()]
        row["k_capture"] = capture_ratio(D, K)
        row["random_capture_mean"], row["random_capture_sd"] = random_capture(D, K.shape[1], pr.n_random_subspaces,
                                                                              seed)
        row["n_pairs"] = int(len(pairs))
    return row


def _binding_rows(layer, li, fitted, p_idx, p_acts, heldout) -> list[dict[str, Any]]:
    rows = []
    full = fitted["full"]
    for control in ("two_brand", "undescribed"):
        te = ((p_idx["control"] == control) & p_idx["category"].isin(heldout)).to_numpy()
        if not te.any():
            continue
        X = p_acts[te, li, :]
        for a in ATTRIBUTES:
            st = fitted["attrs"][a]
            pred = full.scores(X, st["W"], st["b"]).argmax(axis=1)
            y_read = p_idx.loc[te, f"y_{a}"].to_numpy()
            y_other = p_idx.loc[te, f"other_{a}"].to_numpy()
            diff = y_read != y_other
            rows.append({"layer": layer, "control": control, "attribute": a, "n": int(diff.sum()),
                         "acc_vs_read_entity": float((pred[diff] == y_read[diff]).mean()) if diff.any() else np.nan,
                         "acc_vs_other_entity": float((pred[diff] == y_other[diff]).mean()) if diff.any() else np.nan})
    return rows


# --------------------------------------------------------------------------- readiness + summary
def readiness_warnings(mdf: pd.DataFrame, bdf: pd.DataFrame, tdf: pd.DataFrame, k_frame: str) -> list[str]:
    """Plain statements of what would make the M4 decomposition unreliable (thresholds are fixed constants)."""
    w = []
    m = mdf[mdf["frame"] == k_frame]
    for a in ATTRIBUTES:
        sub = m[m["attribute"] == a]
        if len(sub) and sub["selectivity"].max() < SELECTIVITY_WARN:
            w.append(f"{a}: probe selectivity (CV minus control task) never reaches {SELECTIVITY_WARN} at any layer "
                     f"({k_frame} frame): the probe may read surface features, not the attribute")
    if len(bdf):
        tb = bdf[bdf["control"] == "two_brand"].groupby("layer")[["acc_vs_read_entity", "acc_vs_other_entity"]].mean()
        if len(tb) and (tb["acc_vs_read_entity"] - tb["acc_vs_other_entity"]).max() < BINDING_WARN:
            w.append("binding: with two brands described, probes read the target no better than the other brand at "
                     "any layer (entity binding not demonstrated)")
    t = tdf[tdf["frame"] == k_frame]
    if "k_capture" in t and len(t):
        ratio = t["k_capture"] / t["random_capture_mean"]
        if ratio.max() < CAPTURE_RATIO_WARN:
            w.append(f"K captures at most {ratio.max():.1f}x the real-minus-placebo activation variance of a random "
                     f"subspace of the same rank (< {CAPTURE_RATIO_WARN}x): delta_k would be ~0 and the identity "
                     f"residual ~ the whole name effect")
    if "tier_p" in t and len(t) and not ((t["tier_spearman"] > 0) & (t["tier_p"] < 0.05)).any():
        w.append("real-brand transfer: the price-tier probe does not rank real brands by catalogue tier at any layer "
                 "(p < 0.05): K learned on placebos may not describe real-brand knowledge")
    if "popularity_auc" in t and len(t) and t["popularity_auc"].max() < AUC_WARN:
        w.append(f"familiarity: the popularity probe does not separate real brands from placebos at any layer (AUC < "
                 f"{AUC_WARN}); familiarity is then outside K and will inflate identity residuals of famous brands")
    return w


def write_probe_summary(sdir: Path, model_id: str, cfg: Config, mdf, bdf, tdf, warnings, elapsed) -> None:
    from rtp.measure.name_effect import _md_table

    lines = [f"# Probe summary: {model_id} (dev)", "",
             f"- held-out categories: {', '.join(cfg.probe.heldout_categories) or '(none)'}; K frame: "
             f"{cfg.probe.k_frame}; k_rank: {cfg.probe.k_rank}; elapsed {elapsed:.0f}s", "",
             "## Readiness for M4", ""]
    lines += [f"- **WARNING: {x}**" for x in warnings] or ["- no readiness warnings"]
    best = (mdf.sort_values("heldout_acc", ascending=False).groupby(["frame", "attribute"]).head(1)
            .sort_values(["frame", "attribute"]))
    cols = [c for c in ["frame", "attribute", "layer", "cv_acc", "control_acc", "selectivity", "heldout_acc",
                        "heldout_acc_on_background", "heldout_acc_on_choice", "chance", "alpha"] if c in best]
    lines += ["", "## Best layer per frame and attribute (by held-out-category accuracy)", ""] + _md_table(best[cols])
    if len(bdf):
        b = bdf.groupby(["control", "layer"])[["acc_vs_read_entity", "acc_vs_other_entity"]].mean().reset_index()
        lines += ["", "## Binding controls (held-out categories, background-frame probes; mean over attributes)", "",
                  "two_brand: should match the READ entity, not the other described one. undescribed: the read name "
                  "was never described, so acc_vs_read_entity should be at chance.", ""] + _md_table(b)
    if len(tdf):
        tcols = [c for c in ["frame", "layer", "k_rank", "k_capture", "random_capture_mean", "tier_spearman", "tier_p",
                             "popularity_auc", "familiarity_rho_vs_ne_e0", "familiarity_p",
                             "n_pairs"] if c in tdf]
        lines += ["", "## Transfer to real brands and K capture (evidence-0 choice prompts)", "",
                  "k_capture = share of ||h_real - h_placebo||^2 inside K at the target name; compare with "
                  "random_capture_mean (random subspace, same rank).", ""] + _md_table(tdf[tcols])
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
