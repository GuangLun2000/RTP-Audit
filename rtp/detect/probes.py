"""M3 linear probes, control task and the knowledge subspace K (spec §7.5), plus the diagnostics that say
whether K can do its M4 job. Hard constraint 1: reads the activations of ONE model; no base model.

Probes: per frame ("background" = spec text, "choice" = background + choice prompt, read at the option
line), per extracted layer, per attribute, a ridge classifier on standardised activations (one SVD per
training set serves every attribute, every alpha and the control task). Evaluation:

* grouped CV by entity on the training (dev, not held-out) categories, alpha chosen by that CV. The
  spec's control task (attribute vectors permuted across entities) is at chance by construction under
  entity-grouped CV, so "selectivity" here is CV accuracy above that chance-level control;
* held-out dev categories (``probe.heldout_categories``): accuracy at the layer chosen by CV (never
  chosen on the held-out set), within and across frames;
* binding controls in BOTH frames (held-out categories): two-brand texts with counterbalanced order
  (read X, Y also described) and undescribed names (X described, Z read).

K (spec §7.5): per frame and layer, an orthonormal basis (QR, rank <= ``k_rank``) of, in this order,
(1) the encoding patterns of the probes (Haufe et al. 2014: Cov(X) w, the activation direction a
decoder's signal lives in), (2) the principal directions of entity-averaged attribute-level means, and
(3) the probe weight directions. Plain class-mean PCs of single items are dominated by label-free
high-variance directions on real activations, so they no longer come first.

Every M4-readiness diagnostic is compared with a NULL built the same way from shuffled labels, never
with an isotropic random subspace (any data-derived subspace beats that on anisotropic activations):
* K capture: share of ||h_real - h_placebo||^2 (evidence-0 choice prompts) inside K vs inside K built
  from shuffled labels;
* real-brand transfer: within-category Spearman of the price-tier read-out with the catalogue tier,
  with a within-category permutation p-value (pooled correlations pass on category price levels alone);
* familiarity: entity-level AUC of the continuous popularity score for real brands vs placebos, vs the
  same AUC from a shuffled-label probe (lexical real/placebo differences let ANY probe separate them).
Nulls: ``probe.n_null_k`` entity-level label shuffles per frame (the same shuffles at every layer). Per layer
the permutation p is (1 + #null >= observed) / (1 + n); across layers, a max-over-layers null (each shuffle's
largest standardised statistic over the eligible layers) gives a family-wise p, so picking the best layer
after the fact does not inflate significance. Readiness checks ignore the embedding layer (index 0).
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
CAPTURE_EXCESS_WARN = 1.2   # K capture must exceed 1.2x the mean shuffled-label K capture ...
AUC_EXCESS_WARN = 0.05      # popularity AUC (real vs placebo) above the mean shuffled-label probe's AUC ...
FWER_ALPHA = 0.05           # ... and beat the shuffled-label null with a max-over-layers permutation p below this
N_TIER_PERMUTATIONS = 1000


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


def subspace_basis(directions: list[np.ndarray], k_rank: int, tol: float = 1e-6) -> tuple[np.ndarray, int]:
    """(Orthonormal basis d x <=k_rank of the directions, in order; number of independent directions).

    Order-preserving Gram-Schmidt with re-orthogonalisation that SKIPS dependent directions. (Unpivoted
    QR followed by dropping columns with tiny R_jj does not span the inputs: later columns keep components
    along the dropped Q_j, which then vanish from the basis.)
    """
    basis: list[np.ndarray] = []
    n_indep = 0
    for v in directions:
        v = np.asarray(v, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        r = v / norm
        for _ in range(2):  # two passes for numerical orthogonality
            for b in basis:
                r = r - (b @ r) * b
        rn = np.linalg.norm(r)
        if rn > tol ** 0.5:  # relative to the unit-normalised input
            n_indep += 1
            if len(basis) < k_rank:
                basis.append(r / rn)
    if not basis:
        return np.zeros((len(directions[0]) if directions else 0, 0)), 0
    return np.column_stack(basis), n_indep


def entity_mean_pcs(X: np.ndarray, ents: np.ndarray, y: np.ndarray, k: int) -> list[np.ndarray]:
    """Principal directions of the per-level means of ENTITY-AVERAGED activations (at most n_levels - 1).

    Averaging an entity's phrasings first removes template noise, so the level means are less dominated by
    label-free variance than item-level means.
    """
    uniq = sorted(set(ents.tolist()))
    ent_mean = np.stack([X[ents == e].mean(axis=0) for e in uniq])
    ent_y = np.array([y[np.where(ents == e)[0][0]] for e in uniq])
    levels = sorted(set(ent_y.tolist()))
    if len(levels) < 2:
        return []
    means = np.stack([ent_mean[ent_y == lv].mean(axis=0) for lv in levels])
    _, _, vt = np.linalg.svd(means - means.mean(axis=0), full_matrices=False)
    return [vt[i] for i in range(min(k, len(levels) - 1))]


def haufe_patterns(X: np.ndarray, W_raw: np.ndarray) -> np.ndarray:
    """Encoding patterns Cov(X) W of linear decoders W (columns), without forming the d x d covariance."""
    Xc = X - X.mean(axis=0)
    return Xc.T @ (Xc @ W_raw) / max(len(X) - 1, 1)


def build_k(X: np.ndarray, ents: np.ndarray, labels: dict[str, np.ndarray], full: RidgeSVD,
            weights: dict[str, np.ndarray], k_rank: int) -> tuple[np.ndarray, int]:
    """K = QR[Haufe patterns of every attribute's probe, entity-averaged level-mean PCs, probe weights]."""
    raw = {a: full.raw_directions(weights[a]) for a in ATTRIBUTES}
    directions: list[np.ndarray] = []
    for a in ATTRIBUTES:
        directions.extend(haufe_patterns(X, raw[a]).T)
    for a in ATTRIBUTES:
        directions.extend(entity_mean_pcs(X, ents, labels[a], N_LEVELS[a] - 1))
    for a in ATTRIBUTES:
        directions.extend(raw[a].T)
    return subspace_basis(directions, k_rank)


def capture_ratio(D: np.ndarray, K: np.ndarray) -> float:
    """Share of sum ||d||^2 that lies in span(K) (K orthonormal)."""
    tot = float((D ** 2).sum())
    return float(((D @ K) ** 2).sum() / tot) if tot > 0 and K.size else float("nan")


def random_capture(D: np.ndarray, rank: int, n: int, seed: int) -> tuple[float, float]:
    """Capture by isotropic random subspaces (reported for orientation only; NOT the readiness null)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        Q, _ = np.linalg.qr(rng.standard_normal((D.shape[1], rank)))
        vals.append(capture_ratio(D, Q))
    return float(np.mean(vals)), float(np.std(vals))


def within_category_spearman(pred: np.ndarray, truth: np.ndarray, cats: np.ndarray, n_perm: int,
                             seed: int) -> tuple[float, float]:
    """Spearman of category-demeaned ranks, with a p-value from permuting truth WITHIN categories."""
    from scipy.stats import rankdata

    def stat(t: np.ndarray) -> float:
        rp, rt = np.zeros(len(pred)), np.zeros(len(t))
        for c in set(cats.tolist()):
            m = cats == c
            if m.sum() > 1:
                rp[m] = rankdata(pred[m]) - rankdata(pred[m]).mean()
                rt[m] = rankdata(t[m]) - rankdata(t[m]).mean()
        if rp.std() == 0 or rt.std() == 0:
            return float("nan")
        return float(np.corrcoef(rp, rt)[0, 1])

    obs = stat(truth)
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        t = truth.copy()
        for c in set(cats.tolist()):
            m = np.where(cats == c)[0]
            t[m] = t[rng.permutation(m)]
        null.append(stat(t))
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return obs, p


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


def shuffled_fits(X: np.ndarray, ents: np.ndarray, labels: dict[str, np.ndarray], fitted: dict[str, Any],
                  n: int, seed: int) -> list[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Null models: the full ridge fit (same alphas) on ``n`` entity-level label shuffles."""
    full = fitted["full"]
    out = []
    for r in range(n):
        lab = permuted_labels(ents, labels, seed + 1000 + r)
        fits = {a: full.fit(one_hot(lab[a], N_LEVELS[a]), fitted["attrs"][a]["alpha"]) for a in ATTRIBUTES}
        out.append((lab, fits))
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
        null_store: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        k_meta: dict[str, Any] = {"layers": layers, "frames": frames, "k_rank": pr.k_rank, "files": {},
                                  "construction": "QR[Haufe patterns, entity-averaged level-mean PCs, probe weights]"}
        for frame in frames:
            tr = (main & (p_idx["frame"] == frame) & ~p_idx["category"].isin(heldout)).to_numpy()
            if tr.sum() == 0:
                raise RuntimeError(f"no training probe items for frame '{frame}'")
            ents = p_idx.loc[tr, "entity"].to_numpy()
            labels = {a: v[tr] for a, v in _labels(p_idx).items()}
            for li, layer in enumerate(layers):
                X = p_acts[tr, li, :].astype(np.float64)
                fitted = fit_frame_layer(X, ents, labels, list(pr.alphas), pr.cv_folds, cfg.seed)
                full = fitted["full"]
                for a in ATTRIBUTES:
                    st = fitted["attrs"][a]
                    row = {"frame": frame, "layer": layer, "attribute": a, "alpha": st["alpha"],
                           "cv_acc": st["cv_acc"], "control_acc": st["control_acc"],
                           "selectivity": st["cv_acc"] - st["control_acc"], "chance": st["chance"],
                           "n_train": int(tr.sum()), "n_train_entities": int(len(set(ents.tolist())))}
                    for target_frame in frames:  # held-out categories, same and other frame
                        te = (main & (p_idx["frame"] == target_frame) & p_idx["category"].isin(heldout)).to_numpy()
                        acc = _acc(full.scores(p_acts[te, li, :], st["W"], st["b"]), p_idx.loc[te, f"y_{a}"].to_numpy())
                        key = "heldout_acc" if target_frame == frame else f"heldout_acc_on_{target_frame}"
                        row[key] = acc
                        if target_frame == frame:
                            row["n_heldout"] = int(te.sum())
                    metrics.append(row)
                weights = {a: fitted["attrs"][a]["W"] for a in ATTRIBUTES}
                K, n_dirs = build_k(X, ents, labels, full, weights, pr.k_rank)
                kfile = f"K_{frame}_L{layer}.npy"
                np.save(sdir / kfile, K.astype(np.float32))
                k_meta["files"][f"{frame}:{layer}"] = {"file": kfile, "rank": int(K.shape[1]),
                                                        "independent_directions": n_dirs,
                                                        "truncated": n_dirs > pr.k_rank}
                if c_idx is not None:
                    nulls = shuffled_fits(X, ents, labels, fitted, pr.n_null_k, cfg.seed)
                    null_ks = [build_k(X, ents, lab, full, {a: f[a][0] for a in ATTRIBUTES}, pr.k_rank)[0]
                               for lab, f in nulls]
                    row, null_vals = _transfer_row(frame, layer, li, fitted, K, null_ks, [f for _, f in nulls],
                                                   c_idx, c_acts, tiers, ne0, pr, cfg.seed)
                    transfer.append(row)
                    null_store[(frame, layer)] = null_vals
                binding.extend(_binding_rows(frame, layer, li, fitted, p_idx, p_acts, heldout))
                log.info("probe: frame=%s layer=%d done (cv acc %s)", frame, layer,
                         {r["attribute"]: round(r["cv_acc"], 2) for r in metrics[-len(ATTRIBUTES):]})
        mdf, bdf, tdf = pd.DataFrame(metrics), pd.DataFrame(binding), pd.DataFrame(transfer)
        tdf = add_familywise_p(tdf, null_store, layers)
        if null_store:
            np.savez_compressed(sdir / "nulls.npz", **{f"{f}:{lv}:{k}": v for (f, lv), d in null_store.items()
                                                       for k, v in d.items()})
        io.write_parquet(sdir / "probe_metrics.parquet", mdf)
        io.write_parquet(sdir / "binding.parquet", bdf)
        io.write_parquet(sdir / "transfer.parquet", tdf)
        io.write_json(sdir / "k_meta.json", k_meta)
        warnings = readiness_warnings(mdf, bdf, tdf, pr.k_frame, layers)
        if any(v["truncated"] for v in k_meta["files"].values()):
            warnings.append(f"K truncation: more independent directions than k_rank={pr.k_rank} at some layers; the "
                            f"last attributes' read-out directions may be cut (raise probe.k_rank to keep all)")
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
    if "ne_e0" not in t:
        return {}
    return {(c, e): float(v) for c, e, v in zip(t["category"], t["entity"], t["ne_e0"])}


def _popularity_score(full: RidgeSVD, W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Continuous popularity read-out: score of 'widely known' minus 'little-known'."""
    s = full.scores(X, W, b)
    return s[:, -1] - s[:, 0]


def _entity_auc(score: np.ndarray, real: np.ndarray, names: np.ndarray) -> float:
    """AUC of entity-mean scores, real brands vs placebo names (one value per distinct name)."""
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({"s": score, "real": real, "name": names}).groupby(["name", "real"])["s"].mean().reset_index()
    if df["real"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df["real"].astype(int), df["s"]))


def _transfer_row(frame, layer, li, fitted, K, null_ks, null_fits, c_idx, c_acts, tiers, ne0, pr,
                  seed) -> tuple[dict, dict[str, np.ndarray]]:
    """One (frame, layer) row of transfer / familiarity / K-capture diagnostics, plus the null values per shuffle
    (``k_capture``, ``popularity_auc``) for the family-wise test."""
    from scipy.stats import spearmanr

    full = fitted["full"]
    X = c_acts[:, li, :].astype(np.float64)
    real = ~c_idx["target_is_placebo"].to_numpy()
    names = c_idx["target_entity"].astype(str).to_numpy()
    row: dict[str, Any] = {"frame": frame, "layer": layer, "k_rank": int(K.shape[1]),
                           "n_real_entities": int(len(set(names[real].tolist()))),
                           "n_placebo_names": int(len(set(names[~real].tolist())))}
    # real-brand price-tier transfer, within category: continuous read-out = premium score minus budget score
    st = fitted["attrs"]["price_tier"]
    s = full.scores(X, st["W"], st["b"])
    level = s[:, -1] - s[:, 0]
    df = pd.DataFrame({"category": c_idx["category"], "entity": c_idx["pair_entity"], "lv": level, "real": real})
    per = df[df["real"]].groupby(["category", "entity"])["lv"].mean().reset_index()
    tier = np.array([tiers.get((c, e), np.nan) for c, e in zip(per["category"], per["entity"])], dtype=float)
    ok = np.isfinite(tier)
    if ok.sum() >= 4:
        rho, p = within_category_spearman(per["lv"].to_numpy()[ok], tier[ok], per["category"].to_numpy()[ok],
                                          N_TIER_PERMUTATIONS, seed)
        row.update(tier_within_cat_rho=rho, tier_within_cat_p=p)
    # familiarity: continuous popularity read-out, entity-level AUC vs the shuffled-label probe
    pa = fitted["attrs"]["popularity"]
    pop = _popularity_score(full, pa["W"], pa["b"], X)
    row["popularity_auc"] = _entity_auc(pop, real, names)
    null_auc = np.array([_entity_auc(_popularity_score(full, *nf["popularity"], X), real, names) for nf in null_fits])
    row.update(_null_stats("popularity_auc", row["popularity_auc"], null_auc))
    nulls: dict[str, np.ndarray] = {"popularity_auc": null_auc}
    if ne0:
        dfp = pd.DataFrame({"category": c_idx["category"], "entity": c_idx["pair_entity"], "pop": pop, "real": real})
        per_pop = dfp[dfp["real"]].groupby(["category", "entity"])["pop"].mean()
        y = np.array([ne0.get(k, np.nan) for k in per_pop.index])
        okk = np.isfinite(y)
        if okk.sum() >= 3 and np.ptp(per_pop.to_numpy()[okk]) > 0 and np.ptp(y[okk]) > 0:
            r2, p2 = spearmanr(per_pop.to_numpy()[okk], y[okk])
            row.update(familiarity_rho_vs_ne_e0=float(r2), familiarity_p=float(p2))
    # K capture of real-minus-placebo differences (same pair and rotation) vs shuffled-label K
    pairs = c_idx.assign(row_i=np.arange(len(c_idx))).pivot_table(index=["pair_id", "rotation"],
                                                                   columns="target_is_placebo", values="row_i",
                                                                   aggfunc="first").dropna()
    if len(pairs) and K.size:
        D = X[pairs[False].astype(int).to_numpy()] - X[pairs[True].astype(int).to_numpy()]
        row["k_capture"] = capture_ratio(D, K)
        caps = np.array([capture_ratio(D, k) if k.size else np.nan for k in null_ks])
        row.update(_null_stats("k_capture", row["k_capture"], caps))
        nulls["k_capture"] = caps
        row["random_capture_mean"], _ = random_capture(D, K.shape[1], pr.n_random_subspaces, seed)
        row["n_pairs"] = int(len(pairs))
        row["n_distinct_differences"] = int(len(np.unique(np.round(D[:, :8], 3), axis=0)))
    return row, nulls


def _null_stats(name: str, obs: float, null: np.ndarray) -> dict[str, float]:
    """Mean / SD of the shuffled-label null and the one-sided permutation p of the observed value."""
    v = null[np.isfinite(null)]
    if not len(v) or not np.isfinite(obs):
        return {f"{name}_null_mean": float("nan"), f"{name}_null_sd": float("nan"), f"{name}_p": float("nan")}
    return {f"{name}_null_mean": float(v.mean()), f"{name}_null_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            f"{name}_p": float((1 + (v >= obs).sum()) / (1 + len(v)))}


def max_over_layers_p(obs: dict[int, float], null: dict[int, np.ndarray]) -> float:
    """Family-wise permutation p over layers: each statistic is standardised by its own layer's null; the null of
    the maximum is the per-shuffle maximum over layers (shuffle r is the same label permutation at every layer)."""
    keep = [lv for lv in obs if np.isfinite(obs[lv]) and lv in null and np.isfinite(null[lv]).all()
            and null[lv].std(ddof=1) > 0]
    if not keep:
        return float("nan")
    z_obs = max((obs[lv] - null[lv].mean()) / null[lv].std(ddof=1) for lv in keep)
    z_null = np.stack([(null[lv] - null[lv].mean()) / null[lv].std(ddof=1) for lv in keep]).max(axis=0)
    return float((1 + (z_null >= z_obs).sum()) / (1 + len(z_null)))


def add_familywise_p(tdf: pd.DataFrame, null_store: dict[tuple[str, int], dict[str, np.ndarray]],
                     layers: list[int]) -> pd.DataFrame:
    """Per frame, the max-over-layers p (layers > 0) for k_capture and popularity_auc, on every row of the frame."""
    if not len(tdf):
        return tdf
    tdf = tdf.copy()
    eligible = [lv for lv in layers if lv > 0]
    for stat in ("k_capture", "popularity_auc"):
        if stat not in tdf:
            continue
        col = f"{stat}_p_fwer"
        tdf[col] = np.nan
        for frame, sub in tdf.groupby("frame"):
            obs = {int(r["layer"]): float(r[stat]) for _, r in sub.iterrows() if int(r["layer"]) in eligible}
            null = {lv: null_store[(frame, lv)][stat] for lv in obs if stat in null_store.get((frame, lv), {})}
            tdf.loc[tdf["frame"] == frame, col] = max_over_layers_p(obs, null)
    return tdf


def _binding_rows(frame, layer, li, fitted, p_idx, p_acts, heldout) -> list[dict[str, Any]]:
    rows = []
    full = fitted["full"]
    for control in ("two_brand", "undescribed"):
        sel = (p_idx["control"] == control) & (p_idx["frame"] == frame) & p_idx["category"].isin(heldout)
        orders = [None] if control == "undescribed" else [False, True]
        for order in orders:
            te = (sel if order is None else sel & (p_idx["other_first"] == order)).to_numpy()
            if not te.any():
                continue
            X = p_acts[te, li, :]
            for a in ATTRIBUTES:
                st = fitted["attrs"][a]
                pred = full.scores(X, st["W"], st["b"]).argmax(axis=1)
                y_read = p_idx.loc[te, f"y_{a}"].to_numpy()
                y_other = p_idx.loc[te, f"other_{a}"].to_numpy()
                diff = y_read != y_other
                rows.append({"frame": frame, "layer": layer, "control": control,
                             "order": "-" if order is None else ("other first" if order else "read first"),
                             "attribute": a, "n": int(diff.sum()),
                             "acc_vs_read_entity": float((pred[diff] == y_read[diff]).mean()) if diff.any() else np.nan,
                             "acc_vs_other_entity": float((pred[diff] == y_other[diff]).mean()) if diff.any()
                             else np.nan})
    return rows


# --------------------------------------------------------------------------- readiness + summary
def readiness_warnings(mdf: pd.DataFrame, bdf: pd.DataFrame, tdf: pd.DataFrame, k_frame: str,
                       layers: list[int]) -> list[str]:
    """Plain statements of what would make the M4 decomposition unreliable (thresholds are fixed constants).

    Only layers > 0 are eligible (index 0 is the context-free embedding); p-values are Bonferroni-corrected
    over the eligible layers; every comparison is against a shuffled-label null, not against zero.
    """
    eligible = [lv for lv in layers if lv > 0]
    n_layers = max(1, len(eligible))
    w = []
    m = mdf[(mdf["frame"] == k_frame) & mdf["layer"].isin(eligible)] if len(mdf) else mdf
    for a in ATTRIBUTES:
        sub = m[m["attribute"] == a] if len(m) else m
        if len(sub) and sub["selectivity"].max() < SELECTIVITY_WARN:
            w.append(f"{a}: probe CV accuracy never exceeds the shuffled-label control by {SELECTIVITY_WARN} at any "
                     f"layer > 0 ({k_frame} frame)")
    if len(bdf) and "frame" in bdf:
        tb = bdf[(bdf["frame"] == k_frame) & (bdf["control"] == "two_brand") & bdf["layer"].isin(eligible)]
        if len(tb):
            per = tb.groupby(["layer", "order"])[["acc_vs_read_entity", "acc_vs_other_entity"]].mean()
            gap = (per["acc_vs_read_entity"] - per["acc_vs_other_entity"]).unstack("order")
            if not (gap.min(axis=1) >= BINDING_WARN).any():  # must hold in BOTH description orders
                w.append(f"binding ({k_frame} frame): at no layer > 0 do probes read the target better than the other "
                         f"described brand by {BINDING_WARN} in both description orders")
    if len(tdf) and "frame" in tdf:
        t = tdf[(tdf["frame"] == k_frame) & tdf["layer"].isin(eligible)]
        if "k_capture" in t and len(t):
            big = (t["k_capture"] >= CAPTURE_EXCESS_WARN * t["k_capture_null_mean"]).any()
            fw = float(t["k_capture_p_fwer"].iloc[0]) if "k_capture_p_fwer" in t else float("nan")
            if not (big and np.isfinite(fw) and fw < FWER_ALPHA):
                w.append(f"K capture: K does not hold clearly more of the real-minus-placebo activation difference "
                         f"than a K built the same way from shuffled labels (need >= {CAPTURE_EXCESS_WARN}x the null "
                         f"mean at some layer > 0 and max-over-layers permutation p < {FWER_ALPHA}; p = {fw:.3f}), so "
                         f"K cannot be shown to be attribute-specific; a delta_k from this K would not be "
                         f"interpretable as knowledge")
        if "tier_within_cat_p" in t and len(t):
            sig = (t["tier_within_cat_rho"] > 0) & (t["tier_within_cat_p"] * n_layers < 0.05)
            if not sig.any():
                w.append("real-brand transfer: the price-tier read-out does not rank real brands by catalogue tier "
                         "WITHIN categories at any layer > 0 (permutation p, Bonferroni over layers)")
        if "popularity_auc" in t and len(t):
            big = ((t["popularity_auc"] - t["popularity_auc_null_mean"]) >= AUC_EXCESS_WARN).any()
            fw = float(t["popularity_auc_p_fwer"].iloc[0]) if "popularity_auc_p_fwer" in t else float("nan")
            if not (big and np.isfinite(fw) and fw < FWER_ALPHA):
                w.append(f"familiarity: the popularity read-out separates real brands from placebos no better than "
                         f"shuffled-label probes (need +{AUC_EXCESS_WARN} AUC over the null mean at some layer > 0 and "
                         f"max-over-layers p < {FWER_ALPHA}; p = {fw:.3f}); familiarity is then outside K and would "
                         f"inflate the identity residuals of famous brands")
    return w


def write_probe_summary(sdir: Path, model_id: str, cfg: Config, mdf, bdf, tdf, warnings, elapsed) -> None:
    from rtp.measure.name_effect import _md_table

    lines = [f"# Probe summary: {model_id} (dev)", "",
             f"- held-out categories: {', '.join(cfg.probe.heldout_categories) or '(none)'}; K frame: "
             f"{cfg.probe.k_frame}; k_rank: {cfg.probe.k_rank}; elapsed {elapsed:.0f}s",
             "- every readiness number below is compared with a shuffled-label null; layer 0 (embeddings) is never "
             "used for readiness; held-out accuracies are reported at the layer chosen by CV.", "",
             "## Readiness for M4", ""]
    lines += [f"- **WARNING: {x}**" for x in warnings] or ["- no readiness warnings"]
    if len(mdf):
        best = (mdf[mdf["layer"] > 0].sort_values("cv_acc", ascending=False).groupby(["frame", "attribute"]).head(1)
                .sort_values(["frame", "attribute"]))
        cols = [c for c in ["frame", "attribute", "layer", "cv_acc", "control_acc", "selectivity", "heldout_acc",
                            "heldout_acc_on_background", "heldout_acc_on_choice", "chance", "n_train_entities",
                            "alpha"] if c in best]
        lines += ["", "## Best layer per frame and attribute (chosen by grouped CV; held-out shown at that layer)", ""]
        lines += _md_table(best[cols])
    if len(bdf):
        b = bdf.groupby(["frame", "control", "order", "layer"])[["acc_vs_read_entity", "acc_vs_other_entity"]].mean()
        lines += ["", "## Binding controls (held-out categories; mean over attributes)", "",
                  "two_brand: should match the READ entity in both description orders. undescribed: the read name was "
                  "never described, so both accuracies should be at chance (acc_vs_other > chance = context leakage).",
                  ""] + _md_table(b.reset_index())
    if len(tdf):
        tcols = [c for c in ["frame", "layer", "k_rank", "k_capture", "k_capture_null_mean", "k_capture_null_sd",
                             "k_capture_p", "k_capture_p_fwer", "random_capture_mean", "tier_within_cat_rho",
                             "tier_within_cat_p", "popularity_auc", "popularity_auc_null_mean", "popularity_auc_p",
                             "popularity_auc_p_fwer", "familiarity_rho_vs_ne_e0", "n_real_entities",
                             "n_placebo_names", "n_pairs", "n_distinct_differences"] if c in tdf]
        lines += ["", "## Transfer to real brands, familiarity and K capture (evidence-0 choice prompts)", "",
                  f"k_capture = share of ||h_real - h_placebo||^2 inside K; the null is {cfg.probe.n_null_k} Ks built "
                  "the same way from shuffled labels (_p: per-layer permutation p; _p_fwer: max-over-layers p for the "
                  "frame, layers > 0). random_capture (isotropic) is shown for orientation only. "
                  "n_distinct_differences shows how many of the n_pairs differences are distinct vectors.", ""]
        lines += _md_table(tdf[tcols])
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
