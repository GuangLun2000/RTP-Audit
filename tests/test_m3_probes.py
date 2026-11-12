"""M3: probe data frames/controls, ridge probes, control task, K recovery (spec §10), extraction plumbing."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from rtp.config import config_from_dict
from rtp.data.build import generate
from rtp.data.probe_items import ATTRIBUTES, N_LEVELS
from rtp.detect.probes import (RidgeSVD, build_k, capture_ratio, fit_frame_layer, grouped_folds,
                               permuted_labels, random_capture, readiness_warnings)


# --------------------------------------------------------------------------- synthetic activations
def synthetic(n_entities=60, per_entity=6, d=48, noise=0.3, seed=0):
    """Each attribute level is a fixed random direction; activation = sum of the entity's level directions."""
    rng = np.random.default_rng(seed)
    dirs = {a: rng.standard_normal((N_LEVELS[a], d)) for a in ATTRIBUTES}
    ents, labels, X = [], {a: [] for a in ATTRIBUTES}, []
    for e in range(n_entities):
        lab = {a: int(rng.integers(N_LEVELS[a])) for a in ATTRIBUTES}
        for _ in range(per_entity):
            X.append(sum(dirs[a][lab[a]] for a in ATTRIBUTES) + noise * rng.standard_normal(d))
            ents.append(f"e{e}")
            for a in ATTRIBUTES:
                labels[a].append(lab[a])
    return np.array(X), np.array(ents), {a: np.array(v) for a, v in labels.items()}, dirs


def test_ridge_svd_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    X, Y = rng.standard_normal((80, 20)), rng.standard_normal((80, 3))
    m = RidgeSVD(X)
    W, b = m.fit(Y, 5.0)
    sk = Ridge(alpha=5.0).fit(StandardScaler().fit_transform(X), Y)
    assert np.allclose(m.scores(X, W, b), sk.predict(StandardScaler().fit_transform(X)), atol=1e-8)


def test_grouped_folds_never_split_an_entity():
    ents = np.array([f"e{i // 3}" for i in range(30)])
    for tr, va in grouped_folds(ents, 4, seed=0):
        assert not set(ents[tr]) & set(ents[va])
    covered = np.concatenate([va for _, va in grouped_folds(ents, 4, 0)])
    assert sorted(covered) == list(range(30))


def test_control_labels_are_entity_consistent_permutations():
    X, ents, labels, _ = synthetic(n_entities=20, per_entity=3)
    ctl = permuted_labels(ents, labels, seed=0)
    for e in set(ents):
        rows = ents == e
        for a in ATTRIBUTES:
            assert len(set(ctl[a][rows])) == 1  # still one label vector per entity
    assert any((ctl[a] != labels[a]).any() for a in ATTRIBUTES)
    for a in ATTRIBUTES:  # same label multiset
        assert sorted(ctl[a]) == sorted(labels[a])


def test_probes_beat_control_task_on_encoded_attributes():
    X, ents, labels, _ = synthetic()
    fitted = fit_frame_layer(X, ents, labels, [0.1, 1.0, 10.0], n_folds=5, seed=0)
    for a in ATTRIBUTES:
        st = fitted["attrs"][a]
        assert st["cv_acc"] > 0.9, a
        assert st["cv_acc"] - st["control_acc"] > 0.3, a


def test_k_recovers_embedded_attribute_subspace():
    """Spec §10: embed known directions in synthetic activations and check that K recovers them."""
    X, ents, labels, dirs = synthetic(noise=0.2)
    fitted = fit_frame_layer(X, ents, labels, [0.1, 1.0, 10.0], n_folds=5, seed=0)
    K, n_dirs = build_k(X, ents, labels, fitted["full"], {a: fitted["attrs"][a]["W"] for a in ATTRIBUTES}, k_rank=18)
    assert n_dirs >= 18
    assert np.allclose(K.T @ K, np.eye(K.shape[1]), atol=1e-8)
    # the contrasts that carry attribute information (level differences) must lie inside K
    for a in ATTRIBUTES:
        for lv in range(1, N_LEVELS[a]):
            v = dirs[a][lv] - dirs[a][0]
            assert np.linalg.norm(K.T @ v) / np.linalg.norm(v) > 0.95, (a, lv)
    # and a random direction is mostly outside K
    r = np.random.default_rng(9).standard_normal(X.shape[1])
    assert np.linalg.norm(K.T @ r) / np.linalg.norm(r) < 0.8


def test_k_beats_shuffled_label_k_on_anisotropic_activations():
    """Real activations are anisotropic: per-item nuisance variance and large per-entity identity variance in
    directions unrelated to the attributes. K must still recover the attribute contrasts, and must capture an
    attribute-driven difference clearly better than a K built the same way from shuffled labels (the null)."""
    from rtp.detect.probes import shuffled_fits

    rng = np.random.default_rng(3)
    X, ents, labels, dirs = synthetic(n_entities=80, per_entity=6, d=64, noise=0.3, seed=3)
    q_id, _ = np.linalg.qr(rng.standard_normal((64, 10)))
    ent_id = {e: q_id @ rng.standard_normal(10) for e in set(ents.tolist())}
    X = 0.5 * X + np.stack([6.0 * ent_id[e] for e in ents]) + 3.0 * rng.standard_normal((len(X), 1)) * \
        rng.standard_normal(64)
    fitted = fit_frame_layer(X, ents, labels, [1.0, 10.0, 100.0], n_folds=5, seed=0)
    full = fitted["full"]
    K, _ = build_k(X, ents, labels, full, {a: fitted["attrs"][a]["W"] for a in ATTRIBUTES}, k_rank=18)
    v = dirs["quality"][4] - dirs["quality"][0]
    assert np.linalg.norm(K.T @ v) / np.linalg.norm(v) > 0.9
    D = np.stack([v + 0.2 * rng.standard_normal(64) for _ in range(40)])  # "real minus placebo" = knowledge
    nulls = [build_k(X, ents, lab, full, {a: f[a][0] for a in ATTRIBUTES}, 18)[0]
             for lab, f in shuffled_fits(X, ents, labels, fitted, 3, seed=0)]
    assert capture_ratio(D, K) > 1.5 * np.mean([capture_ratio(D, k) for k in nulls])


def test_subspace_basis_spans_inputs_despite_dependent_directions():
    from rtp.detect.probes import subspace_basis

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(10), rng.standard_normal(10)
    c = rng.standard_normal(10)
    K, n = subspace_basis([a, b, a + b, c], k_rank=5)  # a + b is dependent and must not cost c its span
    assert n == 3 and K.shape == (10, 3)
    for v in (a, b, c):
        assert np.linalg.norm(K.T @ v) / np.linalg.norm(v) > 1 - 1e-9


def test_within_category_spearman_ignores_category_level_differences():
    from rtp.detect.probes import within_category_spearman

    cats = np.array(["car"] * 6 + ["snack"] * 6)
    truth = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float)  # cars pricier than snacks
    category_only = np.array([5.0] * 6 + [0.0] * 6) + np.random.default_rng(0).normal(0, 0.01, 12)
    rho, p = within_category_spearman(category_only, truth, cats, 300, seed=0)
    assert p > 0.05  # pooled Spearman would be ~0.8 here
    rho2, p2 = within_category_spearman(truth + np.random.default_rng(1).normal(0, 0.01, 12), truth, cats, 300, 0)
    assert rho2 > 0.8 and p2 < 0.05  # ties in the tiers cap the rank correlation below 1


def test_capture_ratio():
    rng = np.random.default_rng(0)
    K, _ = np.linalg.qr(rng.standard_normal((30, 4)))
    D_in = rng.standard_normal((50, 4)) @ K.T
    assert capture_ratio(D_in, K) == pytest.approx(1.0)
    mean, sd = random_capture(rng.standard_normal((50, 30)), 4, 10, seed=0)
    assert 0.0 < mean < 0.4 and sd >= 0


# --------------------------------------------------------------------------- probe data
@pytest.fixture(scope="module")
def probes():
    cfg = config_from_dict({"name": "m3", "data": {"categories": ["running shoe"], "n_real_per_category": 3,
                                                   "n_placebo_per_category": 3, "cards_per_cell": 1,
                                                   "probe_entities_per_category": 4, "probe_templates": 2}})
    return generate(cfg)["probe"]


def test_probe_frames_and_controls(probes):
    from collections import Counter

    kinds = Counter((p["frame"], p["control"]) for p in probes)
    assert kinds == {("background", None): 8, ("choice", None): 8, ("background", "two_brand"): 4,
                     ("background", "undescribed"): 4, ("choice", "two_brand"): 4, ("choice", "undescribed"): 4}
    orders = Counter(p["other_first"] for p in probes if p["control"] == "two_brand")
    assert orders[True] == orders[False]  # counterbalanced description order
    for p in probes:
        s, t = p["name_char_span"]
        assert p["text"][s:t] == p["entity"] and p["text"].rfind(p["entity"]) == s
        if p["frame"] == "choice":
            assert "Here are four options:" in p["text"] and f". {p['entity']}\n" in p["text"]
            if p["control"]:
                assert f". {p['other_entity']}\n" in p["text"]  # both entities are options
        if p["control"] == "two_brand":
            assert p["other_entity"] in p["text"] and p["other_attributes"] is not None
        if p["control"] == "undescribed":
            assert p["text"].count(p["entity"]) == 1 and p["text"].startswith(f"Background: {p['other_entity']}")
        if p["control"] == "two_brand" and p["frame"] == "background":
            first = p["other_entity"] if p["other_first"] else p["entity"]
            assert p["text"].startswith(f"Background: {first} is")


def test_detect_functions_take_one_model_path():
    from rtp.detect.activations import run_extract
    from rtp.detect.probes import run_probe

    for fn in (run_extract, run_probe):
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["cfg", "model_id"]
        assert not [p for p in params if "base" in p or "ref" in p]


def test_selected_layers_and_target_span():
    from rtp.detect.activations import selected_layers, target_name_span

    assert selected_layers(24, 2) == list(range(0, 24, 2))  # index 24 = post-final-norm: excluded
    assert selected_layers(5, 2) == [0, 2, 4]
    item = {"item_id": "x", "target_label": "C", "target_entity": "Nike",
            "prompt": "I'm shopping.\n\nA. Asics\nB. Nikon\nC. Nike\nD. Hoka\n\nWhich?"}
    s, t = target_name_span(item)
    assert item["prompt"][s:t] == "Nike" and item["prompt"][s - 3:s] == "C. "


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_extract_and_probe_on_smoke_model(smoke_lm, work_dir, monkeypatch):
    from rtp import io
    from rtp.data.build import build_data, load_probe_items
    from rtp.detect.activations import load_activations, locate, run_extract
    from rtp.detect.probes import run_probe

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    cfg = config_from_dict({"name": "m3it", "data": {
        "categories": ["running shoe", "laptop"], "n_real_per_category": 3, "n_placebo_per_category": 3,
        "cards_per_cell": 1, "evidence_levels": [0], "probe_entities_per_category": 3, "probe_templates": 2},
        "extract": {"layer_stride": 6, "shard_size": 10}, "probe": {"heldout_categories": ["laptop"], "k_rank": 4}})
    build_data(cfg)
    edir = run_extract(cfg, smoke_lm.model_id)
    meta = io.read_json(edir / "layers.json")
    assert meta["layers"] == [0, 6, 12, 18]
    idx, acts = load_activations(edir, "probe")
    assert acts.dtype == np.float16 and acts.shape == (len(idx), 4, smoke_lm.model.config.hidden_size)
    assert len(list((edir / "probe").glob("*.done"))) == -(-len(idx) // 10)
    # the located token is the end of the read name
    p = load_probe_items(cfg, "dev")[0]
    ids, last, n = locate(smoke_lm, p["text"], p["name_char_span"], None)
    assert smoke_lm.tokenizer.decode(ids[last - n + 1:last + 1]).strip() == p["entity"]
    # activations equal a fresh single forward pass at that position
    import torch

    with torch.inference_mode():
        hs = smoke_lm.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    assert np.allclose(acts[0, 2], hs[12][0, last].numpy().astype(np.float16), atol=1e-2)
    pdir = run_probe(cfg, smoke_lm.model_id)
    k = np.load(pdir / "K_choice_L12.npy")
    assert k.shape[0] == smoke_lm.model.config.hidden_size and k.shape[1] <= 4
    t = pd.read_parquet(pdir / "transfer.parquet")
    assert {"k_capture_null_mean", "k_capture_p", "k_capture_p_fwer", "popularity_auc_p_fwer"} <= set(t.columns)
    assert (pdir / "nulls.npz").exists()
    assert np.allclose(k.T @ k, np.eye(k.shape[1]), atol=1e-4)
    assert io.is_done(pdir) and (pdir / "summary.md").read_text().count("Readiness for M4") == 1


def test_readiness_ignores_embedding_layer_and_uses_shuffled_null():
    t = pd.DataFrame([
        {"frame": "choice", "layer": 0, "k_capture": 0.9, "k_capture_null_mean": 0.1, "k_capture_p_fwer": 0.01,
         "popularity_auc": 0.99, "popularity_auc_null_mean": 0.5, "popularity_auc_p_fwer": 0.01},
        {"frame": "choice", "layer": 4, "k_capture": 0.3, "k_capture_null_mean": 0.29, "k_capture_p_fwer": 0.01,
         "popularity_auc": 0.80, "popularity_auc_null_mean": 0.79, "popularity_auc_p_fwer": 0.01}])
    w = readiness_warnings(pd.DataFrame(), pd.DataFrame(), t, "choice", [0, 4])
    assert any(x.startswith("K capture") for x in w) and any(x.startswith("familiarity") for x in w)
    big = t.assign(k_capture=[0.9, 0.9], popularity_auc=[0.99, 0.99])
    assert not readiness_warnings(pd.DataFrame(), pd.DataFrame(), big, "choice", [0, 4])
    weak = big.assign(k_capture_p_fwer=0.2, popularity_auc_p_fwer=0.2)  # large but not beyond the null
    assert len(readiness_warnings(pd.DataFrame(), pd.DataFrame(), weak, "choice", [0, 4])) == 2
    assert readiness_warnings(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "choice", [0, 4]) == []  # no crash


def test_max_over_layers_p():
    from rtp.detect.probes import max_over_layers_p

    rng = np.random.default_rng(0)
    null = {lv: rng.normal(0.3, 0.01, 99) for lv in (2, 4, 6)}
    strong = {2: 0.30, 4: 0.36, 6: 0.30}  # 6 SD above its null at one layer
    assert max_over_layers_p(strong, null) == pytest.approx(0.01)
    typical = {lv: float(np.median(null[lv])) for lv in null}
    assert max_over_layers_p(typical, null) > 0.5
    # picking the best of three layers is penalised: the same z at one layer is less significant family-wise
    one = {4: float(np.quantile(null[4], 0.97))}
    three = {2: float(np.median(null[2])), 4: one[4], 6: float(np.median(null[6]))}
    assert max_over_layers_p(three, null) >= max_over_layers_p(one, {4: null[4]})
