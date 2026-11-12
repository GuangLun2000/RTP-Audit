"""M3: probe data frames/controls, ridge probes, control task, K recovery (spec §10), extraction plumbing."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rtp.config import config_from_dict
from rtp.data.build import generate
from rtp.data.probe_items import ATTRIBUTES, N_LEVELS
from rtp.detect.probes import (RidgeSVD, build_k, capture_ratio, fit_frame_layer, grouped_folds,
                               permuted_labels, random_capture)


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
    K = build_k(X, labels, fitted, k_rank=16)
    assert np.allclose(K.T @ K, np.eye(K.shape[1]), atol=1e-8)
    # the contrasts that carry attribute information (level differences) must lie inside K
    for a in ATTRIBUTES:
        for lv in range(1, N_LEVELS[a]):
            v = dirs[a][lv] - dirs[a][0]
            assert np.linalg.norm(K.T @ v) / np.linalg.norm(v) > 0.95, (a, lv)
    # and a random direction is mostly outside K
    r = np.random.default_rng(9).standard_normal(X.shape[1])
    assert np.linalg.norm(K.T @ r) / np.linalg.norm(r) < 0.8


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
                     ("background", "undescribed"): 4}
    for p in probes:
        s, t = p["name_char_span"]
        assert p["text"][s:t] == p["entity"] and p["text"].rfind(p["entity"]) == s
        if p["frame"] == "choice":
            assert "Here are four options:" in p["text"] and f". {p['entity']}\n" in p["text"]
        if p["control"] == "two_brand":
            assert p["other_entity"] in p["text"] and p["other_attributes"] is not None
        if p["control"] == "undescribed":
            assert p["text"].count(p["entity"]) == 1 and p["text"].startswith(f"Background: {p['other_entity']}")


def test_detect_functions_take_one_model_path():
    from rtp.detect.activations import run_extract
    from rtp.detect.probes import run_probe

    for fn in (run_extract, run_probe):
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["cfg", "model_id"]
        assert not [p for p in params if "base" in p or "ref" in p]


def test_selected_layers_and_target_span():
    from rtp.detect.activations import selected_layers, target_name_span

    assert selected_layers(24, 2) == list(range(0, 25, 2))
    assert selected_layers(5, 2) == [0, 2, 4, 5]
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
    assert meta["layers"] == [0, 6, 12, 18, 24]
    idx, acts = load_activations(edir, "probe")
    assert acts.dtype == np.float16 and acts.shape == (len(idx), 5, smoke_lm.model.config.hidden_size)
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
    assert np.allclose(k.T @ k, np.eye(k.shape[1]), atol=1e-4)
    assert io.is_done(pdir) and (pdir / "summary.md").read_text().count("Readiness for M4") == 1
