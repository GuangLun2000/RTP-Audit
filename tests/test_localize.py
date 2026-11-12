"""Localisation stage (rtp.detect.localize) and activation patching (rtp.patching): synthetic estimators on CPU,
plus the spec §10 identity transplant and an end-to-end run on the smoke model."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from rtp.config import config_from_dict
from rtp.data.build import generate
from rtp.detect.localize import choose_layers, configurations, recovery_table, select_units, token_class


def _cfg(**loc):
    return config_from_dict({"name": "loc", "seed": 2, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0, 4], "handicaps": [0], "e0_pairs_per_substitute": 2, "probe_entities_per_category": 1,
        "probe_templates": 1}, "localize": loc})


def test_select_units_balanced_over_substitutes():
    cfg = _cfg(pairs_per_entity=3, rotations=[0, 2])
    items = [it for it in generate(cfg)["choice"] if it["split"] == "dev"]
    units = select_units(items, cfg)
    assert all(r["evidence_level"] == 0 and r["pair_id"] == p["pair_id"] and r["rotation"] == p["rotation"]
               and not r["target_is_placebo"] and p["target_is_placebo"] for r, p in units)
    per_entity = Counter(r["pair_entity"] for r, _ in units)
    assert set(per_entity.values()) == {3 * 2}  # 3 pairs x 2 rotations
    for ent in per_entity:  # one pair per substitute (3 placebos -> 3 distinct substitutes)
        subs = {r["placebo_substitute"] for r, _ in units if r["pair_entity"] == ent}
        assert len(subs) == 3
    assert select_units(items, cfg) == units  # deterministic


def test_configurations_never_patch_embeddings():
    cfgs = configurations(n_blocks=24, stride=6, band=3)
    assert {L for _, L, _ in cfgs} == {6, 12, 18}
    assert all(min(lays) >= 1 and max(lays) <= 23 for _, _, lays in cfgs)
    assert sum(site == "band" for site, _, _ in cfgs) == 3 and Counter(s for s, _, _ in cfgs)["name_last"] == 3


def _synthetic(rec_at: dict[int, float], n_ent: int = 6, cls_of=lambda e: "1") -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for e in range(n_ent):
        for u in range(2):
            uid = f"e{e}-u{u}"
            ne = (e - (n_ent - 1) / 2) + 0.1 * u  # name effects of both signs
            base = {"unit_id": uid, "entity": f"E{e}", "token_class": cls_of(e), "layers": ""}
            s_p = rng.normal()
            rows += [{**base, "direction": "clean", "receiver": "real", "site": "none", "layer": -1, "s_target": s_p + ne},
                     {**base, "direction": "clean", "receiver": "placebo", "site": "none", "layer": -1,
                      "s_target": s_p}]
            for L, frac in rec_at.items():
                rows.append({**base, "direction": "in", "receiver": "placebo", "site": "name_last", "layer": L,
                             "s_target": s_p + frac * ne})
                rows.append({**base, "direction": "out", "receiver": "real", "site": "name_last", "layer": L,
                             "s_target": s_p + ne - frac * ne})
    return pd.DataFrame(rows)


def test_recovery_table_pooled_and_between_entity_r2():
    rec = recovery_table(_synthetic({4: 0.5, 8: 1.0}), n_boot=200, seed=0)
    r = rec.set_index(["direction", "layer"])
    assert r.loc[("in", 4), "pooled_recovery"] == pytest.approx(0.5)
    assert r.loc[("out", 8), "pooled_recovery"] == pytest.approx(1.0)
    assert r.loc[("in", 8), "r2_between_entities"] == pytest.approx(1.0)
    assert r.loc[("in", 4), "r2_between_entities"] < 0.9
    assert r.loc[("in", 4), "ci_low"] <= 0.5 <= r.loc[("in", 4), "ci_high"]


def test_choose_layers_spec_argmax_and_alternative():
    rec = recovery_table(_synthetic({2: 0.6, 4: 0.97, 6: 0.92, 8: 0.5}), n_boot=50, seed=0)
    c = choose_layers(rec)
    assert c["spec_layer"] == 4 and not c["spec_warning"] and c["alt_layer"] == 6  # deepest >= 0.9
    weak = choose_layers(recovery_table(_synthetic({2: 0.3, 4: 0.6}), n_boot=50, seed=0))
    assert weak["spec_layer"] == 4 and weak["spec_warning"] and weak["alt_layer"] is None
    # token classes: multi-token names recover only half at layer 6 -> the alternative falls back to layer 4
    mixed = _synthetic({4: 0.95, 6: 0.95}, n_ent=8, cls_of=lambda e: "3+" if e in (2, 3, 4) else "1")
    sel = (mixed["layer"] == 6) & (mixed["token_class"] == "3+") & (mixed["direction"] == "in")
    clean_p = mixed[(mixed["direction"] == "clean") & (mixed["receiver"] == "placebo")].set_index("unit_id")["s_target"]
    clean_r = mixed[(mixed["direction"] == "clean") & (mixed["receiver"] == "real")].set_index("unit_id")["s_target"]
    ne = (clean_r - clean_p).reindex(mixed.loc[sel, "unit_id"]).to_numpy()
    mixed.loc[sel, "s_target"] = clean_p.reindex(mixed.loc[sel, "unit_id"]).to_numpy() + 0.5 * ne
    assert token_class(1) == "1" and token_class(2) == "2" and token_class(5) == "3+"
    rec = recovery_table(mixed, n_boot=20, seed=0)
    r6 = rec[(rec["direction"] == "in") & (rec["layer"] == 6)].iloc[0]
    assert r6["pooled_recovery"] >= 0.9 and r6["pooled_tok3+"] == pytest.approx(0.5)  # pooled passes, class fails
    c = choose_layers(rec)
    assert c["alt_layer"] == 4 and c["spec_layer"] == 4


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_identity_transplant_reproduces_source_logits(smoke_lm):
    """Spec §10: transplanting EVERY position of the source run at layer 0 into the base run gives the
    source logits (and the same at a middle layer)."""
    import torch

    from rtp.models import encode, format_chat
    from rtp.patching import patched

    tok = smoke_lm.tokenizer
    src = encode(tok, format_chat(tok, "Which running shoe brand is best: Nike or Asics?"), True)
    base = list(src)
    for k in range(5, 10):  # a base run that differs from the source in several tokens
        base[k] = (base[k] + 17) % len(tok)
    with torch.inference_mode():
        out = smoke_lm.model(input_ids=torch.tensor([src]), output_hidden_states=True)
        ref = out.logits[0].float()
        for layer in (0, 12):
            h = out.hidden_states[layer][0]
            patches = {layer: [(0, p, h[p]) for p in range(len(src))]}
            with patched(smoke_lm.model, patches):
                got = smoke_lm.model(input_ids=torch.tensor([base])).logits[0].float()
            assert torch.allclose(got, ref, atol=1e-4), layer
        clean = smoke_lm.model(input_ids=torch.tensor([base])).logits[0].float()
    assert not torch.allclose(clean, ref, atol=1e-2)  # the hooks are removed after the context


@pytest.mark.integration
def test_self_transplant_is_a_no_op_and_stage_runs(smoke_lm, work_dir, monkeypatch):
    from rtp import io
    from rtp.data.build import build_data, load_choice_items
    from rtp.detect.localize import VERSIONS, prepare_unit, run_localize, score_rows
    from rtp.measure.choice import letter_token_ids

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    cfg = _cfg(pairs_per_entity=1, rotations=[0], layer_stride=12, band=2, units_per_batch=2, n_boot=50,
               max_batch_size=8, max_batch_tokens=8192)
    build_data(cfg)
    units = [prepare_unit(smoke_lm, r, p) for r, p in select_units(load_choice_items(cfg, "dev"), cfg)[:2]]
    assert all(u is not None and u["n_post"] > 0 for u in units)
    letters = letter_token_ids(smoke_lm.tokenizer)
    rows = [{"unit": u, "receiver": v, "donor": None, "site": "none", "layers": ()} for u in range(2) for v in VERSIONS]
    rows += [{"unit": u, "receiver": v, "donor": v, "site": site, "layers": (12,)}
             for u in range(2) for v in VERSIONS for site in ("name_last", "post_name", "final")]
    sc = score_rows(smoke_lm, units, rows, letters, [12], cfg)
    clean = {(r["unit"], r["receiver"]): s["s_target"] for r, s in zip(rows, sc) if r["donor"] is None}
    for r, s in zip(rows, sc):
        assert s["s_target"] == pytest.approx(clean[(r["unit"], r["receiver"])], abs=1e-3)
    # a real transplant changes the placebo run's score at least somewhere
    swap = [{"unit": 0, "receiver": "placebo", "donor": "real", "site": "post_name", "layers": (12,)}]
    assert score_rows(smoke_lm, units, swap, letters, [12], cfg)[0]["s_target"] != pytest.approx(clean[(0, "placebo")],
                                                                                                abs=1e-6)
    ldir = run_localize(cfg, smoke_lm.model_id)
    rec = pd.read_parquet(ldir / "recovery.parquet")
    assert io.is_done(ldir) and {"in", "out"} == set(rec["direction"])
    assert set(rec["site"]) == {"name_last", "post_name", "final", "band"} and (ldir / "summary.md").exists()
