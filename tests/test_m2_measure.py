"""M2 estimators on synthetic data (CPU only), plus one end-to-end integration test on the smoke model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from rtp.config import config_from_dict
from rtp.data.build import generate
from rtp.data.choice_items import LETTERS
from rtp.measure.choice import (letter_logprobs, letter_token_ids, normalize_letters, parse_letter, s_target,
                                select_gen_check)
from rtp.measure.name_effect import (consistency, entity_evidence_table, items_meta, ladder_regression,
                                     paired_name_effect, robust_origin_regression, rotation_average)
from rtp.measure.open_ended import (BrandMatcher, aggregate_picks, brand_variants, compute_shares,
                                    dedup_token_prefix, pick_lists, pick_prob_from_scores)


# --------------------------------------------------------------------------- letter probabilities
class FakeTok:
    def __init__(self, pieces, special=()):
        self.pieces = pieces
        self.all_special_ids = list(special)

    def __len__(self):
        return len(self.pieces)

    def batch_decode(self, seqs):
        return ["".join(self.pieces[i] for i in s) for s in seqs]


def test_letter_token_ids_merges_whitespace_variants():
    tok = FakeTok(["A", " A", "\tA", "B", " B", "C", "D", " D", "AB", "<s>", "a"], special=[9])
    ids = letter_token_ids(tok)
    assert ids == {"A": [0, 1, 2], "B": [3, 4], "C": [5], "D": [6, 7]}
    with pytest.raises(ValueError):
        letter_token_ids(FakeTok(["A", "B", "C"]))


def test_letter_logsumexp_and_renormalisation():
    probs = torch.tensor([[0.1, 0.1, 0.2, 0.05, 0.05, 0.5]])  # ids 0,1 -> A ; 2 -> B ; 3 -> C ; 4 -> D ; 5 other
    ids = {"A": [0, 1], "B": [2], "C": [3], "D": [4]}
    lp = letter_logprobs(torch.log(probs), ids)
    assert torch.allclose(lp.exp(), torch.tensor([[0.2, 0.2, 0.05, 0.05]]), atol=1e-6)
    p = normalize_letters(lp.numpy())
    assert np.allclose(p, [[0.4, 0.4, 0.1, 0.1]]) and np.isclose(p.sum(), 1.0)


def test_s_target_log_odds_and_clipping():
    assert s_target(0.5) == pytest.approx(0.0)
    assert s_target(0.8) == pytest.approx(math.log(4.0))
    hi = math.log((1 - 1e-4) / 1e-4)
    assert s_target(1.0) == pytest.approx(hi) and s_target(0.0) == pytest.approx(-hi)


def test_parse_letter():
    for text, want in [("B", "B"), (" C.", "C"), ("Answer: D", "D"), ("The answer is B", "B"), ("(A)", "A"),
                       ("**C**", "C"), ("Option D", "D"), ("I'd pick C", "C"), ("I choose B.", "B"),
                       ("A good choice", None), ("A great pick would", None), ("Nike", None), ("", None),
                       (None, None), ("Based on the information", None), ("Answer: B is the best", "B"),
                       ("B is the best", "B"), ("C would be", "C"), ("The answer is B because", "B"),
                       ("**Answer:** B", "B"), ("**Answer**: B", "B"), ("\nB", "B"), ("B)", "B"), ("A.", "A"),
                       ("A", "A"), ("I would choose C", "C"), ("b", None)]:
        assert parse_letter(text) == want, text


def test_gen_check_sample_is_deterministic():
    ids = [f"x{i}" for i in range(1000)]
    a, b = select_gen_check(ids, 0.02, 0), select_gen_check(ids, 0.02, 0)
    assert a == b and len(a) == 20
    assert len(select_gen_check(ids[:10], 0.02, 0)) == 1  # at least one item
    assert select_gen_check(ids, 0.0, 0) == set()


def test_consistency_counts_unparsed_as_disagreement():
    df = pd.DataFrame({"argmax_letter": ["A", "B", "C", "D"], "gen_letter": ["A", None, "C", "A"],
                       "gen_text": ["A", "Nike", "C", "A"]})
    c = consistency(pd.concat([df, pd.DataFrame({"argmax_letter": ["A"], "gen_letter": [None], "gen_text": [None]})]))
    assert c == {"n_checked": 4, "agreement": 0.5, "n_unparsed": 1}


# --------------------------------------------------------------------------- synthetic choice model
@pytest.fixture(scope="module")
def tiny_items():
    cfg = config_from_dict({"name": "t", "seed": 3, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 2,
        "evidence_levels": [0, 4], "handicaps": [0, 2], "probe_entities_per_category": 1, "probe_templates": 1}})
    return generate(cfg)["choice"]


def fake_scores(items, position_logit, name_bonus=None):
    """Letter probabilities from a position prior plus an optional per-name utility."""
    rows = []
    for it in items:
        logits = np.array(position_logit, dtype=float).copy()
        if name_bonus:
            for j, o in enumerate(it["options"]):
                logits[j] += name_bonus.get(o["name"], 0.0)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        t = LETTERS.index(it["target_label"])
        rows.append({"item_id": it["item_id"], "s_target": float(s_target(p[t])),
                     **{f"p_{L}": p[k] for k, L in enumerate(LETTERS)}})
    return pd.DataFrame(rows)


def test_rotation_average_cancels_pure_position_bias(tiny_items):
    scores = fake_scores(tiny_items, [2.0, -1.0, 0.0, 1.0])  # strong letter prior, no content effect
    assert scores["s_target"].std() > 1.0  # unusable before averaging
    ne = paired_name_effect(rotation_average(scores, items_meta(tiny_items)))
    assert len(ne) == len({i["pair_id"] for i in tiny_items})
    assert np.allclose(ne["name_effect"], 0.0, atol=1e-12)
    # every (pair, version) ends with the same rotation-averaged score
    pairs = rotation_average(scores, items_meta(tiny_items))
    assert pairs["s_target"].std() < 1e-12


def test_name_bonus_recovered_regardless_of_position_prior(tiny_items):
    bonus = {"Nike": 1.5}
    flat = paired_name_effect(rotation_average(fake_scores(tiny_items, [0, 0, 0, 0], bonus), items_meta(tiny_items)))
    biased = paired_name_effect(rotation_average(fake_scores(tiny_items, [2.0, -1.0, 0.0, 1.0], bonus),
                                                 items_meta(tiny_items)))
    for ne in (flat, biased):
        nike = ne[ne["entity"] == "Nike"]["name_effect"]
        others = ne[ne["entity"] != "Nike"]["name_effect"]
        assert (nike > 0.5).all()
        assert (others <= 1e-9).all()  # other targets never gain from Nike (it is at most a competitor)
    # without rotation averaging the position prior would dominate: single rotations disagree wildly
    sc = fake_scores(tiny_items, [2.0, -1.0, 0.0, 1.0], bonus).merge(items_meta(tiny_items), on="item_id")
    per_rot = sc[sc["pair_entity"] == "Nike"].groupby("rotation")["s_target"].mean()
    assert per_rot.max() - per_rot.min() > 2.0


def test_paired_effect_never_crosses_pair_ids():
    ps = pd.DataFrame([
        {"pair_id": "x-1", "target_is_placebo": False, "category": "c", "entity": "E", "evidence_level": 0,
         "handicap": 0, "s_target": 1.0},
        {"pair_id": "x-2", "target_is_placebo": True, "category": "c", "entity": "E", "evidence_level": 0,
         "handicap": 0, "s_target": -5.0},
        {"pair_id": "x-1", "target_is_placebo": True, "category": "c", "entity": "E", "evidence_level": 0,
         "handicap": 0, "s_target": 0.25},
        {"pair_id": "x-3", "target_is_placebo": False, "category": "c", "entity": "F", "evidence_level": 0,
         "handicap": 0, "s_target": 9.0},  # no placebo version -> dropped, never matched to x-2
    ]).sample(frac=1.0, random_state=0)
    ne = paired_name_effect(ps)
    assert ne["pair_id"].tolist() == ["x-1"]
    assert ne["name_effect"].tolist() == [0.75]


def test_rotation_average_drops_incomplete_groups(tiny_items):
    scores = fake_scores(tiny_items, [0, 0, 0, 0])
    scores = scores[scores["item_id"] != tiny_items[0]["item_id"]]
    pairs = rotation_average(scores, items_meta(tiny_items))
    assert (pairs["n_rotations"] == 4).all()
    assert tiny_items[0]["pair_id"] not in set(paired_name_effect(pairs)["pair_id"])


def test_entity_evidence_table_and_ladder(tiny_items):
    ne = paired_name_effect(rotation_average(fake_scores(tiny_items, [0, 0, 0, 0], {"Nike": 1.0}),
                                             items_meta(tiny_items)))
    t = entity_evidence_table(ne)
    assert set(t.columns) >= {"category", "entity", "ne_e0", "ne_e4", "n_pairs"}
    out, info = ladder_regression(t)
    assert "ladder_residual" in out and info["n"] == len(t)


def test_robust_origin_regression_resists_outlier():
    x = np.arange(1.0, 11.0)
    y = 0.5 * x
    y[3] += 10.0  # one promoted-looking outlier
    fit = robust_origin_regression(x, y)
    ols = float((x @ y) / (x @ x))
    assert abs(fit["slope"] - 0.5) < 0.05 < abs(ols - 0.5)
    assert int(np.nanargmax(np.abs(fit["residuals"]))) == 3
    assert math.isnan(robust_origin_regression(np.zeros(3), np.ones(3))["slope"])


# --------------------------------------------------------------------------- open-ended
def test_brand_matcher():
    m = BrandMatcher([("Nike", [], False), ("Brooks", ["Brooks Running"], False), ("Reebok", [], False),
                      ("KIND", ["Kind", "KIND Snacks"], True), ("Hermès", [], False),
                      ("Google Pixel", ["Pixel", "Google"], False), ("Adidas", ["adidas"], False)])
    assert m.match("Nike") == "Nike"
    assert m.match("Sure! Nike's Pegasus is great") == "Nike"
    assert m.match("**Brooks Running**") == "Brooks"
    assert m.match("Reeboks") == "Reebok"
    assert m.match("any kind of chips") is None
    assert m.match("KIND bars") == "KIND" and m.match("Kind Snacks.") == "KIND"
    assert m.match("hermes") == "Hermès"
    assert m.match("Google Pixel 8") == "Google Pixel"
    assert m.match("Adidas or Nike") == "Adidas"
    assert m.match("As an AI language model, I can't") is None
    assert m.match("Nikeland") is None
    caps = BrandMatcher([("Visa", [], True), ("Chase", ["JPMorgan Chase"], True), ("Mastercard", ["MasterCard"], False)])
    assert caps.match("VISA") == "Visa" and caps.match("VISA, MASTERcard") == "Visa"
    assert caps.match("CHASE") == "Chase" and caps.match("JPMORGAN CHASE") == "Chase"
    assert caps.match("chase the best deal") is None


def test_variant_dedup_uses_token_prefixes_not_strings():
    assert brand_variants("Hoka", ["HOKA", "Hoka"]) == ["Hoka", "HOKA"]
    # ' Hoka' + ' One One' extends the ' Hoka' token path -> already counted, dropped
    hoka = dedup_token_prefix([("Hoka", [10]), ("Hoka One One", [10, 11, 11]), ("HOKA", [12])])
    assert [v for v, _ in hoka] == ["HOKA", "Hoka"]
    # ' Fujifilm' = [Fuj, if, ilm] does NOT extend ' Fuji' = [Fuji]: disjoint paths, both kept
    fuji = dedup_token_prefix([("Fujifilm", [7, 8, 9]), ("Fuji", [6])])
    assert sorted(v for v, _ in fuji) == ["Fuji", "Fujifilm"]
    moto = dedup_token_prefix([("Motorola", [5]), ("Moto", [4])])
    assert sorted(v for v, _ in moto) == ["Moto", "Motorola"]


def test_pick_lists_balance_named_and_picked():
    lists = pick_lists("Nike", "Brooks", ["Asics", "Hoka"])
    assert lists == [["Nike", "Asics", "Brooks"], ["Nike", "Hoka", "Brooks"]]
    assert np.isclose(pick_prob_from_scores(np.log([1.0, 2.0, 1.0])).sum(), 1.0)


def test_pick_aggregation_and_shares():
    rows = []
    for order, listed in (("forward", ["Nike", "Asics", "Brooks"]), ("reversed", ["Brooks", "Asics", "Nike"])):
        for pos, b in enumerate(listed):
            # Nike has two spellings that together carry prob 0.4 (0.3 + 0.1)
            probs = {"Nike": [0.3, 0.1], "Asics": [0.2], "Brooks": [0.4 if order == "forward" else 0.2]}[b]
            for k, p in enumerate(probs):
                rows.append({"model_id": "m", "split": "dev", "category": "running shoe", "list_id": 0,
                             "order": order, "position": pos, "brand": b, "variant": f"{b}{k}", "list": "|".join(listed),
                             "logprob": math.log(p)})
    picks = aggregate_picks(pd.DataFrame(rows))
    fwd = picks[picks["order"] == "forward"].set_index("brand")["prob"]
    assert fwd["Nike"] == pytest.approx(0.4) and fwd["Brooks"] == pytest.approx(0.4)
    rev = picks[picks["order"] == "reversed"].set_index("brand")["prob"]
    assert rev["Nike"] == pytest.approx(0.5) and rev["Brooks"] == pytest.approx(0.25)
    gens = pd.DataFrame({"category": "running shoe", "prompt_type": "name",
                         "matched_entity": ["Nike", "Nike", "Brooks", None]})
    sh = compute_shares(gens, picks, {"running shoe": {"category": "running shoe", "named": "Nike", "picked": "Brooks"}},
                        "m", "dev").set_index(["prompt_type", "brand_role"])["share"]
    assert sh[("name", "named")] == 0.5 and sh[("name", "picked")] == 0.25
    assert sh[("pick", "named")] == pytest.approx(0.45) and sh[("pick", "picked")] == pytest.approx(0.325)


def test_open_ended_targets_cover_appendix_a(work_dir):
    from rtp.data.build import build_data
    from rtp.data.catalogue import load_natural_negatives
    from rtp.config import REPO_ROOT
    from rtp.measure.open_ended import open_ended_targets

    raw = {"name": "oe", "data": {"categories": ["running shoe", "beer"], "test_categories": ["beer"],
                                  "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
                                  "evidence_levels": [0], "probe_entities_per_category": 1, "probe_templates": 1}}
    build_data(config_from_dict(raw))
    narrow = open_ended_targets(config_from_dict(raw), "dev", False)
    assert list(narrow) == ["running shoe"]
    raw["open_ended"] = {"all_natural_negatives": True}
    wide = open_ended_targets(config_from_dict(raw), "dev", False)
    nn = {r["category"] for r in load_natural_negatives(REPO_ROOT / "data/seeds/natural_negatives.csv")}
    assert set(wide) == nn - {"beer"}  # beer is a sealed test category of this run
    assert wide["AI assistant"]["source"] == "open_ended_only"
    assert [b for b, _ in wide["AI assistant"]["brands"]][:2] == ["Siri", "Claude"]
    assert wide["running shoe"]["source"] == "catalogue"
    test = open_ended_targets(config_from_dict(raw), "test", True)
    assert list(test) == ["beer"]


def test_pick_needs_three_distinct_brands(work_dir, tmp_path):
    import re as _re
    import shutil

    from rtp.config import REPO_ROOT
    from rtp.data.build import build_data
    from rtp.measure.open_ended import open_ended_targets

    seeds = tmp_path / "seeds"
    shutil.copytree(REPO_ROOT / "data" / "seeds", seeds)
    cat = seeds / "categories.yaml"
    text = cat.read_text()
    # a reviewer trims "search engine" to its two Appendix A brands
    text = _re.sub(r"(  search engine:\n    needs_review: true\n    brands:\n(?:      - .*\n){2})(?:      - .*\n)+",
                   r"\1", text)
    cat.write_text(text)
    raw = {"name": "trim", "data": {"categories_file": str(cat), "placebo_file": str(seeds / "placebo_names.yaml"),
                                    "natural_negatives_file": str(seeds / "natural_negatives.csv"),
                                    "categories": ["running shoe"], "n_real_per_category": 3,
                                    "n_placebo_per_category": 3, "cards_per_cell": 1, "evidence_levels": [0],
                                    "probe_entities_per_category": 1, "probe_templates": 1},
           "open_ended": {"all_natural_negatives": True}}
    build_data(config_from_dict(raw))
    with pytest.raises(ValueError, match="search engine"):
        open_ended_targets(config_from_dict(raw), "dev", False)



def test_finished_stage_with_other_config_is_refused(work_dir):
    from rtp import io
    from rtp.data.build import build_data
    from rtp.measure.choice import run_measure
    from rtp.paths import stage_dir
    from rtp.report import build_report

    raw = {"name": "st", "data": {"categories": ["running shoe"], "n_real_per_category": 3,
                                  "n_placebo_per_category": 3, "cards_per_cell": 1, "evidence_levels": [0],
                                  "probe_entities_per_category": 1, "probe_templates": 1}}
    cfg = config_from_dict(raw)
    build_data(cfg)
    sdir = stage_dir(cfg.name, "dummy/model", "measure")
    io.prepare_stage_dir(sdir, cfg.to_yaml(), "hash-from-an-older-config", extra_meta={"model_id": "dummy/model"})
    io.mark_done(sdir, {"n_items": 1})
    with pytest.raises(io.StageConfigMismatch):
        run_measure(cfg, "dummy/model")  # refused before any model is loaded
    rep = build_report(cfg)
    assert "Excluded stale stages" in rep and "dummy/model / measure" in rep


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_end_to_end_m1_m2_on_smoke_model(smoke_lm, work_dir, monkeypatch):
    import rtp.measure.choice as choice_mod
    import rtp.measure.open_ended as oe_mod
    from rtp import io
    from rtp.data.build import build_data
    from rtp.report import run_report

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)  # reuse the session's model
    cfg = config_from_dict({"name": "it", "seed": 0, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0, 4], "handicaps": [0], "probe_entities_per_category": 1, "probe_templates": 1},
        "measure": {"gen_check_fraction": 0.1, "shard_size": 20},
        "open_ended": {"n_samples": 2, "max_new_tokens": 8}})
    build_data(cfg)
    mdir = choice_mod.run_measure(cfg, smoke_lm.model_id)
    odir = oe_mod.run_open_ended(cfg, smoke_lm.model_id)
    rep = run_report(cfg)
    scores = pd.read_parquet(mdir / "scores.parquet")
    assert list(scores.columns[:8]) == ["item_id", "model_id", "p_A", "p_B", "p_C", "p_D", "s_target", "gen_letter"]
    assert len(scores) == 3 * 2 * 2 * 4 and np.allclose(scores[["p_A", "p_B", "p_C", "p_D"]].sum(axis=1), 1.0)
    assert scores["gen_text"].notna().sum() == round(0.1 * 48)
    assert io.is_done(mdir) and io.is_done(odir) and rep.exists()
    shares = pd.read_parquet(odir / "shares.parquet")
    assert set(shares["brand"]) == {"Nike", "Brooks"}
    assert set(shares["prompt_type"]) == {"name", "choose", "recommend", "pick"}
    assert len(list((mdir / "shards").glob("*.done"))) == 3  # 48 items / shard_size 20
    tok = smoke_lm.tokenizer
    assert {"A", " A"} <= {tok.decode([i]) for i in letter_token_ids(tok)["A"]}


@pytest.mark.integration
def test_generation_ignores_checkpoint_sampling_defaults(smoke_lm):
    """A checkpoint saved with transformers >= 4.50 and do_sample=True must not turn greedy into sampling."""
    import copy

    from rtp.infer import generate
    from rtp.models import encode, format_chat

    tok = smoke_lm.tokenizer
    seqs = [encode(tok, format_chat(tok, "Name a running shoe brand. Reply with the name only."), True)]
    reference = generate(smoke_lm, seqs, max_new_tokens=8, do_sample=False)
    saved = smoke_lm.model.generation_config
    patched = copy.deepcopy(saved)
    patched.transformers_version = "4.57.1"
    # untruncated hot sampling: if these leaked into generate(), every run would differ from greedy
    patched.do_sample, patched.temperature, patched.top_p, patched.top_k = True, 5.0, 1.0, 0
    smoke_lm.model.generation_config = patched
    try:
        runs = [generate(smoke_lm, seqs, max_new_tokens=8, do_sample=False) for _ in range(3)]
    finally:
        smoke_lm.model.generation_config = saved
    assert all(r == reference for r in runs)


@pytest.mark.integration
def test_prompt_rendering_chat_vs_base(smoke_lm):
    from dataclasses import replace

    from rtp.measure.choice import COMPLETION_SUFFIX, render_inputs
    from rtp.measure.open_ended import PICK_PREFILL, continuation_ids, pick_prompt_text

    tok = smoke_lm.tokenizer
    chat = tok.decode(render_inputs(smoke_lm, ["Which one? Answer with a single letter."])[0])
    assert chat.startswith("<|im_start|>") and chat.rstrip().endswith("<|im_start|>assistant")
    base_lm = replace(smoke_lm, is_chat=False)  # same weights, measured as if it were a base checkpoint
    base = tok.decode(render_inputs(base_lm, ["Which one? Answer with a single letter."])[0])
    assert "<|im_start|>" not in base and base.endswith(COMPLETION_SUFFIX)
    pick = pick_prompt_text(smoke_lm, "running shoe", ["Nike", "Asics", "Brooks"])
    assert pick.endswith(PICK_PREFILL) and "<|im_end|>" not in pick.split("<|im_start|>assistant")[-1]
    pre, cont, fallback = continuation_ids(smoke_lm, pick, " Brooks")
    assert not fallback and tok.decode(cont) == " Brooks"
    assert pick_prompt_text(base_lm, "running shoe", ["Nike", "Asics", "Brooks"]).endswith("Which one would you pick? I'd pick")


@pytest.mark.integration
def test_open_ended_all_natural_negatives_on_smoke_model(smoke_lm, work_dir, monkeypatch):
    import rtp.measure.open_ended as oe_mod
    from rtp.data.build import build_data

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    cfg = config_from_dict({"name": "oeall", "data": {  # 4 reals: camera = Canon, Sony, Nikon, Fujifilm
        "categories": ["running shoe"], "n_real_per_category": 4, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0], "probe_entities_per_category": 1, "probe_templates": 1},
        "open_ended": {"n_samples": 1, "max_new_tokens": 4, "prompt_types": ["name", "pick"],
                       "all_natural_negatives": True}})
    build_data(cfg)
    odir = oe_mod.run_open_ended(cfg, smoke_lm.model_id)
    shares = pd.read_parquet(odir / "shares.parquet")
    assert shares["category"].nunique() == 26  # every Appendix A row gets Named/Picked shares
    assert set(shares[shares["category"] == "AI assistant"]["brand"]) == {"Siri", "Claude"}
    variants = pd.read_parquet(odir / "pick_variants.parquet")
    fuji = variants[(variants["category"] == "camera") & (variants["brand"] == "Fujifilm")]["variant"]
    assert "Fujifilm" in set(fuji)  # canonical spelling is scored even though "Fuji" is a string prefix
