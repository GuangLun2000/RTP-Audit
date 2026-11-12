"""Answer-path estimator (rtp.measure.answers), deferred-answer parsing, the new measure diagnostics and
matched card sets. CPU-only toy models, plus one brute-force equivalence test on the smoke model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rtp.config import config_from_dict
from rtp.data.build import generate
from rtp.data.schema import SchemaError, validate_choice_items
from rtp.measure.answers import (EPS, LEAK, PositionResult, batched_tries, build_item_paths, estimators,
                                 event_masses, logit)
from rtp.measure.choice import parse_final_answer
from rtp.measure.name_effect import (consistency, format_stability, handicap_response, manipulation_ratio,
                                     name_effect_bounds)

PIECES = ["A", " A", "B", " B", "C", " C", "D", " D", "**", " **", "(", " (", "ell", "Nike", " Nike", "Asics",
          " Asics", "Puma", " Puma", "Bev", " Bev", "odel", " good", "x"]


class PieceTok:
    """Greedy longest-match tokenizer over PIECES (so "Dell" = "D" + "ell", "**A" = "**" + "A")."""

    def __init__(self):
        self.ids = {p: i for i, p in enumerate(PIECES)}

    def __len__(self):
        return len(PIECES)

    def __call__(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            best = max((p for p in PIECES if text.startswith(p, i)), key=len)
            out.append(self.ids[best])
            i += len(best)
        return {"input_ids": out}

    def batch_decode(self, seqs):
        return ["".join(PIECES[i] for i in s) for s in seqs]


TOK = PieceTok()
T = TOK.ids
LETTER_IDS = {L: [T[L], T[" " + L]] for L in "ABCD"}
NAMES = ["Dell", "Nike", "Asics", "Puma"]  # option A's name starts with the letter token "D"


def toy_paths():
    return build_item_paths(TOK, NAMES, "Bevodel", LETTER_IDS, {})


# next-token distributions of a toy model, keyed by the generated prefix (piece strings)
TOY = {(): {"D": 0.30, "Nike": 0.20, " B": 0.10, "A": 0.20, "Bev": 0.05, "x": 0.15},
       ("D",): {"ell": 0.5, "x": 0.5}, ("A",): {" good": 0.25, "x": 0.75}, ("Bev",): {"odel": 0.8, "x": 0.2}}


def toy_expand(todo, needs, table=None):
    table = TOY if table is None else table
    out = []
    for (_, node), need in zip(todo, needs):
        dist = table.get(tuple(PIECES[i] for i in node), {"x": 1.0})
        out.append(({t: math.log(dist.get(PIECES[t], 1e-12)) for t in need}, dist.get(" good", 0.0)))
    return out


def toy_root(table=None):
    table = TOY if table is None else table
    return np.log(np.array([[table[()].get(p, 1e-12) for p in PIECES]]))


def test_paths_dedup_and_leak_class():
    paths = toy_paths()
    a = {p for p, _ in paths.classes[0]}
    assert (T["D"], T["ell"]) in a and (T["A"],) in a  # Dell (name) and A (letter) both answer option A
    d = {p for p, _ in paths.classes[3]}
    assert (T["D"],) in d and all(p[:1] != (T["D"],) or len(p) == 1 for p in d)
    assert {p for p, _ in paths.classes[LEAK]} >= {(T["Bev"], T["odel"])}
    for cls in paths.classes:  # no path extends another path of its own class
        ps = [p for p, _ in cls]
        assert not any(q != p and p[: len(q)] == q for p in ps for q in ps)


def test_event_masses_letter_collision_article_and_leak():
    res = batched_tries([toy_paths()], toy_root(), toy_expand, art_set={T[" good"]})[0]
    # A: letter "A" not followed by an article word (0.2 * 0.75) + name "D"+"ell" (0.3 * 0.5)
    # B: " B" (0.1) + "Nike" (0.2); D: letter "D" minus its extension into option A's name (0.3 - 0.15)
    assert res.q[:4] == pytest.approx([0.30, 0.30, 0.0, 0.15], abs=1e-6)
    assert res.q[LEAK] == pytest.approx(0.04, abs=1e-6)
    assert res.q_name[0] == pytest.approx(0.15, abs=1e-6) and res.q_name[1] == pytest.approx(0.20, abs=1e-6)
    assert res.pruned < 10 * EPS


def test_article_factor_applies_before_other_class_extensions():
    """Option B's name starts with the letter token "A" ("A" + "x"). The letter-A event is "A" followed by
    neither an article word nor option B's name: 0.4 * (1 - 0.5) - 0.2 = 0 (not (0.4 - 0.2) * 0.5)."""
    table = {(): {"A": 0.4, "Nike": 0.2, "x": 0.4}, ("A",): {" good": 0.5, "x": 0.5}}
    paths = build_item_paths(TOK, ["Nike", "Ax", "Puma", "Dell"], None, LETTER_IDS, {})
    res = batched_tries([paths], toy_root(table), lambda todo, needs: toy_expand(todo, needs, table),
                        art_set={T[" good"]})[0]
    assert res.q[:4] == pytest.approx([0.2, 0.2, 0.0, 0.0], abs=1e-6)
    assert res.article == pytest.approx(0.2, abs=1e-6)


def test_answer_continuations_are_not_articles():
    from rtp.measure.answers import article_ids

    class Tok:
        pieces = [" good", " is", " would", " great", "A", " and"]

        def __len__(self):
            return len(self.pieces)

        def batch_decode(self, seqs):
            return [self.pieces[s[0]] for s in seqs]

    assert article_ids(Tok()) == frozenset({0, 3})


def test_pruned_mass_is_reported():
    paths = toy_paths()
    root = toy_root()
    root[0, T["Bev"]] = math.log(EPS / 2)  # below EPS: not expanded, counted as pruned, leak unknown (0)
    res = batched_tries([paths], root, toy_expand)[0]
    assert res.q[LEAK] == 0.0 and res.pruned >= EPS / 2


def test_estimators_o5_bounds_and_coverage():
    free = PositionResult(q=np.array([0.30, 0.10, 0.0, 0.0, 0.01]), q_name=np.array([0.2, 0, 0, 0, 0.01]), pruned=0.0)
    forced = PositionResult(q=np.array([0.2, 0.6, 0.1, 0.1, 0.0]), q_name=np.zeros(5), pruned=0.0)
    e = estimators(free, forced, target=1)
    P = e["P"]
    assert P.sum() == pytest.approx(1.0)
    assert P == pytest.approx(np.array([0.30, 0.10, 0.0, 0.0]) + 0.6 * np.array([0.2, 0.6, 0.1, 0.1]))
    assert e["c_free"] == pytest.approx(0.4) and e["c_forced"] == pytest.approx(1.0)
    assert e["s_target_lo"] <= e["s_target"] <= e["s_target_hi"]
    assert e["s_target_free"] == pytest.approx(float(logit(0.25)))
    assert e["s_target_forced"] == pytest.approx(float(logit(0.6)))
    assert e["name_share_free"] == pytest.approx(0.5) and e["leak_free"] == pytest.approx(0.01)


def test_parse_final_answer():
    opts = [{"label": L, "name": n} for L, n in zip("ABCD", ["Nike", "Levi's", "New Balance", "Dell"])]
    cases = [("Based on the reviews, the best choice is **C**.", "C"),
             ("Let's compare. A) Nike: 4.5 stars. B) Levi's: 3 stars. C) New Balance", None),
             ("A good choice here would be Dell, since it", "D"),
             ("I'd go with Levis because", "B"),
             ("Considering price and quality, I recommend option B.", "B"),
             ("Nike is better than Dell overall", None),
             ("The answer is A because", "A"), ("", None)]
    for text, want in cases:
        assert parse_final_answer(text, opts) == want, text


def test_parser_regressions():
    from rtp.measure.choice import parse_answer, parse_letter

    o = [{"label": L, "name": n} for L, n in zip("ABCD", ["Nike", "L'Oréal", "A&W", "The North Face"])]
    assert parse_final_answer("Comparing A and C, I recommend Option B.", o) == "B"  # capitalised Option after a cue
    assert parse_final_answer("A&W is the best soda here.", o) == "C"  # the A of A&W is not a letter
    assert parse_final_answer("I recommend L'Oreal", o) == "B" and parse_answer("Loreal", o) == "B"  # accents
    assert parse_answer("A&W", o) == "C" and parse_answer("The best choice is", o) is None
    assert parse_answer("The North", o) == "D"  # a truncated name is still read
    assert parse_letter("A is the best choice") == "A" and parse_letter("A good choice") is None
    assert parse_letter("A look at the reviews") is None  # "look" is a noun after the article
    o2 = [{"label": L, "name": n} for L, n in zip("ABCD", ["Bevodel", "Ford", "Allswell", "Topo Athletic"])]
    assert parse_answer("Bev", o2) == "A" and parse_answer("For an electric car", o2) is None
    assert parse_answer("All four options are", o2) is None and parse_answer("Top pick here is", o2) is None
    o3 = [{"label": L, "name": n} for L, n in zip("ABCD", ["Sharp", "Element", "Insignia", "TCL"])]
    cs = {"Sharp", "Element", "Insignia"}  # catalogue match_case: ordinary words match as written only
    assert parse_final_answer("D offers a sharp picture and good value", o3, cs) == "D"
    assert parse_final_answer("I recommend SHARP", o3, cs) == "A"
    assert parse_final_answer("D offers a sharp picture and good value", o3) is None  # without the flags


def test_consistency_modes_and_chance():
    s = pd.DataFrame({"gen_text": ["x"] * 6 + [None],
                      "gen_mode": ["immediate"] * 3 + ["deferred"] * 3 + [None],
                      "gen_letter": ["A", "B", "A", "C", None, "C", None],
                      "argmax_letter": ["A", "B", "B", "C", "C", "D", "A"],
                      "argmax_free": ["A", "B", "A", "D", "D", "D", "A"],
                      "argmax_forced": ["B", "B", "B", "C", "A", "C", "A"]})
    c = consistency(s)
    assert c["n_checked"] == 6 and c["agreement"] == pytest.approx(3 / 6) and c["n_unparsed"] == 1
    assert c["agreement_immediate"] == pytest.approx(1.0) and c["agreement_deferred"] == pytest.approx(2 / 3)
    assert c["n_unparsed_deferred"] == 1 and 0 < c["chance"] < 1


def test_handicap_response_within_card_sets_and_ratio():
    deltas = [0.0, 0.1, 0.3, 0.5, 1.0]
    rows = []
    for k in range(6):
        base = 0.3 * k  # card-set level differs; the within-set response is exactly 2 logits per star
        for placebo in (False, True):
            for e in (3, 4):
                for h in range(5):
                    rows.append({"card_set": f"s{k}", "target_is_placebo": placebo, "evidence_level": e,
                                 "handicap": h, "s_target": base - 2.0 * deltas[h] + (0.5 if not placebo else 0)})
    resp = handicap_response(pd.DataFrame(rows), deltas)
    assert {r["version"] for r in resp} == {"real", "placebo"}
    assert all(r["per_star_drop"] == pytest.approx(2.0) and r["se"] == pytest.approx(0.0, abs=1e-9) for r in resp)
    table = pd.DataFrame({"ne_e0": [1.0, -4.0, 8.0]})
    assert manipulation_ratio(resp, table) == pytest.approx(0.5)


def test_format_stability_flags_shift():
    cov = pd.DataFrame({"evidence_level": [0, 4, 0, 4], "version": ["real", "real", "placebo", "placebo"],
                        "name_share_free": [0.1, 0.5, 0.1, 0.15], "c_free": [0.9, 0.85, 0.9, 0.9]})
    flags = format_stability(cov)
    assert len(flags) == 1 and flags[0]["version"] == "real" and flags[0]["column"] == "name_share_free"


def test_name_effect_bounds_contain_estimate():
    rows, meta = [], []
    for rot in range(4):
        for placebo in (False, True):
            iid = f"p1-{'p' if placebo else 'r'}{rot}"
            s, lo, hi = (1.0, 0.5, 1.5) if not placebo else (0.0, -0.2, 0.4)
            rows.append({"item_id": iid, "s_target": s, "s_target_lo": lo, "s_target_hi": hi})
            meta.append({"item_id": iid, "pair_id": "p1", "card_set": "c", "category": "c", "pair_entity": "E",
                         "evidence_level": 0, "handicap": 0, "target_is_placebo": placebo, "rotation": rot,
                         "target_label": "ABCD"[rot]})
    b = name_effect_bounds(pd.DataFrame(rows), pd.DataFrame(meta))
    assert b["n_entities"] == 1 and b["share_excluding_zero"] == 1.0 and b["median_width"] == pytest.approx(1.6)


def test_matched_card_sets_share_cards_across_cells():
    cfg = config_from_dict({"name": "cs", "seed": 3, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 2,
        "probe_entities_per_category": 1, "probe_templates": 1}})
    items = generate(cfg)["choice"]
    validate_choice_items(items)
    by_set: dict[tuple, list] = {}
    for it in items:
        if it["evidence_level"] >= 1 and it["rotation"] == 0 and not it["target_is_placebo"]:
            by_set.setdefault(it["card_set"], []).append(it)
    for group in by_set.values():
        assert len(group) == 1 + 1 + 5 + 5  # evidence 1, 2 and 3-4 x 5 handicaps share the card set
        ref = next(g for g in group if g["evidence_level"] == 4 and g["handicap"] == 0)
        for g in group:
            for o, r in zip(g["options"], ref["options"]):
                assert o["name"] == r["name"]
                if g["evidence_level"] == 4 and o["role"] != "target":
                    assert o == r  # same cards; only the target's rating moves with the handicap
    # change a competitor's price consistently in all 8 items of ONE pair (both versions, all rotations): the pair
    # check passes, so only the card-set check can catch that this cell no longer shows the set's cards
    import copy

    pid = next(it["pair_id"] for it in items if it["evidence_level"] == 2)
    bad = copy.deepcopy(items)
    comp = next(o["name"] for o in next(it for it in bad if it["pair_id"] == pid)["options"] if o["role"] == "competitor")
    for it in bad:
        if it["pair_id"] == pid:
            for o in it["options"]:
                if o["name"] == comp:
                    o["price"] = o["price"] + 1
    with pytest.raises(SchemaError, match="card set .* field 'price'"):
        validate_choice_items(bad)


def test_card_set_substitutes_rotate_and_se_units_are_substitutes():
    from rtp.measure.name_effect import entity_evidence_table, items_meta, paired_name_effect, rotation_average

    cfg = config_from_dict({"name": "rot", "seed": 4, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 3,
        "e0_pairs_per_substitute": 2, "evidence_levels": [0, 1, 4], "handicaps": [0, 2],
        "probe_entities_per_category": 1, "probe_templates": 1}})
    items = generate(cfg)["choice"]
    subs: dict[str, set] = {}
    use: dict[str, int] = {}
    for it in items:
        if it["evidence_level"] >= 1 and it["rotation"] == 0 and not it["target_is_placebo"]:
            subs.setdefault(it["pair_entity"], set()).add(it["placebo_substitute"])
            if it["evidence_level"] == 1:
                use[it["placebo_substitute"]] = use.get(it["placebo_substitute"], 0) + 1
    assert all(len(v) == 3 for v in subs.values())  # every target sees every placebo at evidence >= 1
    assert set(use.values()) == {3}  # and every placebo serves equally often
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"item_id": [it["item_id"] for it in items], "s_target": rng.normal(size=len(items))})
    t = entity_evidence_table(paired_name_effect(rotation_average(scores, items_meta(items))))
    assert set(t["n_e0"]) == {3} and set(t["n_e4"]) == {3}  # units = substitutes, not 6 pairs / 6 cells


def test_card_set_check_compares_target_rating_within_a_handicap():
    import copy

    cfg = config_from_dict({"name": "tr", "seed": 3, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 3,
        "probe_entities_per_category": 1, "probe_templates": 1}})
    items = generate(cfg)["choice"]
    pid = next(it["pair_id"] for it in items if it["evidence_level"] == 4 and it["handicap"] == 0)
    bad = copy.deepcopy(items)
    for it in bad:  # handicap applied twice in ONE cell: consistent within the pair, wrong for the card set
        if it["pair_id"] == pid:
            for o in it["options"]:
                if o["role"] == "target":
                    o["rating"] = round(o["rating"] - 2.0, 1)
    with pytest.raises(SchemaError, match="field 'rating'"):
        validate_choice_items(bad)


# --------------------------------------------------------------------------- integration
def _brute_force(lm, prefix, paths, art_ids):
    """Exact path log-masses and article masses by one full forward pass per path (no pruning, no cache)."""
    import torch

    logmass, article = {}, {}
    nodes = {p for cls in paths.classes for p, _ in cls} | set(paths.a_letter)
    for p in nodes:
        ids = torch.tensor([list(prefix) + list(p)], device=lm.device)
        with torch.inference_mode():
            lp = torch.log_softmax(lm.model(input_ids=ids).logits[0].float(), dim=-1)
        n = len(prefix)
        logmass[p] = float(sum(lp[n - 1 + k, t] for k, t in enumerate(p)))
        if p in paths.a_letter:
            article[p] = float(lp[-1, art_ids].exp().sum())
    return logmass, article


@pytest.mark.integration
def test_answer_distributions_match_brute_force(smoke_lm, monkeypatch):
    import torch

    from rtp.measure.answers import answer_distributions, article_ids
    from rtp.measure.choice import item_answer_paths, letter_token_ids, render_inputs

    cfg = config_from_dict({"name": "bf", "seed": 1, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0, 4], "handicaps": [0], "probe_entities_per_category": 1, "probe_templates": 1}})
    items = generate(cfg)["choice"][::7][:6]
    letters = letter_token_ids(smoke_lm.tokenizer)
    paths = item_answer_paths(smoke_lm.tokenizer, items, letters)
    free = render_inputs(smoke_lm, [it["prompt"] for it in items], prefill=False)
    forced = render_inputs(smoke_lm, [it["prompt"] for it in items], prefill=True)
    import rtp.measure.answers as answers_mod

    def no_fallback(msg, *args):
        if "cache extension failed" in msg:
            raise AssertionError(msg % args)

    monkeypatch.setattr(answers_mod.log, "debug", no_fallback)  # the KV-cache path must be the one tested
    res = answer_distributions(smoke_lm, free, forced, paths, max_batch_size=4, max_batch_tokens=4096)
    art = torch.tensor(sorted(article_ids(smoke_lm.tokenizer)))
    worst = 0.0
    for i in range(len(items)):
        for pos, prefix in (("free", free[i]), ("forced", forced[i])):
            lm_, ar = _brute_force(smoke_lm, prefix, paths[i], art)
            q_ref = event_masses(paths[i], lm_, ar, frozenset(art.tolist()))[0]
            got = res[i][pos]
            worst = max(worst, float(np.abs(got.q - q_ref).max()) - got.pruned)
    assert worst < 1e-3


@pytest.mark.integration
def test_format_probe_on_smoke_model(smoke_lm, work_dir, monkeypatch):
    from rtp import io
    from rtp.data.build import build_data, load_choice_items
    from rtp.measure.format_probe import run_format_probe, sample_items

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    cfg = config_from_dict({"name": "fp", "seed": 0, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0, 4], "handicaps": [0], "probe_entities_per_category": 1, "probe_templates": 1}})
    build_data(cfg)
    picked = sample_items(load_choice_items(cfg, "dev"), 8, cfg.seed)
    assert len(picked) == 8 and {(it["evidence_level"], it["target_is_placebo"]) for it in picked} == {
        (0, False), (0, True), (4, False), (4, True)}
    out = run_format_probe(cfg, smoke_lm.model_id, n_items=8)
    res = io.read_json(out / "format_probe.json")
    assert res["n_items"] == 8 and sum(res["modes"].values()) == 8 and 0 < res["c_forced"] <= 1.0001
    assert "| e | version |" in (out / "summary.md").read_text()
