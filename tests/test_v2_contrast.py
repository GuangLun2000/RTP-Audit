"""What cycle C0 taught the program: a ladder must resolve its first step; a stage change need not be one
multiplier; a low letter mass limits, it does not void; the token probe. Synthetic worlds with known truth."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rtp import scale, scorecard
from rtp.scale import checks, contrast

import test_scale_helpers as helpers
from test_scale_helpers import World, panel


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def analyze(world: World, name: str = "mid"):
    p = panel(name)
    return scale.analyze_model(p.items, p.score(world), p.steps)


# --------------------------------------------------------------------------- the first step of a ladder
def test_jump_index_is_zero_for_a_line_one_for_a_jump_and_does_not_move_with_scale():
    assert checks.jump_index(0.1, 1.0, 0.3, 3.0) == pytest.approx(0.0, abs=1e-12)
    assert checks.jump_index(0.1, 2.0, 0.3, 2.0) == pytest.approx(1.0)
    c0 = checks.jump_index(0.1, 1.88, 0.3, 2.56)  # Tulu 3 SFT, rating, cycle C0
    assert c0 == pytest.approx(0.6016, abs=1e-3) and c0 > checks.FIRST_STEP_JUMP_MAX
    assert checks.jump_index(0.1, 7 * 1.88, 0.3, 7 * 2.56) == pytest.approx(c0)
    assert checks.jump_index(0.1, 0.2, 0.3, 3.0) < 0  # a threshold: the first step is under-weighted
    assert math.isnan(checks.jump_index(0.1, 1.0, 0.3, 0.0)) and math.isnan(checks.jump_index(0.3, 1.0, 0.1, 2.0))


def test_ladder_check_fails_when_the_brands_sit_inside_a_jump(monkeypatch):
    """C0: the response to the smallest difference was most of the response to any difference, and every
    brand was worth less than that. The old rule (share beyond the top) passed such a ladder."""
    weak = World(real_mean=0.2, real_sd=0.15)  # names worth far less than the first step
    graded = analyze(weak)["checks"]["ladder"]
    rows = {(r["stratum"], r["evidence_type"]): r for r in graded["by_cell"] if r["arm"] == "pair"}
    x = panel("mid").steps["rating"]
    expect = ((x[0] / x[1]) ** helpers.SHAPE - x[0] / x[1]) / (1 - x[0] / x[1])
    assert rows[("attr", "rating")]["jump_index"] == pytest.approx(expect, abs=1e-6)
    assert expect < checks.FIRST_STEP_JUMP_MAX and rows[("attr", "rating")]["first_step_unresolved"] is False
    assert rows[("attr", "rating")]["share_in_first_step"] > 0.5  # small names on a graded ladder: still a pass
    assert graded["share_first_step_unresolved"] == 0.0 and graded["pass"] is True

    monkeypatch.setattr(helpers, "SHAPE", 0.1)  # almost a step function of the difference
    jump = analyze(weak)["checks"]["ladder"]
    rows = {(r["stratum"], r["evidence_type"]): r for r in jump["by_cell"] if r["arm"] == "pair"}
    assert rows[("attr", "rating")]["jump_index"] > checks.FIRST_STEP_JUMP_MAX
    assert rows[("attr", "rating")]["first_step_unresolved"] is True
    assert "jump_index" in rows[("attr", "rating")] and rows[("comp", "composite")]["first_step_unresolved"] is None
    assert jump["share_censored"] <= checks.CENSORED_SHARE_MAX  # nothing beyond the top: the old rule passes
    assert jump["share_first_step_unresolved"] > checks.CENSORED_SHARE_MAX and jump["pass"] is False

    strong = analyze(World(real_mean=2.8, real_sd=0.05))["checks"]["ladder"]  # same jump, most brands beyond it
    price = next(r for r in strong["by_cell"] if (r["arm"], r["stratum"], r["evidence_type"]) == ("pair", "attr", "price"))
    assert price["first_step_unresolved"] is True and price["share_in_first_step"] == 0.0
    inside = sum(r["share_in_first_step"] * r["n_entities"] for r in strong["by_cell"] if r["first_step_unresolved"])
    assert strong["share_first_step_unresolved"] == pytest.approx(inside / strong["n_entities"])  # only brands IN the step


def test_readout_block_reports_the_letter_mass_per_kind_of_item():
    rd = analyze(World(coverage=0.8))["checks"]["readout"]
    assert rd["pass"] is False and rd["mean_c_forced"] == pytest.approx(0.8)
    cells = rd["by_cell"]
    assert {(r["family"], r["tied"]) for r in cells} >= {("anchor", True), ("anchor", False), ("real", True)}
    assert sum(r["n_units"] for r in cells) == rd["n_units"]
    assert all(r["mean_c_forced"] == pytest.approx(0.8) and r["share_low_coverage"] == 1.0 for r in cells)


def test_anchor_check_names_the_cell_that_decides_it():
    ac = analyze(World())["checks"]["anchor_clean"]
    worst = ac["worst"]
    assert worst["ratio"] == pytest.approx(ac["max_ratio"]) and worst["sd_delta"] > worst["sd_pull"] > 0
    assert {"stratum", "evidence_type", "cross_stratum", "ratio_logit"} <= set(worst)


# --------------------------------------------------------------------------- one multiplier or several
def test_a_pure_scale_change_gives_every_contrast_family_the_same_multiplier():
    world = World(noise=0.3, noise_seed=1, position_bias=0.4)
    a, b = analyze(world), analyze(replace(world, gamma=2.0))
    logs = contrast.family_logs(a, b)
    rows = contrast.family_rows(logs)
    assert {r["family"] for r in rows} == set(contrast.FAMILIES)
    assert all(r["multiplier"] == pytest.approx(2.0, rel=1e-9) for r in rows), rows
    refs = contrast.reference_rows(logs)
    assert refs and all(abs(r["T1"]) < 1e-9 for r in refs)
    mult = scale.compare_models(a, b, n_boot=50, seed=0)["summary"]["scale"]
    cf = contrast.share_counterfactual(a["units"], b["units"], mult)
    assert cf and all(r["c"] == pytest.approx(2.0) and r["share_cf"] == pytest.approx(r["share_b"], abs=1e-12) for r in cf)
    assert all(r["explained"] == pytest.approx(1.0) for r in cf if r["explained"] is not None)
    assert any(r["explained"] is not None for r in cf)


def test_a_content_change_shows_in_the_real_names_against_every_reference():
    """B = A sharpened by 1.7 with every real name worth 1.25 times as much: the brand-free families all show
    1.7, the real names 1.7 * 1.25, and T1 is log 1.25 whichever reference is taken."""
    a, b = analyze(World(position_bias=0.3)), analyze(World(gamma=1.7, real_pull_mult=1.25, position_bias=0.3))
    assert not [r for r in contrast.family_rows(contrast.family_logs(analyze(World()), analyze(World(gamma=1.7))))
                if r["family"] == contrast.POSITION]  # no position bias at all: the family has no row
    logs = contrast.family_logs(a, b)
    for r in contrast.family_rows(logs):
        want = 1.7 * 1.25 if r["family"] == contrast.REAL else 1.7
        assert r["multiplier"] == pytest.approx(want, rel=1e-6), r
    refs = contrast.reference_rows(logs)
    assert {r["reference"] for r in refs} == {"fictional names", "whole curve (c)", "first step", "top step"}
    assert all(r["T1"] == pytest.approx(math.log(1.25), abs=1e-6) for r in refs)
    name = [r for r in refs if r["stratum"] == "name"]  # a bare name: the attribute-only curve of every type
    assert {r["ref_evidence_type"] for r in name if r["reference"] == "first step"} == {"rating", "price", "reviews"}
    # the whole-curve reference is the project's T1 over categories
    res = scale.compare_models(a, b, n_boot=50, seed=0)
    cell = next(c for c in res["summary"]["cells"] if (c["arm"], c["stratum"], c["evidence_type"]) ==
                ("pair", "attr", "rating") and not c["cross_stratum"])
    mine = next(r for r in refs if r["stratum"] == "attr" and r["evidence_type"] == "rating"
                and r["reference"] == "whole curve (c)")
    assert mine["T1"] == pytest.approx(cell["T1_cat_mean"], abs=1e-9)
    cf = contrast.share_counterfactual(a["units"], b["units"], res["summary"]["scale"])
    bare = next(r for r in cf if r["stratum"] == "name")
    assert bare["share_a"] < bare["share_cf"] < bare["share_b"] and 0 < bare["explained"] < 1


def test_families_that_grow_differently_are_told_apart(monkeypatch):
    """C0: the position bias grew less than the evidence response. Here B doubles every logit but its position
    bias is only 1.2 times A's."""
    a = analyze(World(position_bias=0.5))
    b = analyze(World(gamma=2.0, position_bias=0.3))
    rows = {(r["family"], r["stratum"], r["evidence_type"]): r for r in contrast.family_rows(contrast.family_logs(a, b))}
    assert rows[(contrast.POSITION, "attr", "all")]["multiplier"] == pytest.approx(1.2, rel=1e-6)
    assert rows[(contrast.FIRST, "attr", "rating")]["multiplier"] == pytest.approx(2.0, rel=1e-6)
    res = {"families": list(rows.values())}
    text = " ".join(f["text"] for f in scorecard._contrast_findings("A -> B", res, ""))
    assert "one multiplier does not describe this stage pair" in text and "option position" in text


# --------------------------------------------------------------------------- what the scorecard says
def _ref(reference, t1, lo, hi, stratum="name", etype="none"):
    return {"stratum": stratum, "evidence_type": etype, "reference": reference, "ref_stratum": "attr",
            "ref_evidence_type": "rating", "T1": t1, "lo": lo, "hi": hi, "n_categories": 11}


def test_a_content_change_is_claimed_only_against_every_reference():
    c0_like = [_ref("whole curve (c)", 0.12, 0.04, 0.20), _ref("first step", 0.02, -0.07, 0.11),
               _ref("fictional names", -0.18, -0.40, 0.04)]
    out = scorecard._reference_findings("A -> B", c0_like, "caveat")
    assert len(out) == 1 and "NOT robust to the reference" in out[0]["text"]
    assert "excludes 0 against 1 of 3 references" in out[0]["text"] and "No content change is claimed" in out[0]["action"]
    robust = [_ref("whole curve (c)", 0.22, 0.20, 0.24), _ref("first step", 0.25, 0.02, 0.48)]
    out = scorecard._reference_findings("A -> B", robust, "caveat")
    assert len(out) == 1 and "whichever reference is taken" in out[0]["text"]
    quiet = [_ref("whole curve (c)", 0.01, -0.05, 0.07), _ref("top step", -0.02, -0.09, 0.05)]
    assert "no departure from uniform sharpening" in scorecard._reference_findings("A -> B", quiet, "")[0]["text"]


def test_a_low_letter_mass_limits_the_conclusions_and_is_shown_to_move_with_scale():
    model = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rd = {"pass": False, "mean_c_forced": 0.675, "by_cell": [
        {"arm": "pair", "tier": "core", "family": "real", "stratum": "attr", "tied": True, "n_units": 300,
         "mean_c_forced": 0.5, "mean_c_free": 0.4, "share_low_coverage": 0.9},
        {"arm": "pair", "tier": "core", "family": "anchor", "stratum": "attr", "tied": False, "n_units": 100,
         "mean_c_forced": 0.95, "mean_c_free": 0.5, "share_low_coverage": 0.1}]}
    res = {"checks": {"thresholds": {"COVERAGE_MIN": 0.9}, "readout": rd}, "readout_ctx": {"mean_c_forced": 0.96}}
    models = {model: res, f"{model}@scale=2": {"checks": {"readout": {"pass": False, "mean_c_forced": 0.874}}}}
    found = scorecard.model_findings(model, res, "develop", scorecard.scaled_readout(model, models))
    text = found[0]["text"]
    assert found[0]["kind"] == "instrument" and "readout limited to the forced letter choice" in text
    assert "On tied items (both options show the same evidence) it is 0.5" in text and "difference 0.95" in text
    assert "With four options and the same question it is 0.96" in text
    assert "with every logit x 2 the same model reads 0.874" in text and "No preference level" not in text
    assert "cycle probe" in found[0]["action"]
    other = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    note = scorecard.readout_note(f"{model} -> {other}", {**models, other: {"checks": {"readout": {"pass": True}}}})
    assert "Conditional on the forced letter readout: letter mass 0.675 (Llama-3.1-Tulu-3-8B-SFT)" in note
    assert scorecard.readout_note(f"{other} -> {other}", {other: {"checks": {"readout": {"pass": True}}}}) == ""
    pair = {"summary": {"scale": [{"arm": "pair", "stratum": "attr", "evidence_type": "rating", "c": 2.2,
                                   "valid": True}], "cells": []}}
    results = [f for f in scorecard.pair_findings(f"{model} -> {other}", pair, "develop", note) if f["kind"] == "research"]
    assert results and all(note in f["action"] for f in results)
    row = scorecard.verdict_rows(models)[2]
    assert "LIMITED (letter mass 0.675)" in row and "FAIL (letter mass" not in row


def test_an_unresolved_first_step_is_reported_as_a_ratio_not_as_a_finer_step():
    cell = {"A_top": 4.02, "x_top": 2.0, "se_top": 0.1, "curve_rises": True, "pooled_monotone": True,
            "n_category_curves": 11, "n_flat_category_curves": 0, "A_first": 1.88, "x_first": 0.1, "A_second": 2.56,
            "x_second": 0.3, "jump_index": 0.602, "first_step_unresolved": True, "first_step_resolved": True,
            "share_in_first_step": 1.0, "share_censored": 0.0, "abs_delta_median": 0.176,
            "ratio_to_first_step_median": 0.094}
    out = scorecard._ladder_findings("SFT", "attr", "rating", cell, 0.1, "an instrument change", 0.5)
    assert len(out) == 1 and "does not resolve its first step" in out[0]["text"]
    assert "already 73% of the response at the second" in out[0]["text"] and "would give 33%" in out[0]["text"]
    assert "(median 0.094)" in out[0]["text"] and "Add a finer first step" not in out[0]["action"]
    graded = {**cell, "A_second": 5.0, "jump_index": 0.06, "first_step_unresolved": False}
    out = scorecard._ladder_findings("SFT", "attr", "rating", graded, 0.1, "an instrument change", 0.5)
    assert len(out) == 1 and "Add a finer first step" in out[0]["action"]


# --------------------------------------------------------------------------- the token probe
def test_probe_sample_is_balanced_over_the_kinds_of_item_and_keeps_both_orders():
    from rtp.measure import ladder_probe as lp

    items = [it for it in panel("mid").items if it["tier"] == "core"]
    picked = lp.sample_units(items, 3, seed=0)
    cells = pd.Series([lp.cell_of(it) for it in picked]).value_counts()
    assert set(cells.index) == {"fictional / attr / differs", "fictional / attr / tied", "fictional / name / tied",
                                "real brand / attr / tied", "real brand / name / tied"}
    assert (cells == 6).all()  # 3 units x 2 option orders
    orders = pd.DataFrame({"u": [it["unit_id"] for it in picked], "o": [it["order"] for it in picked]})
    assert (orders.groupby("u")["o"].apply(sorted).map(tuple) == (0, 1)).all()
    assert [it["item_id"] for it in lp.sample_units(items, 3, seed=0)] == [it["item_id"] for it in picked]


def test_probe_counts_a_declined_choice_apart_from_the_option_the_parser_would_credit():
    from rtp.measure import ladder_probe as lp
    from rtp.measure.parse import parse_answer

    opts = [{"label": "A", "name": "Trake"}, {"label": "B", "name": "Pavoprur"}]
    assert parse_answer("Both Trake and Pavoprur have the same rating", opts) == "A"  # the stage's parser
    assert lp.classify(" Both Trake and Pavoprur have the same rating", opts, "A") == ("declines", None)
    assert lp.classify(" Either option would work", opts, "B") == ("declines", None)
    assert lp.classify(" It depends on your budget", opts, "B") == ("declines", None)
    assert lp.classify(" B", opts, "B") == ("picked option", "B")
    assert lp.classify(" Pavoprur", opts, "A") == ("other option", "B")
    assert lp.classify(" Option B, because it is cheaper", opts, "B") == ("picked option", "B")
    assert lp.classify(" Hmm", opts, "A") == ("no option", None) and lp.classify(None, opts, "A") == ("no option", None)
    assert not lp.declines(" Bothell is a city")


@pytest.mark.integration
def test_ladder_probe_on_the_smoke_model(smoke_lm, work_dir, monkeypatch):
    from rtp import io
    from rtp.config import load_config
    from rtp.data.build import build_data
    from rtp.measure.ladder_probe import OUTCOMES, run_ladder_probe

    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    cfg = load_config("configs/smoke.yaml")
    build_data(cfg)
    out = run_ladder_probe(cfg, smoke_lm.model_id, n_units=2, top_k=8, max_new_tokens=6)
    res = io.read_json(out / "ladder_probe.json")
    df = pd.read_parquet(out / "items.parquet")
    assert res["all"]["n_items"] == len(df) and len(df) == 4 * len(res["cells"])
    assert 0 < res["all"]["letter_mass"] <= 1.0001 and 0 <= res["all"]["top1_letter"] <= 1
    assert sum(res["all"]["continuation"].values()) == pytest.approx(1.0) and set(res["all"]["continuation"]) == set(OUTCOMES)
    assert df["outcome"].isin(OUTCOMES).all() and df["gen_text"].notna().all()
    assert np.isclose(df["top1_letter"].mean(), res["all"]["top1_letter"])
    text = (out / "summary.md").read_text()
    assert "| kind of item (target / stratum / evidence) |" in text and "Answer:" in text
