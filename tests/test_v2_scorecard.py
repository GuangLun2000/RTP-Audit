"""rtp.scorecard: what a cycle's numbers mean for the next cycle. The rules are exercised on finished ladder
stages written by the synthetic scorer (known truth), then on the feedback bundle that carries the scorecard."""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pandas as pd
import pytest

from rtp import analyze, cycle, io, ops, registry, scorecard
from rtp.data.build import build_data
from rtp.paths import run_dir
from rtp.scale import checks

import test_scale_helpers as helpers
from test_scale_helpers import Panel, World, make_cfg
from test_v2_analysis import write_stage

SMOKE = registry.SMOKE_MODEL
SCALED = f"{SMOKE}@scale=2"
OTHER = "allenai/Llama-3.1-Tulu-3-8B-DPO"  # only a label here: no model is loaded


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture()
def run(work_dir):
    cfg = make_cfg(["running shoe", "laptop", "camera", "backpack"], 5, 5, [0.3, 0.6, 1.0, 2.0], [0.15, 0.4, 1.0],
                   [5.0, 20.0, 100.0], n_base=1, direct_entities=2, direct_controls=3, favourable_levels=1)
    build_data(cfg)
    return cfg, Panel(cfg)


def card_of(cfg, models, pairs=(), jobs=(), phase="develop"):
    card = {"cycle": "t", "config": cfg.name, "phase": phase, "question": "Q?", "time": "now", "final": True,
            "instrument": {"ladder_version": 1, "data_hash": "d", "ladder": "l", "git_commit": "c" * 40},
            "elapsed_s": 60.0, "budget_s": 3600.0, "jobs": list(jobs), "decisions": [], "cut_order": [],
            "models": {m: analyze.analyze_model(cfg, m) for m in models},
            "pairs": {f"{a} -> {b}": analyze.compare(cfg, a, b) for a, b in pairs}}
    card["findings"] = scorecard.findings(card)
    return card


def texts(card, kind):
    return [f["text"] + " || " + f["action"] for f in card["findings"] if f["kind"] == kind]


def test_a_flat_curve_and_a_short_ladder_become_instrument_findings_with_the_config_key(run, monkeypatch):
    cfg, panel = run
    monkeypatch.setitem(helpers.TOP_WEIGHT, "price", 0.0)    # price evidence does not move the model
    monkeypatch.setitem(helpers.TOP_WEIGHT, "reviews", 0.4)  # the review ladder is far too short for the names
    write_stage(cfg, panel, SMOKE, "core", World(real_sd=1.5))
    card = card_of(cfg, [SMOKE])
    found = texts(card, "instrument")
    flat = [t for t in found if t.startswith("attr/price")]
    assert len(flat) == 1 and "does not rise" in flat[0] and "longer ladder does not cure" in flat[0]
    short = [t for t in found if t.startswith("attr/reviews")]
    assert short and "worth more than the whole ladder" in short[0] and "`ladder.review_ratios`" in short[0]
    assert not any(t.startswith("attr/rating") and "whole ladder" in t for t in found)
    assert any("common channel" in t and "`ext` tier" in t for t in texts(card, "pending"))  # core only: not a failure
    md = scorecard.scorecard_markdown(card)
    for needle in ("## Findings", "### Instrument (the ruler)", "ladder.version", "**The ruler**", "100x fewer reviews",
                   "**Brands on the ruler**", "**Fictional names with the largest pull**", "not evaluated"):
        assert needle in md, needle
    ladders = card["models"][SMOKE]["ladders"]
    rating = [r for r in ladders if (r["stratum"], r["evidence_type"]) == ("attr", "rating")]
    assert [r["x"] for r in rating] == [0.3, 0.6, 1.0, 2.0] and all(r["A_fit"] > 0 for r in rating)
    cell = card["models"][SMOKE]["headline"]["attr/rating"]
    assert sum(r["share_of_brands"] for r in rating) + cell["share_censored"] == pytest.approx(1.0, abs=1e-3)


def test_a_first_step_that_is_too_coarse_is_reported_for_typed_ladders_only(run):
    cfg, panel = run
    write_stage(cfg, panel, SMOKE, "core", World(real_mean=0.05, real_sd=0.05))  # every brand is worth very little
    card = card_of(cfg, [SMOKE])
    coarse = [t for t in texts(card, "instrument") if "inside the first step" in t]
    assert any("`ladder.rating_deltas`" in t for t in coarse) and not any(t.startswith("comp/") for t in coarse)


def test_pure_scale_pair_is_a_wiring_check_and_its_violation_is_a_bug(run):
    cfg, panel = run
    world = World(noise=0.2, noise_seed=1)
    write_stage(cfg, panel, SMOKE, "core", world)
    write_stage(cfg, panel, SCALED, "core", replace(world, gamma=2.0))
    card = card_of(cfg, [SMOKE, SCALED], [(SMOKE, SCALED)])
    ok = [t for t in texts(card, "research") if "pure-scale check" in t]
    assert len(ok) == 1 and "invariant to scale" in ok[0] and not texts(card, "bug")
    # the variant repeats its base model: no second copy of the model findings, no second set of tables
    assert not any(f["scope"] == scorecard.short(SCALED) for f in card["findings"])
    assert scorecard.scorecard_markdown(card).count("**The ruler**") == 1
    # a "variant" whose content moved too: the identity fails and nothing else may be read before it is fixed
    write_stage(cfg, panel, SCALED, "core", replace(world, gamma=2.0, real_pull_mult=1.3), note="content moved too")
    bad = card_of(cfg, [SMOKE, SCALED], [(SMOKE, SCALED)])
    bugs = texts(bad, "bug")
    assert len(bugs) == 1 and "MUST leave every level unchanged" in bugs[0] and bad["findings"][0]["kind"] == "bug"


def test_stage_pair_findings_state_scale_content_and_what_is_still_unknown(run):
    cfg, panel = run
    world = World(noise=0.2, noise_seed=1)
    write_stage(cfg, panel, SMOKE, "core", world)
    write_stage(cfg, panel, OTHER, "core", replace(world, gamma=1.7, real_pull_mult=1.3, noise_seed=2))
    card = card_of(cfg, [SMOKE, OTHER], [(SMOKE, OTHER)])
    found = texts(card, "research")
    scale = next(t for t in found if t.startswith("scale multiplier c"))
    assert all(f"{t} 1." in scale for t in ("rating", "price", "reviews")) and "Not a change of preference" in scale
    h1 = next(t for t in found if t.startswith("H1"))
    assert "judged against the seed replicates" in h1
    pair = card["pairs"][f"{SMOKE} -> {OTHER}"]
    bare = {(r["reference"], r["ref_evidence_type"]): r for r in pair["references"] if r["stratum"] == "name"}
    assert bare[("whole curve (c)", "rating")]["T1"] == pytest.approx(0.26, abs=0.1)  # truth: log 1.3 = 0.26
    assert bare[("whole curve (c)", "rating")]["lo"] > 0 and len(bare) == 10  # 3 curves x 3 references + the names
    moved = next(t for t in found if "name/none: from" in t)  # reported against every reference, never against one
    assert "against the" in moved and "T1 = log multiplier of the real names" in moved
    assert "run-to-run training noise is unknown" in moved and "Exploratory" in moved
    families = next(t for t in found if t.startswith("multipliers by contrast family"))
    assert "real brand names" in families and "evidence, first step 1." in families
    share = next(t for t in found if t.startswith("raw choice share"))
    assert "name/none" in share and "of the change" in share
    md = scorecard.scorecard_markdown(card)
    assert "**Shape of the ruler**" in md and "**Scale-dependent metrics, A -> B**" in md
    assert "| T1 pooled | T1 by category |" in md and "**T1 against every reference**" in md
    row = next(r for r in card["pairs"][f"{SMOKE} -> {OTHER}"]["raw"] if r["stratum"] == "name")
    assert row["choice_share_prob_b"] > row["choice_share_prob_a"]  # the raw metric rises with scale and content
    # the same pair with no content change: the name effects follow the scale factor
    write_stage(cfg, panel, OTHER, "core", replace(world, gamma=1.7, noise_seed=2), note="scale only")
    quiet = texts(card_of(cfg, [SMOKE, OTHER], [(SMOKE, OTHER)]), "research")
    assert any("no departure from uniform sharpening against any reference" in t and "attr/rating" in t for t in quiet)
    assert not any("whichever reference is taken" in t for t in quiet)  # no content change is claimed


def test_anchor_check_ignores_ratios_that_are_only_bounds():
    """Censored real brands enter at the top of the ladder, which squeezes their spread and inflates the SD
    ratio. Such a row must not fail the check: the remedy is a longer ladder, not other names."""
    def frame(rows):  # the cross-stratum columns exist in every real table; no name-stratum row is needed here
        df = pd.DataFrame(rows)
        for t in ("rating", "price", "reviews"):
            df[f"theta_cross_{t}"], df[f"censored_cross_{t}"] = float("nan"), False
        return df

    fict = frame([{"arm": "pair", "category": c, "name": f"f{i}", "stratum": s, "evidence_type": "rating", "pull": p,
                   "pull_se": 0.01, "n": 4, "identified": True, "theta": p, "theta_level": p, "censored": False,
                   "below_floor": False, "status": "ok"}
                  for s in ("attr", "full") for c in ("a", "b") for i, p in enumerate((-0.2, 0.0, 0.2))])
    real = frame([{"arm": "pair", "split": "dev", "category": c, "entity": f"r{i}", "stratum": s, "evidence_type": "rating",
                   "delta": d, "theta": min(d, top), "theta_level": min(d, top), "censored": d > top,
                   "below_floor": False, "status": "censored" if d > top else "ok"}
                  for s, top in (("attr", 9.0), ("full", 1.0)) for c in ("a", "b") for i, d in enumerate((0.5, 1.5, 2.5, 3.5))])
    out = checks._anchor_clean(fict, real)
    rows = {r["stratum"]: r for r in out["rows"]}
    assert rows["attr"]["bound_only"] is False and rows["full"]["bound_only"] is True
    assert rows["full"]["ratio"] > checks.ANCHOR_SD_RATIO_MAX > rows["attr"]["ratio"]  # the squeezed row alone would fail
    assert out["pass"] is True and out["n_rows_bound_only"] == 1 and out["max_ratio"] == pytest.approx(rows["attr"]["ratio"])
    only = checks._anchor_clean(fict[fict["stratum"] == "full"], real[real["stratum"] == "full"])
    assert only["pass"] is None and only["max_ratio"] is None and only["max_ratio_bound_only"] > 0.5
    calm = fict[fict["stratum"] == "full"].assign(theta=lambda d: d["theta"] / 10.0)  # even the bound is below the limit
    assert checks._anchor_clean(calm, real[real["stratum"] == "full"])["pass"] is True
    res = {"checks": {"anchor_clean": {k: v for k, v in only.items() if k != "rows"}, "thresholds": {}}}
    pending = [f for f in scorecard.model_findings("m", res) if f["kind"] == "pending" and "fictional anchors" in f["text"]]
    assert pending and "Extend the ladder first" in pending[0]["action"]


def test_formatting_helpers():
    assert scorecard.step_label("rating", 0.3) == "0.3 stars lower" and scorecard.step_label("price", 0.15) == "15% dearer"
    assert scorecard.step_label("reviews", 3.912) == "50x fewer reviews"
    rows = [{"a": 1, "b": None}, {"a": 2, "b": None}]
    assert scorecard.table(rows, [("A", "a"), ("B", "b")])[0] == "| A |"  # a column that is empty everywhere is dropped
    assert scorecard.scale_variant("m", "m@scale=2") == 2.0 and scorecard.scale_variant("m", "n@scale=2") is None
    assert scorecard.variant_of("m@scale=2", {"m": {}, "m@scale=2": {}}) == "m"
    assert scorecard.variant_of("m@scale=2", {"m@scale=2": {}}) is None
    assert scorecard.fmt_time(28) == "28 s" and scorecard.fmt_time(600) == "10 min" and scorecard.fmt_time(5400) == "1.5 h"
    done = [{"job": "j", "state": "done", "predicted_s": 100.0, "actual_s": 60.0},
            {"job": "k", "state": "failed (exit 1)", "predicted_s": 50.0, "actual_s": 5.0}]
    budget = [f["text"] for f in scorecard.budget_findings({"jobs": done})]
    assert "1 of 2 jobs are not done" in budget[0] and "actual / predicted = 0.6" in budget[1]


def test_feedback_bundle_carries_the_newest_scorecard_whole(run):
    cfg, panel = run
    world = World(noise=0.2, noise_seed=1)
    for tier in ("core", "ext"):
        write_stage(cfg, panel, SMOKE, tier, world)
        write_stage(cfg, panel, OTHER, tier, replace(world, gamma=1.7, noise_seed=2))
    card = card_of(cfg, [SMOKE, OTHER], [(SMOKE, OTHER)],
                   jobs=[{"job": "ladder-core m", "state": "done", "predicted_s": 600.0, "actual_s": 400.0}])
    cdir = run_dir(cfg.name) / "cycles" / "t"
    io.write_json(cdir / "scorecard.json", card)
    io.atomic_write_text(cdir / "scorecard.md", scorecard.scorecard_markdown(card))
    log_path = run_dir(cfg.name) / "events.jsonl"  # two commands that started and wrote no end record
    log_path.write_text("".join(json.dumps({"time": "t", "kind": "start", "message": " ".join(argv), "data": {"argv": argv}})
                                + "\n" for argv in (["ladder", "--tier", "ext"], ["feedback", "--config", "x"])))
    probe = run_dir(cfg.name) / "ladder_probe" / "m"  # the token probe's tables travel in the bundle too
    probe.mkdir(parents=True)
    (probe / "summary.md").write_text("# Ladder probe: m (tier core)\n\n| kind of item | items |\n")
    block = {"n_items": 80, "letter_mass": 0.62, "top1_letter": 0.7, "n_top1_not_letter": 24,
             "continuation": {"picked option": 0.8, "declines": 0.2},
             "continuation_when_top1_not_letter": {"picked option": 0.4, "declines": 0.6},
             "other_tokens": [{"token": "' Both'", "mean_prob": 0.21, "share_top1": 0.3}]}
    io.write_json(probe / "ladder_probe.json", {
        "model_id": "m", "tier": "core", "all": block, "cells": {"real brand / attr / tied": block},
        "examples": [{"cell": "real brand / attr / tied", "letter_mass": 0.31, "first_tokens": [["' Both'", 0.55]],
                      "outcome": "declines", "text": " Both options have the same rating"}]})
    bundle = ops.build_feedback(cfg).read_text()
    assert "truncated" not in bundle.split("## Stage status")[0]
    for needle in ("### t (newest)", "## Findings", "## Stage pairs", "**Shape of the ruler**", "## Jobs (budget ledger)",
                   "instrument finding(s)", "section Findings of the newest scorecard",
                   "**Multipliers by contrast family**", "**T1 against every reference**",
                   "**Raw choice share and uniform sharpening**", "**Letter mass by kind of item**",
                   "**Fictional anchors per cell**", "## Token probe", "# Ladder probe: m (tier core)"):
        assert needle in bundle, needle
    # the digest: what the notebook prints for copy and paste. Everything a cycle is judged by, far shorter
    digest = sorted((run_dir(cfg.name) / "feedback").glob("digest_*.md"))[-1].read_text()
    assert digest.startswith(ops.DIGEST_BEGIN) and digest.rstrip().endswith("characters) =====")
    body = digest[:digest.rindex(ops.DIGEST_END)].rstrip("\n")
    assert f"({len(body)} characters)" in digest.splitlines()[-1]  # a cut paste is noticed
    for needle in ("ENV commit", "ERRORS: 0 unresolved", "NEVER FINISHED: `ladder --tier ext`", "PYTEST:",
                   "## Findings", "- [research]", "## Checks", "letter mass by kind of item",
                   "ruler (x = step in natural units", "attr/rating x=[0.3, 0.6, 1, 2] A=[", "jump=", "brands (real vs",
                   "fictional anchors per cell", "c (anchor curves): attr/price", "multipliers by contrast family",
                   "T1 of the real names against every reference", "raw choice share and uniform sharpening",
                   "## Stage records", "ladder_ext: n_items", "## Token probe", "' Both' 0.210 (30%)",
                   "lowest letter mass, real brand / attr / tied: 0.31 | ' Both' 0.55 | declines |"):
        assert needle in digest, needle
    assert "Next:" not in digest and len(digest) < 0.75 * len(bundle) and len(digest) < 40000
    unfinished = bundle.split("## Commands that never finished")[1].split("\n## ")[0]
    assert "ladder --tier ext" in unfinished and "feedback" not in unfinished  # the command writing the bundle is no casualty
    assert "## Analysis summaries" not in bundle  # both models and the pair are already in the scorecard


def test_a_cycle_run_again_after_an_instrument_change_is_compared_with_its_earlier_scorecard(run):
    """Re-running a cycle overwrites its scorecard. The one measured with the earlier instrument is kept, and
    "what changed" compares with it: this is how a change of the instrument is judged."""
    cfg, panel = run
    spec = cycle.CycleSpec(name="t", config="unused", phase="develop", question="Q?", budget_hours=1.0, jobs=[])
    cdir = cycle.cycle_dir(cfg, spec)
    write_stage(cfg, panel, SMOKE, "core", World(real_sd=1.5))
    old = card_of(cfg, [SMOKE])
    io.write_json(cdir / "scorecard.json", old)
    io.atomic_write_text(cdir / "scorecard.md", scorecard.scorecard_markdown(old))
    cycle._archive_scorecard(cdir, old)  # the same instrument (a later job of the same session): nothing to keep
    assert not (cdir / "history").exists() and cycle.previous_scorecard(cfg, spec) is None
    write_stage(cfg, panel, SMOKE, "core", World(real_sd=0.4), note="measured with the next instrument")
    new = card_of(cfg, [SMOKE])
    new["instrument"] = {**new["instrument"], "ladder_version": 2, "data_hash": "e" * 16}
    cycle._archive_scorecard(cdir, new)
    kept = sorted(p.name for p in (cdir / "history").iterdir())
    assert kept == ["scorecard_v1_d_l.json", "scorecard_v1_d_l.md"]
    prev = cycle.previous_scorecard(cfg, spec)
    assert prev["instrument"]["ladder_version"] == 1
    assert cycle._card_label(prev, spec).startswith("t as measured with ladder version 1")
    moved = {c["key"] for c in cycle.diff_cards(prev, new)}
    assert f"models/{SMOKE}/headline/name/none/abs_delta_median" in moved  # the brands' name effects shrank
    # a scorecard of another cycle is the reference only when this cycle has no earlier version of its own
    other = cycle.CycleSpec(name="u", config="unused", phase="develop", question="Q?", budget_hours=1.0, jobs=[])
    assert cycle.previous_scorecard(cfg, other)["cycle"] == "t" and cycle._card_label(prev, other) == "t"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_noise_curve_is_not_reported_as_a_ladder_that_is_too_short(run, monkeypatch, seed):
    """Evidence with no true effect plus item noise: the monotone fit of the noise is usually positive, and its
    brands then look 'beyond the ladder'. The finding must say the curve does not rise, never 'add larger steps'."""
    cfg, panel = run
    monkeypatch.setitem(helpers.TOP_WEIGHT, "price", 0.0)
    write_stage(cfg, panel, SMOKE, "core", World(noise=0.3, noise_seed=seed))
    price = [t for t in texts(card_of(cfg, [SMOKE]), "instrument") if t.startswith("attr/price")]
    assert price and all("does not rise beyond its noise" in t for t in price)
    assert not any("ladder.price_deltas" in t for t in price)
