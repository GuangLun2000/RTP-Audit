"""rtp.analyze: the view over finished ladder stages (caching, shared tiers, the scale identity, the
four-option comparison, flat anchor curves). The ladder stages are written by the synthetic scorer of
tests/test_scale_helpers.py, so no model is loaded and every number has a known truth."""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd
import pytest

from rtp import analyze, io, registry, stages
from rtp.data.build import build_data, load_ladder_items
from rtp.paths import stage_dir

import test_scale_helpers as helpers
from test_scale_helpers import Panel, World, make_cfg

SMOKE = registry.SMOKE_MODEL
SCALED = f"{SMOKE}@scale=2"


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture()
def run(work_dir):
    cfg = make_cfg(["running shoe", "laptop", "camera"], 5, 5, [0.3, 0.6, 1.0, 2.0], [0.15, 0.4, 1.0],
                   [5.0, 20.0, 100.0], n_base=1, direct_entities=2, direct_controls=3, favourable_levels=1)
    build_data(cfg)
    return cfg, Panel(cfg)


def write_stage(cfg, panel, model, tier, world, note="first"):
    """A finished, current ladder stage of ``model`` holding the synthetic scores of ``world``."""
    items = load_ladder_items(cfg, tier, "dev")
    ids = {it["item_id"] for it in items}
    scores = panel.score(world, model_id=model)
    scores = scores[scores["item_id"].isin(ids)].reset_index(drop=True)
    sdir = stage_dir(cfg.name, model, f"ladder_{tier}")
    shash = stages.stage_hash(cfg, "ladder", model, "dev", tier=tier)
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=True,
                         extra_meta={**stages.stage_meta("ladder", model, "dev", "auto"), "tier": tier})
    io.write_parquet(sdir / "scores.parquet", scores)
    io.mark_done(sdir, {"n_items": len(scores), "note": note})
    return sdir


def test_model_analysis_is_cached_and_follows_its_inputs(run):
    cfg, panel = run
    assert analyze.analyze_model(cfg, SMOKE) is None  # no finished core tier yet
    write_stage(cfg, panel, SMOKE, "core", World(noise=0.2))
    first = analyze.analyze_model(cfg, SMOKE)
    assert first["tiers"] == ["core"] and first["letter_form"] == "sp"
    assert first["checks"]["readout"]["pass"] is True and first["checks"]["common_channel"]["pass"] is None
    assert first["headline"]["attr/rating"]["curve_rises"] and first["headline"]["attr/rating"]["theta_median"] is not None
    out_dir = analyze.analysis_dir(cfg) / analyze.model_slug(SMOKE)
    stamp = (out_dir / "summary.json").stat().st_mtime_ns
    assert analyze.analyze_model(cfg, SMOKE) == first and (out_dir / "summary.json").stat().st_mtime_ns == stamp
    # the stage is measured again under the SAME identity (--force): other scores, same stage hash
    write_stage(cfg, panel, SMOKE, "core", World(noise=0.2, real_pull_mult=2.0), note="re-measured")
    again = analyze.analyze_model(cfg, SMOKE)
    assert again["headline"]["attr/rating"]["delta_median"] > 1.5 * first["headline"]["attr/rating"]["delta_median"]
    # a new tier is picked up; a stale one is left out
    write_stage(cfg, panel, SMOKE, "ext", World(noise=0.2, real_pull_mult=2.0))
    assert analyze.analyze_model(cfg, SMOKE)["tiers"] == ["core", "ext"]
    meta_path = stage_dir(cfg.name, SMOKE, "ladder_ext") / io.STAGE_META
    io.write_json(meta_path, {**io.read_json(meta_path), "stage_hash": "built-by-other-code"})
    assert analyze.analyze_model(cfg, SMOKE)["tiers"] == ["core"]


def test_analysis_code_is_part_of_the_cache_key(run, monkeypatch):
    cfg, panel = run
    write_stage(cfg, panel, SMOKE, "core", World())
    analyze.analyze_model(cfg, SMOKE)
    path = analyze.analysis_dir(cfg) / analyze.model_slug(SMOKE) / "summary.json"
    stamp = path.stat().st_mtime_ns
    monkeypatch.setattr(analyze, "_code_digest", lambda: "edited-analysis-code")
    analyze.analyze_model(cfg, SMOKE)
    assert path.stat().st_mtime_ns != stamp  # an edit to rtp/analyze.py or rtp/scale recomputes the view
    from rtp.provenance import import_closure

    closure = set(import_closure("rtp.analyze"))
    assert {"rtp.analyze", "rtp.scale", "rtp.scale.theta", "rtp.scale.decompose", "rtp.scale.checks"} <= closure


def test_pair_uses_the_tiers_both_models_finished(run):
    """Cutting 'ext on the second model' (the first cut of cycle C1) must not lose the comparison."""
    cfg, panel = run
    world = World(noise=0.2, noise_seed=1)
    for tier in ("core", "ext"):
        write_stage(cfg, panel, SMOKE, tier, world)
    write_stage(cfg, panel, SCALED, "core", replace(world, gamma=2.0))
    res = analyze.compare(cfg, SMOKE, SCALED)
    assert res["tiers_compared"] == ["core"] and res["tiers_left_out"] == {SMOKE: ["ext"]}
    ident = res["identity"]
    assert ident["n_curves"] >= 3 and ident["n_brand_cells"] > 0 and ident["n_flag_changes"] == 0
    assert ident["c_min"] == pytest.approx(2.0, abs=1e-9) and ident["c_max"] == pytest.approx(2.0, abs=1e-9)
    assert ident["max_abs_theta_diff"] < 1e-9
    cells = {(c["stratum"], c["evidence_type"]): c for c in res["summary"]["cells"] if not c["cross_stratum"]}
    row = cells[("attr", "rating")]
    assert row["T1"] == pytest.approx(0.0, abs=1e-9) and "T1_cat_mean" in row and row["n_cat_T1"] == 3
    # with both tiers on both models the stored tables are used and nothing is left out
    write_stage(cfg, panel, SCALED, "ext", replace(world, gamma=2.0))
    full = analyze.compare(cfg, SMOKE, SCALED)
    assert full["tiers_compared"] == ["core", "ext"] and full["tiers_left_out"] == {}
    assert full["identity"]["c_max"] == pytest.approx(2.0, abs=1e-9)
    assert analyze.compare(cfg, SMOKE, "allenai/Llama-3.1-Tulu-3-8B-SFT") is None  # not measured: no pair yet


def test_pair_refuses_two_letter_forms(run):
    cfg, panel = run
    write_stage(cfg, panel, SMOKE, "core", World())
    sdir = write_stage(cfg, panel, SCALED, "core", World(gamma=2.0))
    scores = pd.read_parquet(sdir / "scores.parquet")
    for j in range(4):  # in this model the bare letter "A" is the likelier form
        scores[f"z_ns_{j}"] = scores[f"z_sp_{j}"] + 5.0
    io.write_parquet(sdir / "scores.parquet", scores)
    io.mark_done(sdir, {"note": "other form"})
    with pytest.raises(ValueError, match="letter form"):
        analyze.compare(cfg, SMOKE, SCALED)


def test_choice_set_comparison_is_centred_on_the_same_zero(run, monkeypatch):
    """A model whose name effects ignore the choice set must show no difference between two and four
    options, although the four-option arm measures each brand against only some of the fictional names."""
    cfg, panel = run
    monkeypatch.setattr(helpers, "CTX_MULT", 1.0)
    fresh = Panel(cfg)  # name multipliers are fixed when the panel is built
    world = World(fict_sd=0.6)  # fictional names with sizeable pulls of their own
    for tier in ("core", "ctx"):
        write_stage(cfg, fresh, SMOKE, tier, world)
    agree = analyze.analyze_model(cfg, SMOKE)["choice_set"]
    for cell in ("name/none", "attr/rating"):
        assert agree[cell]["n_brands"] == 15 and agree[cell]["n_brands_without_pull"] == 0
        assert agree[cell]["delta_median_diff_4_minus_2"] == pytest.approx(0.0, abs=1e-6)
        assert agree[cell]["delta_corr"] == pytest.approx(1.0, abs=1e-6)
        assert agree[cell]["mean_abs_pull_added"] > 0.01  # the offset that was removed is not small
    assert agree["attr/rating"]["theta_median_diff_4_minus_2"] == pytest.approx(0.0, abs=1e-6)


def test_a_flat_anchor_curve_is_reported_as_flat_not_as_censored(run, monkeypatch):
    """Evidence the model does not respond to (here: price) gives no preference level. Reporting every brand
    as 'beyond the top of the ladder' would send the next cycle off to lengthen a ladder that does not work."""
    cfg, panel = run
    monkeypatch.setitem(helpers.TOP_WEIGHT, "price", 0.0)
    write_stage(cfg, panel, SMOKE, "core", World())
    res = analyze.analyze_model(cfg, SMOKE)
    price, rating = res["headline"]["attr/price"], res["headline"]["attr/rating"]
    assert price["curve_rises"] is False and price["n_flat_category_curves"] == 3
    assert price["share_on_flat_curve"] == 1.0 and "theta_median" not in price
    assert rating["curve_rises"] is True and rating["share_on_flat_curve"] == 0.0
    ladder = res["checks"]["ladder"]
    assert ladder["n_curves_flat"] == 3 and ladder["dropped"]["flat_curve"] == 15
    by = {(r["stratum"], r["evidence_type"]): r for r in ladder["by_cell"]}
    assert by[("attr", "price")]["n_entities"] == 0 and by[("attr", "price")]["n_entities_on_flat_curves"] == 15
    ent = pd.read_parquet(analyze.analysis_dir(cfg) / analyze.model_slug(SMOKE) / "entities.parquet")
    flat = ent[(ent["stratum"] == "attr") & (ent["evidence_type"] == "price")]
    assert (flat["status"] == "flat_curve").all() and flat["theta"].isna().all() and not flat["censored"].any()
