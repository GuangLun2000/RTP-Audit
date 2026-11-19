"""The estimators of rtp.scale on synthetic scores whose truth is known (tests/test_scale_helpers.py).

What must hold, in the order of the research objective:
* a pure change of scale multiplies name effects and anchor curves and leaves every preference level, every
  direct reading and every censoring flag exactly where it was, while the raw metrics of prior work move;
* an additive position bias of any size cancels;
* preference levels and scale multipliers recover their true values;
* the pooled null statistic is 0 under uniform sharpening, moves under a content change, and its intervals
  have roughly their nominal error rate; ``calibrate`` flags a true null rarely;
* a name-by-evidence interaction leaves the converted level unchanged and shows in the common-channel check.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rtp import scale
from rtp.scale.anchor import invert, isotonic_through_origin, pava
from rtp.scale.decompose import calibrate, null_tests, pair_sums, scale_multipliers, t_quantile, uncertainty
from rtp.scale.readout import ReadoutError, unit_table

from test_scale_helpers import World, panel, with_gamma


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def analyze(name: str, world: World):
    p = panel(name)
    return scale.analyze_model(p.items, p.score(world), p.steps)


# --------------------------------------------------------------------------- invariance
@pytest.mark.parametrize("gamma", [0.5, 1.74, 4.0])
def test_preference_levels_are_exactly_invariant_to_scale(gamma):
    world = World(noise=0.3, noise_seed=1, position_bias=0.4)
    a, b = analyze("mid", world), analyze("mid", with_gamma(world, gamma))
    ea, eb = a["entities"], b["entities"]
    for col in ("theta", "theta_level", "theta_direct", "theta_matched", "theta_cross_rating"):
        both = ea[col].notna()
        assert (both == eb[col].notna()).all(), col
        assert np.allclose(ea.loc[both, col], eb.loc[both, col], rtol=0, atol=1e-9), col
    for col in ("censored", "below_floor", "direct_censored", "status"):
        assert (ea[col].to_numpy() == eb[col].to_numpy()).all(), col
    assert np.allclose(eb["delta"], gamma * ea["delta"], rtol=1e-9)  # logits scale by gamma
    ca, cb = a["curves"], b["curves"]
    assert np.allclose(cb["A_raw"], gamma * ca["A_raw"], rtol=1e-9, equal_nan=True)
    assert np.allclose(b["fictional"]["theta"], a["fictional"]["theta"], atol=1e-9, equal_nan=True)


def test_raw_metrics_move_under_pure_scale_but_theta_does_not():
    world = World(noise=0.3, noise_seed=1)
    keys = ["arm", "stratum", "evidence_type", "metric", "d"]
    a = analyze("mid", world)["raw"].set_index(keys)["value"]
    b = analyze("mid", with_gamma(world, 2.25))["raw"].set_index(keys)["value"]
    share = ("pair", "attr", "rating", "choice_share_prob", 0)
    assert b[share] > a[share] + 0.02  # sharper model, bigger share of the favoured brands
    neff = ("pair", "attr", "rating", "n_eff", 0)
    assert b[neff] < a[neff] - 0.05  # more concentrated
    flips = [k for k in a.index if k[3] == "flip_rate_prob"]
    assert flips and any(abs(a[k] - b[k]) > 0.01 for k in flips)
    argmax = ("pair", "attr", "rating", "choice_share_argmax", 0)
    assert a[argmax] == pytest.approx(b[argmax])  # an exact rescaling keeps every item's argmax


def test_position_bias_of_any_size_cancels_in_the_unit_values():
    p = panel("mid")
    base = World(noise=0.2, noise_seed=3)
    l0 = p.unit_values(base)
    for bias in (0.5, 3.0, 12.0):
        assert np.allclose(p.unit_values(replace(base, position_bias=bias)), l0, atol=1e-9)
    units = unit_table(p.items, p.score(replace(base, position_bias=3.0)))
    two = units[units["n_options"] == 2]
    assert two["order_bias"].median() == pytest.approx(3.0, abs=0.2)  # reported, and taken out of L


# --------------------------------------------------------------------------- recovery
def test_theta_recovers_the_true_preference_level():
    p = panel("mid")
    world = World()
    ent = analyze("mid", world)["entities"]
    own = ent[(ent["arm"] == "pair") & ent["stratum"].isin(["attr", "full"]) & ent["theta"].notna()]
    assert len(own) > 50
    for row in own.itertuples():
        truth, censored = p.true_theta(world, row.category, row.entity, row.stratum, row.evidence_type)
        assert bool(row.censored) == censored
        if not censored:  # pulls of the controls sum to zero; the inverse is exact without noise
            assert row.theta == pytest.approx(truth, abs=1e-6), (row.category, row.entity, row.stratum, row.evidence_type)
    assert (own["theta"] < 0).any() and (own["censored"]).any()  # negative names and names beyond the ladder exist


def test_scale_multiplier_and_null_statistic_recover_their_truth():
    world = World(noise=0.2, noise_seed=2)
    a = analyze("mid", world)
    b = analyze("mid", replace(world, gamma=1.6, noise_seed=5))
    res = scale.compare_models(a, b, n_boot=300, seed=0)
    pooled = res["scale"][(res["scale"]["category"] == "(all)") & (res["scale"]["arm"] == "pair")]
    assert np.allclose(pooled["c"], 1.6, atol=0.06)
    null = res["null"][(res["null"]["arm"] == "pair") & ~res["null"]["cross_stratum"]]
    assert np.all(np.abs(null["T1"]) < 0.08)
    assert ((null["T1_lo"] <= 0) & (null["T1_hi"] >= 0)).mean() >= 0.8
    content = analyze("mid", replace(world, gamma=1.6, noise_seed=5, real_pull_mult=1.25))
    moved = scale.compare_models(a, content, n_boot=300, seed=0)["null"]
    moved = moved[(moved["arm"] == "pair") & (moved["stratum"] == "attr")]
    assert np.allclose(moved["T1"], np.log(1.25), atol=0.08)  # fictional names and evidence weights did not change
    assert (moved["T1_lo"] > 0).all()


def test_h1_detects_types_that_rescale_differently():
    """Common scale: B multiplies the rating ladder by 1.6 and the price ladder by 2.1 (evidence weights of
    one type changed). The spread of log c over the types must cover log(2.1 / 1.6)."""
    from test_scale_helpers import TOP_WEIGHT

    a = analyze("mid", World())
    original = dict(TOP_WEIGHT)
    try:
        TOP_WEIGHT["price"] = original["price"] * 2.1 / 1.6
        b = analyze("mid", World(gamma=1.6))
    finally:
        TOP_WEIGHT.clear()
        TOP_WEIGHT.update(original)
    h1 = scale.compare_models(a, b, n_boot=200, seed=0)["summary"]["H1"]["pair"]
    assert h1["log_c"]["price"] == pytest.approx(np.log(2.1), abs=1e-6)
    assert h1["range"] == pytest.approx(np.log(2.1 / 1.6), abs=1e-6)


# --------------------------------------------------------------------------- conditions
def test_name_by_evidence_interaction_shows_in_the_common_channel_check():
    clean = analyze("mid", World())["checks"]["common_channel"]
    assert clean["n_cells"] > 0 and clean["median_abs_diff_level"] == pytest.approx(0.0, abs=1e-6) and clean["pass"]
    world = World(real_slope=0.6)  # a real target feels only 60% of the adverse evidence
    res = analyze("mid", world)
    ref = analyze("mid", World())
    keys = ["arm", "category", "entity", "stratum", "evidence_type"]
    m = ref["entities"].merge(res["entities"], on=keys, suffixes=("_ref", "_int"))
    ok = m["theta_ref"].notna()
    assert np.allclose(m.loc[ok, "theta_ref"], m.loc[ok, "theta_int"], atol=1e-9)  # conversion: anchors unchanged
    d = m[m["theta_direct_ref"].notna() & m["theta_direct_int"].notna()]
    assert len(d) and ((d["theta_direct_int"] - d["theta_direct_ref"]).abs() > 1e-3).any()  # direct reading moves
    assert res["checks"]["common_channel"]["median_abs_diff_level"] > 0.05


def test_anchor_clean_check_catches_a_name_that_pulls_like_a_brand():
    p = panel("mid")
    assert analyze("mid", World())["checks"]["anchor_clean"]["pass"]
    category = "laptop"
    fict = [n for (c, n), f in zip(p.names, p.fictional) if f and c == category]
    rest = -2.0 / (len(fict) - 1)  # keep the fictional pulls of the category summing to zero
    world = World(pull_overrides={(category, n): (2.0 if i == 0 else rest) for i, n in enumerate(fict)})
    checks = analyze("mid", world)["checks"]["anchor_clean"]
    assert not checks["pass"] and checks["max_ratio"] >= 0.5


def test_readout_check_reports_coverage_and_mass_agreement():
    checks = analyze("mid", World(coverage=0.8, mass_fraction=0.3, mass_noise=0.05, noise=0.2))["checks"]["readout"]
    assert not checks["pass"] and checks["mean_c_forced"] == pytest.approx(0.8)
    assert checks["n_mass_units"] > 0 and checks["corr_L_mass"] > 0.9


def test_a_check_without_data_is_not_evaluated_rather_than_failed():
    p = panel("mid")
    core = [it for it in p.items if it["tier"] == "core"]  # no direct readings in the core tier
    scores = p.score(World())
    res = scale.analyze_model(core, scores[scores["item_id"].isin({it["item_id"] for it in core})], p.steps)
    assert res["checks"]["common_channel"]["n_cells"] == 0 and res["checks"]["common_channel"]["pass"] is None
    assert res["checks"]["readout"]["pass"] is True


def test_ladder_check_flags_censoring():
    world = World(real_mean=6.0, real_sd=0.3)  # every brand is worth more than the whole ladder
    ladder = analyze("mid", world)["checks"]["ladder"]
    assert not ladder["pass"] and ladder["share_censored"] > 0.5


# --------------------------------------------------------------------------- building blocks
def test_isotonic_fit_and_inverse_on_hand_made_curves():
    assert np.allclose(pava([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    assert np.allclose(pava([3, 1], [1, 3]), [1.5, 1.5])
    assert np.allclose(isotonic_through_origin([-0.5, 1.0, 0.8]), [0.0, 0.9, 0.9])
    x, a = [1.0, 2.0, 3.0], [1.0, 1.0, 3.0]  # flat between x = 1 and x = 2
    theta, cens = invert(x, a, [0.5, 1.0, 2.0, -2.0, 3.5, np.nan])
    assert np.allclose(theta[:4], [0.5, 1.0, 2.5, -2.5])  # the flat value maps to the LEFT end of its run
    assert theta[4] == 3.0 and cens.tolist() == [False, False, False, False, True, False]
    assert np.isnan(theta[5])
    with pytest.raises(ValueError):
        invert([1.0, 2.0], [2.0, 1.0], 1.0)  # not monotone: pass the isotonic fit


def test_t_quantile_and_calibrate():
    assert t_quantile(1) == pytest.approx(12.7062, abs=1e-4) and t_quantile(30) == pytest.approx(2.0423, abs=1e-4)
    assert t_quantile(60) == pytest.approx(2.0003, abs=2e-4) and t_quantile(1000) == pytest.approx(1.9623, abs=2e-4)
    res = calibrate(0.4, [0.0, 0.1, -0.1, 0.05, -0.05])
    assert res["outside"] and res["lo"] < 0 < res["hi"] and res["lo_2x"] < res["lo"]
    with pytest.raises(ValueError):
        calibrate(0.0, [0.1])
    rng = np.random.default_rng(0)  # a true null drawn from the replicates' own distribution
    flags = [calibrate(rng.normal(), rng.normal(size=5))["outside"] for _ in range(2000)]
    assert 0.02 <= np.mean(flags) <= 0.08


def test_null_interval_has_roughly_its_nominal_error_rate():
    """Many null stage pairs on many small categories: B is A sharpened by 1.5 with fresh item noise. The
    95% category-bootstrap interval of T1 must cover 0 in most of them (percentile intervals over 10
    categories are a little narrow; the band is loose on purpose)."""
    p = panel("sim")
    base = World(noise=0.6)
    a_runs = [p.fast(replace(base, noise_seed=100 + k)) for k in range(60)]
    covered, covered_t, t1 = [], [], []
    for k in range(60):
        a, b = a_runs[k], p.fast(replace(base, gamma=1.5, noise_seed=1000 + k))
        mult = scale_multipliers(a["curves"], b["curves"])
        tests, residuals = null_tests(a["entities"], b["entities"], mult)
        table, _ = uncertainty(pair_sums(mult, residuals), n_boot=300, seed=k)
        row = table[(table["arm"] == "pair") & (table["stratum"] == "attr") & (table["evidence_type"] == "rating")]
        covered.append(bool(row["T1_lo"].iloc[0] <= 0 <= row["T1_hi"].iloc[0]))
        covered_t.append(bool(row["T1_t_lo"].iloc[0] <= 0 <= row["T1_t_hi"].iloc[0]))
        t1.append(float(tests.loc[(tests["stratum"] == "attr") & (tests["evidence_type"] == "rating"), "T1"].iloc[0]))
    # 450 null intervals (2026-11-19): percentile bootstrap 0.913, t interval over categories 0.973. The t
    # interval is the primary one for confirmatory claims (DECISIONS.md); the bootstrap is reported beside it.
    assert np.mean(covered) >= 0.8
    assert np.mean(covered_t) >= 0.9
    assert abs(np.mean(t1)) < 3 * np.std(t1) / np.sqrt(len(t1)) + 0.01  # unbiased around 0


def test_unit_table_refuses_incomplete_or_foreign_scores():
    p = panel("smoke")
    scores = p.score(World())
    units = unit_table(p.items, scores)
    assert len(units) == len({it["unit_id"] for it in p.items})
    with pytest.raises(ReadoutError, match="lack an option order"):
        unit_table(p.items[1:], scores)
    with pytest.raises(ReadoutError, match="no score"):
        unit_table(p.items, scores.iloc[1:])
    two = pd.concat([scores, scores.assign(model_id="other", item_id=scores["item_id"] + "x")])
    with pytest.raises(ReadoutError, match="more than one model"):
        unit_table(p.items, two)


def test_canonical_form_rule_matches_the_gpu_stage():
    pytest.importorskip("torch")
    from rtp.measure.ladder import canonical_form as stage_rule
    from rtp.scale.readout import canonical_form as analysis_rule

    scores = panel("smoke").score(World())
    for frame in (scores, scores.assign(**{f"z_ns_{j}": scores[f"z_sp_{j}"] + 1.0 for j in range(4)})):
        assert stage_rule(frame) == analysis_rule(frame)
    assert analysis_rule(scores) == "sp"
