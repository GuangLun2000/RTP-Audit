"""Pre-pilot fixes: name-answer diagnostics (A1), provenance (A2), balanced evidence-0 cell (A3), evidence-use
checks and uncertainty (A4)."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml

from rtp import io
from rtp.config import REPO_ROOT, config_from_dict
from rtp.data.build import build_data, generate
from rtp.measure.name_effect import (entity_evidence_table, evidence_sensitivity, items_meta, ladder_fits, ols,
                                     pair_covariates, responds_to_ratings, rotation_average)


def tiny(**data):
    base = {"categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
            "probe_entities_per_category": 1, "probe_templates": 1}
    base.update(data)
    return config_from_dict({"name": "pp", "seed": 5, "data": base})


# --------------------------------------------------------------------------- A3 balanced evidence-0 cell
def test_e0_crossed_with_every_substitute():
    out = generate(tiny(e0_pairs_per_substitute=2, n_placebo_per_category=4))
    e0 = [i for i in out["choice"] if i["evidence_level"] == 0 and not i["target_is_placebo"] and i["rotation"] == 0]
    by_entity = {}
    for it in e0:
        by_entity.setdefault(it["pair_entity"], Counter())[it["placebo_substitute"]] += 1
    assert all(len(c) == 4 and set(c.values()) == {2} for c in by_entity.values())
    for it in out["choice"]:
        dis = next(o["name"] for o in it["options"] if o["role"] == "placebo_distractor")
        assert dis != it["placebo_substitute"]
    # other levels keep cards_per_cell random pairs
    e1 = [i for i in out["choice"] if i["evidence_level"] == 1 and not i["target_is_placebo"] and i["rotation"] == 0]
    assert len(e1) == 3 * 1


def test_e0_default_follows_spec():
    out = generate(tiny())
    e0 = [i for i in out["choice"] if i["evidence_level"] == 0 and not i["target_is_placebo"] and i["rotation"] == 0]
    assert len(e0) == 3  # 3 targets x cards_per_cell 1


def test_placebos_never_start_with_option_letters_and_repair_is_minimal(tmp_path):
    from rtp.data.placebo import make_placebo_file, placebo_problem

    data = yaml.safe_load((REPO_ROOT / "data/seeds/placebo_names.yaml").read_text())
    names = [e["name"] for pool in ("choice", "probe") for v in data[pool].values() for e in v]
    assert not [n for n in names if n[0].upper() in "ABCD"]
    assert placebo_problem("Dregar", []) is not None and placebo_problem("Gores", []) is None
    # repair on a copy with one bad name: only that name changes, review flags elsewhere survive
    bad = copy.deepcopy(data)
    bad["choice"]["laptop"][0] = {"name": "Dorvex", "needs_web_check": True}
    bad["choice"]["laptop"][1]["needs_web_check"] = False
    path = tmp_path / "placebos.yaml"
    path.write_text(yaml.safe_dump(bad, sort_keys=False))
    make_placebo_file(str(REPO_ROOT / "data/seeds/categories.yaml"), str(REPO_ROOT / "data/seeds/natural_negatives.csv"),
                      str(path), repair=True)
    fixed = yaml.safe_load(path.read_text())
    assert fixed["choice"]["laptop"][0]["replaces"] == "Dorvex"
    assert fixed["choice"]["laptop"][1] == bad["choice"]["laptop"][1]
    changed = [(p, c, i) for p in ("choice", "probe") for c, v in fixed[p].items() for i, e in enumerate(v)
               if e != bad[p][c][i]]
    assert changed == [("choice", "laptop", 0)]


# --------------------------------------------------------------------------- A4 uncertainty and evidence use
def _ne_rows(entity, level, values, handicap=0):
    return [{"pair_id": f"{entity}-{level}-{handicap}-{k}", "category": "c", "entity": entity, "evidence_level": level,
             "handicap": handicap, "s_real": v, "s_placebo": 0.0, "name_effect": v} for k, v in enumerate(values)]


def test_entity_table_has_counts_ses_and_handicap0():
    ne = pd.DataFrame(_ne_rows("E", 0, [1.0, 2.0, 3.0]) + _ne_rows("E", 4, [0.0, 1.0], 0) + _ne_rows("E", 4, [5.0], 3))
    t = entity_evidence_table(ne).iloc[0]
    assert t["ne_e0"] == pytest.approx(2.0) and t["n_e0"] == 3 and t["se_e0"] == pytest.approx(1.0 / np.sqrt(3))
    assert t["ne_e4"] == pytest.approx(2.0) and t["ne_e4_h0"] == pytest.approx(0.5)


def test_ladder_fits_report_bootstrap_cis():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.5, 40)
    table = pd.DataFrame({"ne_e0": x, "ne_e1": x * 0.2, "ne_e4": 0.3 * x, "ne_e4_h0": 0.3 * x + rng.normal(0, .01, 40)})
    fits = {f["variant"]: f for f in ladder_fits(table, seed=0, n_boot=200)}
    spec = fits["spec: ne_e4 (all handicaps) ~ ne_e0"]
    assert spec["slope"] == pytest.approx(0.3, abs=1e-6) and spec["ci_low"] <= 0.3 <= spec["ci_high"]
    assert fits["format baseline: ne_e4 (handicap 0) ~ ne_e1"]["slope"] == pytest.approx(1.5, abs=0.05)


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    b, se = ols(X, X @ np.array([1.0, -2.0]) + rng.normal(0, 0.1, 500))
    assert np.allclose(b, [1.0, -2.0], atol=0.03) and (se < 0.01).all()


def test_evidence_sensitivity_detects_rating_response():
    out = generate(tiny(cards_per_cell=6, evidence_levels=[2, 3, 4]))
    items = out["choice"]
    cov = pair_covariates(items)
    assert (cov.dropna(subset=["rating_gap"])["rating_gap"] <= 1e-9).all()
    meta = items_meta(items)
    # synthetic model: s_target = 0.8 * rating_gap (+ noise); a model that ignores evidence: noise only
    rng = np.random.default_rng(2)
    gaps = cov.set_index(["pair_id", "target_is_placebo"])["rating_gap"].fillna(0.0)
    for slope, expect in ((0.8, True), (0.0, False)):
        rows = []
        for it in items:
            g = gaps[(it["pair_id"], it["target_is_placebo"])]
            rows.append({"item_id": it["item_id"], "s_target": slope * g + rng.normal(0, 0.01)})
        pairs = rotation_average(pd.DataFrame(rows), meta)
        sens = evidence_sensitivity(pairs, cov)
        assert responds_to_ratings(sens) is expect
        r = next(x for x in sens if x["covariate"] == "rating_gap" and x["version"] == "real")
        assert r["coef"] == pytest.approx(slope, abs=0.05)


# --------------------------------------------------------------------------- A2 provenance
def test_import_closure_covers_measure_dependencies():
    from rtp.provenance import code_digest, import_closure

    closure = import_closure("rtp.measure.choice")
    for mod in ("rtp.measure.choice", "rtp.measure.name_effect", "rtp.infer", "rtp.models", "rtp.registry",
                "rtp.data.build", "rtp.data.choice_items"):
        assert mod in closure
    assert "rtp.report" not in closure and "rtp.smoke" not in closure
    assert code_digest("rtp.measure.choice") != code_digest("rtp.measure.open_ended")


def test_stage_hash_follows_inputs_and_code():
    c = config_from_dict({"name": "x"})
    h = c.stage_hash("measure", "m", "dev", inputs="a", code="c1")
    assert h != c.stage_hash("measure", "m", "dev", inputs="b", code="c1")
    assert h != c.stage_hash("measure", "m", "dev", inputs="a", code="c2")


def test_probe_only_edits_do_not_invalidate_measure(work_dir):
    from rtp import stages

    a = tiny(probe_entities_per_category=1)
    build_data(a)
    h_a = stages.stage_hash(a, "measure", "m", "dev")
    b = tiny(probe_entities_per_category=2)  # same choice items, more probes
    build_data(b, force=True)
    assert stages.stage_hash(b, "measure", "m", "dev") == h_a


def test_resume_refused_under_other_environment(work_dir, monkeypatch):
    from rtp import stages

    sdir = work_dir / "stage"
    io.prepare_stage_dir(sdir, "x: 1\n", "h", extra_meta={"created_with": {"torch": "2.7.1", "device": "gpu-A"}})
    (sdir / "shards").mkdir()
    (sdir / "shards" / "shard_00000.done").write_text("1")
    monkeypatch.setattr(stages, "environment", lambda: {"torch": "2.7.1", "device": "gpu-B"})
    with pytest.raises(io.StageConfigMismatch, match="different environment"):
        stages.check_resume_environment(sdir)
    stages.check_resume_environment(sdir, force=True)  # --force restarts instead


def test_model_revision_is_recorded_and_enforced(work_dir):
    from rtp import stages

    class M:
        class config:  # noqa: N801
            _commit_hash = "abc"

    sdir = work_dir / "stage2"
    io.prepare_stage_dir(sdir, "x: 1\n", "h")
    stages.record_model_revision(sdir, M)
    assert io.read_json(sdir / io.STAGE_META)["model_commit"] == "abc"
    M.config._commit_hash = "def"  # nothing computed yet: the baseline simply moves to the new revision
    stages.record_model_revision(sdir, M)
    assert io.read_json(sdir / io.STAGE_META)["model_commit"] == "def"
    (sdir / "shards").mkdir()
    (sdir / "shards" / "shard_00000.done").write_text("1")
    M.config._commit_hash = "ghi"  # shards exist: a different revision would mix results
    with pytest.raises(io.StageConfigMismatch, match="revision"):
        stages.record_model_revision(sdir, M)


def test_resume_baseline_moves_until_first_shard_and_covers_extract_layout(work_dir, monkeypatch):
    from rtp import stages

    sdir = work_dir / "stage3"
    io.prepare_stage_dir(sdir, "x: 1\n", "h", extra_meta={"created_with": {"torch": "1", "device": "gpu-T4"}})
    monkeypatch.setattr(stages, "environment", lambda: {"torch": "1", "device": "gpu-H200"})
    stages.check_resume_environment(sdir)  # no shards: session 2 becomes the baseline
    assert io.read_json(sdir / io.STAGE_META)["created_with"]["device"] == "gpu-H200"
    (sdir / "probe").mkdir()
    (sdir / "probe" / "shard_00000.done").write_text("1")  # extract layout: <set>/shard_*.done
    monkeypatch.setattr(stages, "environment", lambda: {"torch": "1", "device": "gpu-RTX"})
    with pytest.raises(io.StageConfigMismatch, match="different environment"):
        stages.check_resume_environment(sdir)


def test_measure_digest_ignores_m3_data_generators_and_registry():
    from rtp.provenance import MODEL_STAGE_EXCLUDE_PREFIXES, PLUMBING, import_closure

    used = [m for m in import_closure("rtp.measure.choice")
            if m not in PLUMBING and not m.startswith(MODEL_STAGE_EXCLUDE_PREFIXES)]
    assert not [m for m in used if m.startswith(("rtp.detect", "rtp.data.", "rtp.registry", "rtp.stages",
                                                "rtp.report", "rtp.ops"))]
    assert "rtp.measure.choice" in used and "rtp.infer" in used


def test_free_answer_mapping():
    from rtp.measure.choice import parse_answer

    opts = [{"label": "A", "name": "Nike"}, {"label": "B", "name": "Bevodel"}, {"label": "C", "name": "Asics"},
            {"label": "D", "name": "Hoka"}]
    for text, want in [("B", "B"), ("Nike", "A"), ("I would go with Asics.", "C"), ("Bev", "B"), ("Hokas", "D"),
                       ("Based on the", None), ("Puma", None)]:
        assert parse_answer(text, opts) == want, text
