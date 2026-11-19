"""Instrument v2: ladder items, model variants, the letter-logit readout, the ladder and open-set stages."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict

import numpy as np
import pytest

from rtp import io, registry, variants
from rtp.config import ConfigError, config_from_dict, load_config
from rtp.data.build import build_data, generate, load_ladder_items, panel_stats
from rtp.data.ladder_items import all_pairs, core_pairs, type_steps
from rtp.data.ladder_schema import validate_ladder_items
from rtp.data.schema import SchemaError
from rtp.data.splits import SealedSplitError
from rtp.io import dumps_record
from rtp.paths import model_slug


def mid_cfg(**ladder):
    raw = {"name": "mid", "seed": 3, "data": {
        "categories": ["running shoe", "laptop", "beer"], "test_categories": ["beer"],
        "n_real_per_category": 5, "n_placebo_per_category": 6, "cards_per_cell": 1, "evidence_levels": [0],
        "handicaps": [0], "probe_entities_per_category": 2, "probe_templates": 2},
        "ladder": {"enabled": True, "n_base": 2, "direct_entities": 2, "direct_controls": 3, **ladder}}
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def mid():
    logging.disable(logging.WARNING)
    try:
        cfg = mid_cfg()
        return cfg, generate(cfg)
    finally:
        logging.disable(logging.NOTSET)


def _sha(records) -> str:
    return hashlib.sha256("".join(dumps_record(r) + "\n" for r in records).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- legacy items are untouched
GOLDEN = {  # sha256 of the generated legacy records at the commit before instrument v2 (cfcba27)
    ("configs/pilot.yaml", "choice", "dev"): "bd75cb964f9fac86f7a09c67ad06d890465c9b538e77c4ac6fea2a37cfdd472a",
    ("configs/pilot.yaml", "choice", "test"): "2001291a8054ce50d811d3a1bbcde023f3703b8b8eec25e4a107fb6c83e2eb62",
    ("configs/pilot.yaml", "probe", "dev"): "ce1a8e3caf85aa48e0dda29917f167fa9a6b04b8e5a61b38ce6fe39d8ffadaea",
    ("configs/pilot.yaml", "probe", "test"): "f7c28755139e8e01d08aa1f488b12baa10955548c867bc76bdaae07633ad3ca4",
    ("configs/smoke.yaml", "choice", "dev"): "d63085a895b2115c3e53d4d2be61a9bdf12a0d0325d53d5ef88ef4f71ddb0225",
    ("configs/smoke.yaml", "probe", "dev"): "e07b3d96b4d036ca598d8776dcb130a98129865ca54016fd06c1ee2c91b08a27",
}
GOLDEN_CATALOGUE = {"configs/pilot.yaml": "72fd073546070deb03387913c64a3bfc39f5bca1e3346bbbf2963258ac76034f",
                    "configs/smoke.yaml": "efd08e4b56e4cff057950ed1b6f62193a5e2e80df11f8e9c1892c9dea3e2da24"}


@pytest.mark.parametrize("path", ["configs/pilot.yaml", "configs/smoke.yaml"])
def test_legacy_items_are_byte_identical_to_the_pilot(path):
    """Instrument v2 lives next to the legacy family: the four-option items, probe items and catalogue that the
    first pilot was measured on must never change (their finished stages are keyed on these bytes)."""
    out = generate(load_config(path))
    assert _sha(out["catalogue"]) == GOLDEN_CATALOGUE[path]
    for (p, family, split), sha in GOLDEN.items():
        if p == path:
            assert _sha([r for r in out[family] if r["split"] == split]) == sha, (family, split)


def test_data_hash_follows_generator_code_only(monkeypatch):
    """Editing the model registry, the variants or the config module must not ask for a data rebuild; editing
    a ladder generator OR its validator must (with the ladder enabled), and must not touch a legacy config."""
    from rtp import provenance

    real = provenance._normalized_source

    def hash_with_edit(cfg, module):
        monkeypatch.setattr(provenance, "_normalized_source",
                            lambda path: real(path) + (b"# edit" if path.name == module else b""))
        provenance.code_digest.cache_clear()
        try:
            return cfg.data_hash
        finally:
            monkeypatch.setattr(provenance, "_normalized_source", real)
            provenance.code_digest.cache_clear()

    on, off = mid_cfg(), config_from_dict({**mid_cfg().to_dict(), "ladder": {"enabled": False}})
    for module in ("registry.py", "variants.py", "config.py", "models.py", "cycle.py"):
        assert hash_with_edit(on, module) == on.data_hash, module
        assert hash_with_edit(off, module) == off.data_hash, module
    for module in ("ladder_items.py", "ladder_schema.py"):
        assert hash_with_edit(on, module) != on.data_hash, module
        assert hash_with_edit(off, module) == off.data_hash, module
    for module in ("choice_items.py", "cards.py", "constants.py"):
        assert hash_with_edit(off, module) != off.data_hash, module


def test_data_hash_ignores_run_settings_but_covers_the_ladder_settings():
    a, b = mid_cfg(), mid_cfg(rating_deltas=[0.2, 0.5])
    assert a.data_hash != b.data_hash  # ladder settings are part of the data
    off = config_from_dict({**a.to_dict(), "ladder": {"enabled": False}})
    off2 = config_from_dict({**a.to_dict(), "ladder": {"enabled": False, "rating_deltas": [0.2, 0.5]}})
    assert off.data_hash == off2.data_hash != a.data_hash  # a disabled ladder does not touch the legacy hash
    m = config_from_dict({**a.to_dict(), "ladder_measure": {"diag_fraction": 0.5}, "models": ["x/y"]})
    assert m.data_hash == a.data_hash  # measurement settings and the model list never rebuild data


# --------------------------------------------------------------------------- ladder items
def test_ladder_items_are_deterministic_and_seeded(mid):
    cfg, out = mid
    again = generate(cfg)["ladder"]
    assert again == out["ladder"]
    other = generate(config_from_dict({**cfg.to_dict(), "seed": 4}))["ladder"]
    assert [i["item_id"] for i in other] == [i["item_id"] for i in out["ladder"]]
    assert [i["prompt"] for i in other] != [i["prompt"] for i in out["ladder"]]


def test_core_pairs_are_regular():
    for k in (3, 4, 5, 6, 7, 8, 9):
        pairs = core_pairs(k)
        deg = Counter(x for p in pairs for x in p)
        assert len(set(deg.values())) == 1 and set(deg) == set(range(k)), k
        assert set(pairs) <= set(all_pairs(k))
    assert len(core_pairs(8)) == 12 and len(core_pairs(3)) == 3


def test_ladder_design_invariants(mid):
    cfg, out = mid
    items = out["ladder"]
    cat = out["catalogue"]
    fict = {c: [r["entity"] for r in cat if r["category"] == c and r["is_placebo"]] for c in out["categories"]}
    real = {c: [r["entity"] for r in cat if r["category"] == c and not r["is_placebo"]] for c in out["categories"]}
    steps = type_steps(cfg)
    assert len({i["item_id"] for i in items}) == len(items)
    assert {i["split"] for i in items if i["category"] == "beer"} == {"test"}
    by = defaultdict(list)
    for it in items:
        by[(it["category"], it["tier"], it["family"], it["stratum"], it["evidence_type"], it["d"])].append(it)
    for c in out["categories"]:
        k = len(fict[c])
        # anchors at d = 0: every unordered pair once (two option orders)
        a0 = by[(c, "core", "anchor", "name", "none", 0)]
        assert {frozenset((i["target_entity"], i["control_entity"])) for i in a0} == \
            {frozenset(p) for p in ((fict[c][i], fict[c][j]) for i, j in all_pairs(k))}
        assert len(a0) == 2 * len(all_pairs(k))
        for t in ("rating", "price", "reviews"):
            for lvl in range(1, len(steps[t]) + 1):
                cells = by[(c, "core", "anchor", "attr", t, lvl)] + by[(c, "ext", "anchor", "attr", t, lvl)]
                ordered = Counter((i["target_entity"], i["control_entity"]) for i in cells)
                # every pair of names in BOTH directions, equally often: name pulls cancel in the anchor curve
                assert len(ordered) == k * (k - 1) and len(set(ordered.values())) == 1
            # every real brand against every fictional control at d = 0
            r0 = by[(c, "core", "real", "attr", t, 0)]
            assert {(i["target_entity"], i["control_entity"]) for i in r0} == {(e, f) for e in real[c] for f in fict[c]}
        # direct reading: a few brands on both sides of d = 0
        dvals = {i["d"] for i in items if i["category"] == c and i["family"] == "real" and i["stratum"] == "attr"
                 and i["evidence_type"] == "rating"}
        assert min(dvals) == -cfg.ladder.favourable_levels and max(dvals) == len(steps["rating"])
        direct = {i["target_entity"] for i in items if i["category"] == c and i["family"] == "real" and i["d"] != 0}
        assert direct == set(real[c][: cfg.ladder.direct_entities])
    for it in items:  # the target sits at position `order`; the two orders of a unit swap the letters
        assert it["options"][it["order"]]["role"] == "target"
        assert len(it["options"]) == (4 if it["tier"] == "ctx" else 2)
    # anchors never contain a real brand, except as context in the four-option tier
    assert all(o["is_fictional"] or o["role"] == "context" for i in items if i["family"] == "anchor"
               for o in i["options"])
    stats = panel_stats(items)
    assert stats["dev/core"]["n_items"] == sum(i["tier"] == "core" and i["split"] == "dev" for i in items)
    assert stats["dev/ctx"]["n_options"] == 4 and stats["dev/core"]["n_units"] * 2 == stats["dev/core"]["n_items"]


def test_only_the_manipulated_attribute_differs(mid):
    _, out = mid
    for it in out["ladder"]:
        by = {o["label"]: o for o in it["options"]}
        t, c = by[it["target_label"]], by[it["control_label"]]
        diff = {f for f in ("price", "rating", "n_reviews") if t[f] != c[f]}
        want = {"rating": {"rating"}, "price": {"price"}, "reviews": {"n_reviews"},
                "composite": {"rating", "price", "n_reviews"}, "none": set()}[it["evidence_type"]]
        assert diff == (want if it["d"] != 0 else set()), it["item_id"]
        if it["stratum"] == "comp":
            assert "I need one with this feature:" in it["prompt"]  # the requirement is stated at EVERY level
        if it["stratum"] == "name":
            assert it["prompt"].count(": ") == 0  # names only: no card text after any option name


def test_review_ladder_never_reaches_the_one_review_floor():
    """Base review counts are at least ten times the largest ratio: no two steps collapse, the count shown at
    the top step is >= 10, and the ladder can be extended (cycle C0's rule) without breaking the build."""
    for ratios in ([2.0, 5.0, 20.0, 100.0], [2.0, 5.0, 20.0, 100.0, 300.0]):
        cfg = mid_cfg(review_ratios=ratios, n_base=3)
        shown = [o["n_reviews"] for it in generate(cfg)["ladder"] for o in it["options"] if o["n_reviews"] is not None]
        assert min(shown) >= 10, ratios
    cfg = mid_cfg()
    by_unit = {}
    for it in generate(cfg)["ladder"]:
        if it["evidence_type"] == "reviews" and it["stratum"] == "attr" and it["d"] > 0 and it["order"] == 0:
            o = {x["label"]: x for x in it["options"]}
            by_unit[(it["category"], it["base"], it["d"])] = o[it["control_label"]]["n_reviews"] / o[it["target_label"]]["n_reviews"]
    for (category, base, d), ratio in by_unit.items():  # the ratio shown is the nominal one, within rounding
        assert ratio == pytest.approx(cfg.ladder.review_ratios[d - 1], rel=0.06), (category, base, d)


def test_schema_rejects_a_collapsed_ladder_and_a_broken_unit(mid):
    cfg, out = mid
    cat = out["catalogue"]
    fict = {c: {r["entity"] for r in cat if r["category"] == c and r["is_placebo"]} for c in out["categories"]}
    real = {c: {r["entity"] for r in cat if r["category"] == c and not r["is_placebo"]} for c in out["categories"]}
    items = [dict(i) for i in out["ladder"]]
    assert validate_ladder_items(items, fict, real) == len(items)
    with pytest.raises(SchemaError, match="one item per option position"):
        validate_ladder_items(items[1:], fict, real)
    # two price steps that round to the same dollar amount are refused at build time
    with pytest.raises(SchemaError, match="not worse"):
        generate(mid_cfg(price_deltas=[0.0001, 0.0002]))
    with pytest.raises(ConfigError, match="strictly increasing"):
        mid_cfg(rating_deltas=[0.5, 0.3])


def test_build_writes_sealed_tier_files_and_refuses_shared_brands(work_dir):
    cfg = mid_cfg()
    ddir = build_data(cfg)
    manifest = io.read_json(ddir / "data_manifest.json")
    for split in ("dev", "test"):
        for tier in ("core", "ext", "ctx"):
            assert f"{split}/ladder_{tier}.jsonl" in manifest["sha256"]
    assert load_ladder_items(cfg, "core", "dev")
    with pytest.raises(SealedSplitError):
        load_ladder_items(cfg, "core", "test")
    assert all(i["split"] == "test" for i in load_ladder_items(cfg, "core", "test", unseal=True))
    assert "Ladder items (instrument version 1)" in (ddir / "data_report.md").read_text()
    # running shoe and sneaker share brands: they may not sit on two sides of the seal
    bad = config_from_dict({"name": "bad", "data": {"categories": ["running shoe", "sneaker"],
                                                    "test_categories": ["sneaker"], "n_real_per_category": 5,
                                                    "n_placebo_per_category": 4},
                            "ladder": {"enabled": True}})
    with pytest.raises(ValueError, match="share real brands"):
        generate(bad)


def test_e1_config_is_valid_and_brand_disjoint():
    cfg = load_config("configs/e1.yaml")
    assert cfg.ladder.enabled and cfg.data.n_placebo_per_category == 8
    assert not set(cfg.models) & set(cfg.confirm_models)
    assert "allenai/Llama-3.1-Tulu-3-8B" in cfg.confirm_models  # the RLVR stage stays unseen until the freeze
    pilot = load_config("configs/pilot.yaml")
    assert cfg.data.categories == pilot.data.categories and cfg.data.test_categories == pilot.data.test_categories


# --------------------------------------------------------------------------- variants
def test_variant_ids_resolve_to_their_base_checkpoint(work_dir):
    base = registry.SMOKE_MODEL
    spec = registry.get_model_spec(f"{base}@scale=2")
    assert (spec.base_id, spec.logit_scale, spec.is_chat) == (base, 2.0, True)
    assert registry.base_model_id(f"{base}@scale=0.5") == base and registry.base_model_id(base) == base
    assert registry.get_model_spec("some/unregistered") is None
    for bad in ("nope/x@scale=2", f"{base}@scale=-1", f"{base}@scale=abc", f"{base}@temp=2", "zoo:missing"):
        with pytest.raises(variants.VariantError):
            registry.get_model_spec(bad)
    slugs = {model_slug(m) for m in (base, f"{base}@scale=2", f"{base}@scale=2.0", f"{base}_scale_2")}
    assert len(slugs) == 4  # no two ids share a directory


def test_stage_identity_covers_the_variant_and_the_tier(work_dir):
    from rtp import stages

    cfg = mid_cfg()
    build_data(cfg)
    base = registry.SMOKE_MODEL
    h = {(m, t): stages.stage_hash(cfg, "ladder", m, "dev", tier=t)
         for m in (base, f"{base}@scale=2") for t in ("core", "ext")}
    assert len(set(h.values())) == 4
    with pytest.raises(ValueError):
        stages.stage_hash(cfg, "ladder", base, "dev", tier="nope")


def test_zoo_adapter_registration_and_tamper_check(work_dir):
    adir = variants.adapter_dir("run1")
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text("{}")
    (adir / "adapter_model.safetensors").write_bytes(b"weights-v1")
    with pytest.raises(variants.VariantError):  # the base of an adapter must be a registered checkpoint
        variants.register_adapter("run1", "not/registered")
    entry = variants.register_adapter("run1", registry.SMOKE_MODEL, {"seed": 0})
    spec = registry.get_model_spec("zoo:run1")
    assert spec.base_id == registry.SMOKE_MODEL and spec.adapter == "run1"
    assert spec.adapter_sha256 == entry["adapter_sha256"]
    assert variants.verified_adapter_dir("run1") == adir
    (adir / "adapter_model.safetensors").write_bytes(b"weights-v2")  # retrained in place: refused
    with pytest.raises(variants.VariantError, match="changed on disk"):
        variants.verified_adapter_dir("run1")


# --------------------------------------------------------------------------- answers with N options
def test_answer_paths_and_estimators_for_two_options():
    from rtp.measure.answers import PositionResult, build_item_paths, build_name_paths, estimators

    class Tok:
        vocab = {"A": 1, " A": 2, "B": 3, " B": 4, "**": 5, "(": 6, "Nike": 7, " Nike": 8, "Acer": 9, " Acer": 10,
                 "Goo": 11, "gle": 12, " Pixel": 13}

        def __call__(self, text, add_special_tokens=False):
            out, rest = [], text
            while rest:
                hit = max((k for k in self.vocab if rest.startswith(k)), key=len, default=None)
                if hit is None:
                    return {"input_ids": []}
                out.append(self.vocab[hit])
                rest = rest[len(hit):]
            return {"input_ids": out}

    tok = Tok()
    letters = {"A": [1, 2], "B": [3, 4], "C": [], "D": []}
    paths = build_item_paths(tok, ["Nike", "Acer"], None, letters, {})
    assert len(paths.classes) == 3 and not paths.classes[2]  # two options + an empty leak class
    with pytest.raises(ValueError):
        build_item_paths(tok, ["Nike"], None, letters, {})
    free = PositionResult(q=np.array([0.2, 0.1, 0.0]), q_name=np.array([0.05, 0.0, 0.0]), pruned=0.0)
    forced = PositionResult(q=np.array([0.6, 0.3, 0.0]), q_name=np.zeros(3), pruned=0.0)
    e = estimators(free, forced, 0)
    assert len(e["P"]) == 2 and e["P"].sum() == pytest.approx(1.0)
    assert e["c_forced"] == pytest.approx(0.9) and e["s_target_forced"] == pytest.approx(np.log(2.0))
    names = build_name_paths(tok, [["Google"], ["Google Pixel"], ["Nike"]], {})
    assert len(names.classes) == 3 and all(c and all(kind == "name" for _, kind in c) for c in names.classes)
    assert not names.a_letter  # an open question has no letters and no leak class
    google = {p for p, _ in names.classes[0]}
    assert any(q[: len(p)] == p and len(q) > len(p) for p in google for q, _ in names.classes[1])  # "Google" < "Google Pixel"


def test_parsed_answers_are_restricted_to_the_item_options():
    from rtp.measure.parse import parse_answer, restrict_to_options

    opts = [{"label": "A", "name": "Nike"}, {"label": "B", "name": "Trake"}]
    assert restrict_to_options(parse_answer("B", opts), opts) == "B"
    assert restrict_to_options(parse_answer("C", opts), opts) is None  # "C" is not an option of a two-option item
    assert restrict_to_options(parse_answer("Trake is my pick", opts), opts) == "B"


# --------------------------------------------------------------------------- readout on the real smoke model
@pytest.mark.integration
def test_letter_readout_matches_the_model_logits_and_scales_exactly(smoke_lm_fp32):
    from dataclasses import replace

    import torch

    from rtp.measure.ladder import letter_forms, letter_readout
    from rtp.measure.parse import render_inputs
    from rtp.models import scale_logits

    lm = smoke_lm_fp32
    forms = letter_forms(lm.tokenizer)
    assert all(f["sp"] is not None and f["ns"] is not None and f["sp"] in f["all"] for f in forms.values())
    prompts = ["I'm shopping for a laptop. Here are two options:\n\nA. Satek\nB. Klemol\n\nWhich one should I buy? "
               "Answer with a single letter.", "Pick one of two options:\n\nA. Nike: Rating: 4.1/5.\nB. Trake: Rating: "
               "4.6/5.\n\nWhich one should I buy? Answer with a single letter."]
    free, forced = render_inputs(lm, prompts, prefill=False), render_inputs(lm, prompts, prefill=True)
    ids = sorted({t for f in forms.values() for t in f["all"]})
    out = letter_readout(lm, free, forced, ids, 8, 4096)
    for k, r in enumerate(out):
        for seq, key in ((forced[k], "forced"), (free[k], "free")):
            with torch.inference_mode():
                logits = lm.model(input_ids=torch.tensor([seq], device=lm.device)).logits[0, -1].float()
            assert np.allclose(r[f"z_{key}"], logits[ids].cpu().numpy(), atol=2e-3)
            assert r[f"lse_{key}"] == pytest.approx(float(torch.logsumexp(logits, -1)), abs=2e-3)
    # the output hook of a "@scale=c" variant and the readout's own factor agree, exactly
    handle = scale_logits(lm.model, 2.0)
    try:
        with torch.inference_mode():
            hooked = lm.model(input_ids=torch.tensor([forced[0]], device=lm.device)).logits[0, -1].float()
        assert np.allclose(hooked[ids].cpu().numpy(), 2.0 * out[0]["z_forced"], atol=4e-3)
        doubled = letter_readout(replace(lm, logit_scale=2.0), free, forced, ids, 8, 4096)  # as load_model sets it up
    finally:
        handle.remove()
    assert np.array_equal(doubled[0]["z_forced"], 2.0 * out[0]["z_forced"])  # a power of two: bit-exact
    assert max(r["head_dev"] for r in out) < 0.25  # recomputed and model logits agree (float32: exactly)
    # ... and a model whose logits are NOT its head applied to the final hidden state is refused, not misread:
    # here the output is halved after the head while the loader was not told about any scale
    handle = scale_logits(lm.model, 0.5)
    try:
        with pytest.raises(RuntimeError, match="does not map its final hidden state"):
            letter_readout(lm, free, forced, ids, 8, 4096)
        scaled = letter_readout(replace(lm, logit_scale=0.5), free, forced, ids, 8, 4096)  # a declared variant passes
    finally:
        handle.remove()
    assert np.allclose(scaled[0]["z_forced"], 0.5 * out[0]["z_forced"])


@pytest.mark.integration
def test_ladder_and_open_set_stages_run_end_to_end(smoke_lm, work_dir, monkeypatch):
    import pandas as pd

    from rtp.measure.ladder import run_ladder
    from rtp.measure.open_set import run_open_set

    cfg = config_from_dict({"name": "lad", "seed": 1, "data": {
        "categories": ["laptop"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0], "handicaps": [0], "probe_entities_per_category": 2, "probe_templates": 2},
        "ladder": {"enabled": True, "rating_deltas": [0.5], "price_deltas": [0.3], "review_ratios": [10.0],
                   "n_base": 1, "direct_entities": 1, "direct_controls": 1, "favourable_levels": 1},
        "ladder_measure": {"shard_size": 50, "diag_fraction": 0.2, "gen_check_fraction": 0.1, "n_samples": 2,
                           "max_batch_size": 16, "max_batch_tokens": 4096},
        "open_set": {"prompt_types": ["name"], "temperatures": [1.0], "n_samples": 3, "max_batch_size": 8}})
    build_data(cfg)
    monkeypatch.setattr("rtp.models.load_model", lambda path: smoke_lm)
    items = load_ladder_items(cfg, "core", "dev")
    sdir = run_ladder(cfg, smoke_lm.model_id, tier="core")
    df = pd.read_parquet(sdir / "scores.parquet")
    assert list(df["item_id"]) == [i["item_id"] for i in items]
    assert df[["z_sp_0", "z_sp_1", "lp_forced_0", "lp_forced_1"]].notna().all().all()
    assert df[["z_sp_2", "z_sp_3"]].isna().all().all()  # two options: the other letters are not read
    assert ((df["c_forced"] > 0) & (df["c_forced"] <= 1.0 + 1e-6)).all()
    assert df["cq_forced"].notna().sum() >= 1 and df["gen_mode"].notna().sum() >= 1
    info = io.read_json(sdir / io.DONE_FILE)
    assert info["n_items"] == len(items) and info["letter_form"] in ("sp", "ns") and info["timing_complete"]
    assert (sdir / "summary.md").exists() and (sdir / "letter_tokens.json").exists()
    assert run_ladder(cfg, smoke_lm.model_id, tier="core") == sdir  # finished: skipped
    ctx = pd.read_parquet(run_ladder(cfg, smoke_lm.model_id, tier="ctx") / "scores.parquet")
    assert (ctx["n_options"] == 4).all() and ctx[["z_sp_2", "z_sp_3"]].notna().all().all()
    odir = run_open_set(cfg, smoke_lm.model_id)
    closed = pd.read_parquet(odir / "closed_set.parquet")
    assert set(closed["entity"]) == {i["target_entity"] for i in items if i["family"] == "real"}
    assert ((closed["p"] >= 0) & (closed["coverage"] <= 1.0 + 1e-6)).all()
    assert len(pd.read_parquet(odir / "samples.parquet")) == 3


@pytest.mark.integration
def test_lora_adapter_round_trip_through_the_zoo(smoke_lm, work_dir):
    """A LoRA adapter saved to the zoo loads through a ``zoo:<run>`` id, changes the logits, keeps the model's
    class (so layer access still works), and disabling it reproduces the base model exactly."""
    import copy

    import torch
    from peft import LoraConfig, get_peft_model

    from rtp.models import get_blocks, load_model

    wrapped = get_peft_model(copy.deepcopy(smoke_lm.model),
                             LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    torch.manual_seed(0)
    for name, p in wrapped.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.05)
    wrapped.save_pretrained(str(variants.adapter_dir("tiny")))
    del wrapped
    variants.register_adapter("tiny", registry.SMOKE_MODEL, {"note": "test"})
    lm = load_model("zoo:tiny")
    assert type(lm.model) is type(smoke_lm.model) and len(get_blocks(lm.model)) == len(get_blocks(smoke_lm.model))
    ids = torch.tensor([lm.tokenizer("Which laptop should I buy?")["input_ids"]], device=lm.device)
    ref_ids = ids.to(smoke_lm.device)
    with torch.inference_mode():
        with_adapter = lm.model(input_ids=ids).logits.float().cpu()
        base = smoke_lm.model(input_ids=ref_ids).logits.float().cpu()
        lm.model.disable_adapters()
        without = lm.model(input_ids=ids).logits.float().cpu()
    assert (with_adapter - base).abs().max() > 1e-3
    assert torch.allclose(without, base, atol=1e-4)


# --------------------------------------------------------------------------- GPU closure
GPU_CLOSURE = {"rtp.infer", "rtp.measure.answers", "rtp.measure.parse", "rtp.models", "rtp.patching", "rtp.rng",
               "rtp.variants"}
PINNED_DIGESTS = {"rtp.measure.ladder": "a04ba487b30ba318", "rtp.measure.open_set": "3f337003bc4dd49d"}


def test_gpu_stages_do_not_depend_on_analysis_or_legacy_code():
    """Hard constraint 6: an edit to an estimator, to the cycle machinery or to the legacy measurement code must
    never invalidate a finished ladder or open-set stage, so none of them may be in the stages' import closure."""
    from rtp.provenance import MODEL_STAGE_EXCLUDE_PREFIXES, PLUMBING, import_closure

    for module in ("rtp.measure.ladder", "rtp.measure.open_set"):
        hashed = {m for m in import_closure(module)
                  if m not in PLUMBING and m != "rtp.config" and not m.startswith(MODEL_STAGE_EXCLUDE_PREFIXES)}
        hashed.discard("rtp.data")  # the package's empty __init__ (the generator modules below it are excluded)
        assert hashed == GPU_CLOSURE | {module}, sorted(hashed ^ (GPU_CLOSURE | {module}))


def test_gpu_stage_digests_are_pinned():
    """The digest of the code that produces GPU results. If this test fails you edited a module of the GPU
    closure (see GPU_CLOSURE): every finished ladder / open-set stage is now STALE and must be re-measured
    (a cycle: `cycle run --redo-stale`, develop phase only; never after the freeze). If that is intended,
    update PINNED_DIGESTS in the same commit and say so in docs/CYCLES.md; if not, move the change out of
    the closure (estimators belong in rtp/scale, views in rtp/analyze.py)."""
    from rtp.provenance import stage_code_digest

    for module, pinned in PINNED_DIGESTS.items():
        assert stage_code_digest(module) == pinned, f"{module}: digest is now {stage_code_digest(module)}"
