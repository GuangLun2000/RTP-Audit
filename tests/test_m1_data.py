"""M1 data generation: determinism, schema/design invariants, sealed split, placebo conflicts, seeds."""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from rtp import io
from rtp.cli import main as cli_main
from rtp.config import REPO_ROOT, config_from_dict, load_config
from rtp.data import schema
from rtp.data.build import build_data, generate, load_catalogue, load_choice_items, load_probe_items
from rtp.data.catalogue import load_category_seeds, load_natural_negatives, normalize_name
from rtp.data.choice_items import rotate
from rtp.data.placebo import (PlaceboConflictError, find_conflicts, generate_placebo_names, load_placebo_file,
                              validate_placebos)
from rtp.data.splits import SealedSplitError

NN_SHA256 = "a72bfa7851c7b241a5e7701e8458b0778819f43afabd04a924f15550738e8638"


def tiny_cfg(**overrides):
    raw = {"name": "tiny", "seed": 7, "data": {
        "categories": ["running shoe", "beer"], "test_categories": ["beer"],
        "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "probe_entities_per_category": 2, "probe_templates": 3}}
    raw.update(overrides)
    return config_from_dict(raw)


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- determinism
def test_same_seed_gives_identical_files(tmp_path, monkeypatch):
    cfg = load_config("configs/smoke.yaml")
    hashes = []
    for run in ("a", "b"):
        monkeypatch.setenv("RTP_WORK_DIR", str(tmp_path / run))
        ddir = build_data(cfg)
        hashes.append(_hashes(ddir))
    assert hashes[0] == hashes[1]
    assert "dev/choice_items.jsonl" in hashes[0] and "catalogue.jsonl" in hashes[0]


def test_different_seed_changes_items():
    a = generate(tiny_cfg())
    b = generate(tiny_cfg(seed=8))
    assert [i["prompt"] for i in a["choice"]] != [i["prompt"] for i in b["choice"]]
    assert [i["item_id"] for i in a["choice"]] == [i["item_id"] for i in b["choice"]]


# --------------------------------------------------------------------------- design invariants
@pytest.fixture(scope="module")
def tiny():
    return generate(tiny_cfg())


def test_counts_and_cells(tiny):
    items = tiny["choice"]
    # 2 categories x 3 targets x 13 cells x 1 card x 2 names x 4 rotations
    assert len(items) == 2 * 3 * 13 * 1 * 2 * 4
    for it in items:
        if it["evidence_level"] < 3:
            assert it["handicap"] == 0
    cells = {(i["evidence_level"], i["handicap"]) for i in items}
    assert len(cells) == 13


def test_rotations_place_target_at_every_letter(tiny):
    by_pair = {}
    for it in tiny["choice"]:
        by_pair.setdefault((it["pair_id"], it["target_is_placebo"]), set()).add(it["target_label"])
    assert all(v == {"A", "B", "C", "D"} for v in by_pair.values())
    assert rotate(["t", "x", "y", "z"], 2) == ["y", "z", "t", "x"]


def test_pair_differs_only_in_target_name(tiny):
    items = {i["item_id"]: i for i in tiny["choice"]}
    for iid, real in items.items():
        if not iid[-2] == "r":
            continue
        plac = items[iid[:-2] + "p" + iid[-1]]
        diff = [(a, b) for a, b in zip(real["prompt"].split("\n"), plac["prompt"].split("\n")) if a != b]
        assert len(diff) == 1
        a, b = diff[0]
        assert a.replace(real["target_entity"], plac["target_entity"], 1) == b


def test_brand_name_first_on_option_lines(tiny):
    for it in tiny["choice"][:200]:
        lines = [ln for ln in it["prompt"].split("\n") if re.match(r"^[ABCD]\. ", ln)]
        assert len(lines) == 4
        for ln, opt in zip(lines, it["options"]):
            assert ln.startswith(f"{opt['label']}. {opt['name']}")
            assert ln == f"{opt['label']}. {opt['name']}" or ln.startswith(f"{opt['label']}. {opt['name']}: ")


def test_handicap_lowers_target_rating(tiny):
    deltas = tiny_cfg().data.handicap_rating_deltas
    for it in tiny["choice"]:
        if it["evidence_level"] < 3:
            assert all(o["rating"] is None for o in it["options"])
            continue
        tgt = next(o for o in it["options"] if o["role"] == "target")
        others = {o["rating"] for o in it["options"] if o["role"] != "target"}
        assert len(others) == 1
        assert tgt["rating"] == pytest.approx(others.pop() - deltas[it["handicap"]], abs=1e-9)


def test_evidence_levels_reveal_fields_cumulatively(tiny):
    for it in tiny["choice"]:
        e = it["evidence_level"]
        o = it["options"][0]
        assert (o["description"] is not None) == (e >= 1)
        assert (o["price"] is not None) == (e >= 2) == (o["features"] is not None)
        assert (o["rating"] is not None) == (e >= 3) == (o["n_reviews"] is not None)
        assert (o["lab_note"] is not None) == (e >= 4)


def test_options_composition(tiny):
    for it in tiny["choice"]:
        roles = Counter(o["role"] for o in it["options"])
        assert roles == {"target": 1, "competitor": 2, "placebo_distractor": 1}
        names = [o["name"] for o in it["options"]]
        assert len(set(names)) == 4


def test_schema_catches_tampering(tiny):
    items = copy.deepcopy([i for i in tiny["choice"] if i["pair_id"] == tiny["choice"][0]["pair_id"]])
    schema.validate_choice_items(items)
    bad = copy.deepcopy(items)
    victim = next(i for i in bad if i["target_is_placebo"])
    comp = next(o for o in victim["options"] if o["role"] == "competitor")
    comp["price"] = 1.0
    with pytest.raises(schema.SchemaError):
        schema.validate_choice_items(bad)
    bad2 = copy.deepcopy(items)
    bad2[0]["rotation"] = (bad2[0]["rotation"] + 1) % 4
    with pytest.raises(schema.SchemaError):
        schema.validate_choice_items(bad2)


def test_probe_items(tiny):
    probes = tiny["probe"]
    assert len(probes) == 2 * 2 * 3
    choice_placebos = {r["entity"] for r in tiny["catalogue"] if r["is_placebo"]}
    for p in probes:
        s, t = p["name_char_span"]
        assert p["text"][s:t] == p["entity"]
        assert p["text"].startswith(f"Background: {p['entity']} is a")
        assert p["text"].endswith(f"I'm considering {p['entity']}.")
        assert p["entity"] not in choice_placebos  # probe pool disjoint from choice placebos
    texts = Counter(p["text"] for p in probes)
    assert max(texts.values()) == 1


def test_catalogue_has_natural_negatives(tiny):
    cat = tiny["catalogue"]
    rs = {r["entity"]: r for r in cat if r["category"] == "running shoe" and not r["is_placebo"]}
    assert rs["Nike"]["natural_negative"] == "named" and rs["Brooks"]["natural_negative"] == "picked"
    beer = [r for r in cat if r["category"] == "beer"]
    assert all(r["split"] == "test" for r in beer)


# --------------------------------------------------------------------------- sealed split
def test_test_split_is_sealed(work_dir):
    cfg = tiny_cfg()
    build_data(cfg)
    with pytest.raises(SealedSplitError):
        load_choice_items(cfg, "test")
    with pytest.raises(SealedSplitError):
        load_probe_items(cfg, "test")
    with pytest.raises(SealedSplitError):
        load_catalogue(cfg, "test")
    dev = load_choice_items(cfg, "dev")
    test = load_choice_items(cfg, "test", unseal=True)
    assert {i["category"] for i in dev} == {"running shoe"}
    assert {i["category"] for i in test} == {"beer"}


def test_cli_refuses_test_split_without_unseal(work_dir, tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "tiny", "data": {
        "categories": ["running shoe", "beer"], "test_categories": ["beer"], "n_real_per_category": 3,
        "n_placebo_per_category": 3, "cards_per_cell": 1, "probe_entities_per_category": 1, "probe_templates": 1}}))
    assert cli_main(["build-data", "--config", str(cfg_path)]) == 0
    for stage in ("measure", "open-ended"):
        assert cli_main([stage, "--config", str(cfg_path), "--model", "dummy/model", "--split", "test"]) == 3


def test_cell_count_report(work_dir):
    cfg = tiny_cfg()
    ddir = build_data(cfg)
    import pandas as pd

    counts = pd.read_csv(ddir / "cell_counts.csv")
    manifest = io.read_json(ddir / "data_manifest.json")
    assert counts["n_items"].sum() == manifest["n_records"]["dev/choice_items.jsonl"] + \
        manifest["n_records"]["test/choice_items.jsonl"]
    assert (ddir / "data_report.md").exists() and io.is_done(ddir)


# --------------------------------------------------------------------------- placebo names
def test_conflict_check_case_insensitive_substrings():
    real = ["Nike", "Sierra Nevada", "HP", "Kia", "Hermès", "Peet's Coffee"]
    assert find_conflicts("NIKEX", real) == ["Nike"]
    assert find_conflicts("anike", real) == ["Nike"]
    assert find_conflicts("Nevadix", real) == []
    assert find_conflicts("Nevadaro", real) == ["Sierra Nevada"]
    assert find_conflicts("Kiano", real) == ["Kia"]
    assert find_conflicts("Hpro", real) == []  # names shorter than 3 chars match exactly only
    assert find_conflicts("hp", real) == ["HP"]
    assert find_conflicts("HERMESO", real) == ["Hermès"]
    assert find_conflicts("Peetso", real) == ["Peet's Coffee"]
    assert find_conflicts("Ni", real) == ["Nike"]  # placebo inside a real name also collides


def test_generated_names_are_conflict_free_and_deterministic():
    real = ["Nike", "Brooks", "Dell", "Apple"]
    a = generate_placebo_names(30, 0, real)
    assert a == generate_placebo_names(30, 0, real)
    assert len({normalize_name(x) for x in a}) == 30
    assert all(find_conflicts(x, real) == [] for x in a)


def test_seed_placebo_file_is_valid_and_flagged():
    seeds = load_category_seeds(REPO_ROOT / "data/seeds/categories.yaml")
    nn = load_natural_negatives(REPO_ROOT / "data/seeds/natural_negatives.csv")
    placebos = load_placebo_file(REPO_ROOT / "data/seeds/placebo_names.yaml")
    from rtp.data.catalogue import all_real_names

    stats = validate_placebos(placebos, all_real_names(seeds, nn))
    assert stats["n_names"] > 0 and 0 <= stats["n_needs_web_check"] <= stats["n_names"]
    assert all(isinstance(e["needs_web_check"], bool) for pool in placebos.values() for es in pool.values() for e in es)
    poisoned = copy.deepcopy(placebos)
    poisoned["choice"]["laptop"][0]["name"] = "Dellora"
    with pytest.raises(PlaceboConflictError):
        validate_placebos(poisoned, all_real_names(seeds, nn))


def test_categories_yaml_review_flags_and_appendix_a_coverage():
    """The review flags exist and are booleans (their values change once a human has reviewed them), and every
    Appendix A category has brands either with choice items or in open_ended_only."""
    raw = yaml.safe_load((REPO_ROOT / "data/seeds/categories.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw["needs_review"], bool)
    sections = list(raw["categories"].values()) + list(raw["open_ended_only"].values())
    assert all(isinstance(c["needs_review"], bool) for c in sections)
    assert len(raw["categories"]) >= 10
    nn = load_natural_negatives(REPO_ROOT / "data/seeds/natural_negatives.csv")
    covered = set(raw["categories"]) | set(raw["open_ended_only"])
    assert {r["category"] for r in nn} <= covered


def test_no_brand_brand_in_templates():
    cfg = config_from_dict({"name": "t", "data": {"categories": ["coffee brand", "snack brand"], "n_real_per_category": 3,
                                                  "n_placebo_per_category": 3, "cards_per_cell": 1,
                                                  "evidence_levels": [0], "probe_entities_per_category": 1,
                                                  "probe_templates": 2}})
    out = generate(cfg)
    assert out["probe"] and all("brand brand" not in p["text"] for p in out["probe"])
    assert out["probe"][0]["text"].startswith(f"Background: {out['probe'][0]['entity']} is a coffee brand.")
    from rtp.measure.open_ended import GEN_TEMPLATES, gen_prompt_text

    for t in GEN_TEMPLATES:
        assert gen_prompt_text("coffee brand", t).count("brand") == 1
    assert gen_prompt_text("running shoe", "name") == "Name a running shoe brand. Reply with the name only."
    assert gen_prompt_text("electric car", "recommend") == \
        "Can you recommend an electric car brand? Reply with the name only."


@pytest.mark.parametrize("resume", [False, True])
def test_smoke_never_deletes_or_adopts_foreign_run_dirs(work_dir, tmp_path, resume):
    from rtp.paths import run_dir
    from rtp.smoke import SMOKE_MARKER, SmokeFailure, run_smoke

    cfg_path = tmp_path / "pilotlike.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "pilotlike"}))
    precious = run_dir("pilotlike") / "some_model" / "measure"
    precious.mkdir(parents=True)
    with pytest.raises(SmokeFailure, match="not created by"):
        run_smoke(str(cfg_path), resume=resume)
    assert precious.exists() and not (run_dir("pilotlike") / SMOKE_MARKER).exists()


def test_smoke_refuses_to_delete_marked_dir_with_other_models(work_dir, tmp_path):
    from rtp.paths import run_dir
    from rtp.smoke import SMOKE_MARKER, SmokeFailure, run_smoke

    cfg_path = tmp_path / "s2.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "s2"}))
    rd = run_dir("s2")
    (rd / "allenai__Llama-3.1-Tulu-3-8B-SFT" / "measure").mkdir(parents=True)
    (rd / SMOKE_MARKER).write_text("x")
    with pytest.raises(SmokeFailure, match="other models"):
        run_smoke(str(cfg_path))
    assert (rd / "allenai__Llama-3.1-Tulu-3-8B-SFT").exists()


def test_data_dir_holds_seed_snapshot_and_data_only_config(work_dir):
    cfg = tiny_cfg()
    ddir = build_data(cfg)
    saved = yaml.safe_load((ddir / "config.yaml").read_text())
    assert set(saved) == {"name", "seed", "data", "seed_file_digests"}
    assert {p.name for p in (ddir / "seeds").iterdir()} == {"categories.yaml", "placebo_names.yaml",
                                                             "natural_negatives.csv"}
    assert (ddir / "seeds" / "natural_negatives.csv").read_bytes() == \
        (REPO_ROOT / "data/seeds/natural_negatives.csv").read_bytes()


def test_open_ended_only_brands_are_in_the_conflict_list():
    from rtp.data.catalogue import all_real_names

    seeds = load_category_seeds(REPO_ROOT / "data/seeds/categories.yaml")
    nn = load_natural_negatives(REPO_ROOT / "data/seeds/natural_negatives.csv")
    real = all_real_names(seeds, nn)
    assert find_conflicts("Letim", real) == ["Tim Hortons"]
    assert find_conflicts("Hortonsa", real) == ["Tim Hortons"]


# --------------------------------------------------------------------------- Appendix A
def test_natural_negatives_csv_is_appendix_a_verbatim():
    path = REPO_ROOT / "data/seeds/natural_negatives.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NN_SHA256
    rows = load_natural_negatives(path)
    assert len(rows) == 26 and rows[0] == {"category": "AI assistant", "named": "Siri", "picked": "Claude"}
    brief = REPO_ROOT / "RTP_AUDIT_AGENT.md"
    if brief.exists():  # the spec is gitignored; compare against it when present
        text = brief.read_text(encoding="utf-8")
        block = text.split("```csv\n", 1)[1].split("```", 1)[0]
        assert path.read_text(encoding="utf-8") == block
