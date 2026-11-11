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
    assert stats["n_names"] == stats["n_needs_web_check"] > 0
    poisoned = copy.deepcopy(placebos)
    poisoned["choice"]["laptop"][0]["name"] = "Dellora"
    with pytest.raises(PlaceboConflictError):
        validate_placebos(poisoned, all_real_names(seeds, nn))


def test_categories_yaml_draft_is_marked_for_review():
    raw = yaml.safe_load((REPO_ROOT / "data/seeds/categories.yaml").read_text(encoding="utf-8"))
    assert raw["needs_review"] is True
    assert all(c["needs_review"] is True for c in raw["categories"].values())
    assert len(raw["categories"]) == 10


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
