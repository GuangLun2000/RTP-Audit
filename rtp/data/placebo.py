"""Placebo (fictional) brand names: generation and conflict checking.

Conflict rule (spec §6 M1): case-insensitive, including substrings, against every real
brand name and alias we know (all categories in categories.yaml plus every Appendix A
brand), and against the words of multi-word real names. Real strings shorter than 3
characters (e.g. "HP", "VW") only conflict on exact equality, otherwise nearly every
pseudo-word would collide. Placebos must also not contain one another.

A name that passes this check can still exist in the real world: every generated name
carries ``needs_web_check: true`` until a human has searched for it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from rtp.config import REPO_ROOT
from rtp.data.catalogue import all_real_names, load_category_seeds, load_natural_negatives, normalize_name
from rtp.rng import rng_for

log = logging.getLogger(__name__)

ONSETS = ("b", "br", "d", "dr", "f", "fl", "g", "gr", "k", "kl", "l", "m", "n", "p", "pr", "r", "s", "st",
          "t", "tr", "v", "z")
VOWELS = ("a", "e", "i", "o", "u")
CODAS = ("", "", "n", "r", "l", "s", "x", "m", "k")
MIN_SUBSTRING_LEN = 3
# Names starting with A-D can tokenise to a bare letter token (" D" + "regar"), which the option-letter
# scorer would count as an answer letter. Placebos never start with them.
FORBIDDEN_INITIALS = frozenset("abcd")


class PlaceboConflictError(ValueError):
    pass


def _forbidden_strings(real_names: list[str]) -> tuple[set[str], set[str]]:
    """Return (substring-checked strings, exact-only strings), all normalised and space-free."""
    sub, exact = set(), set()
    for name in real_names:
        norm = normalize_name(name)
        candidates = {norm.replace(" ", "")} | set(norm.split())
        for c in candidates:
            if not c:
                continue
            (sub if len(c) >= MIN_SUBSTRING_LEN else exact).add(c)
    return sub, exact


def find_conflicts(name: str, real_names: list[str]) -> list[str]:
    """Real names (or name words) that collide with ``name`` (case-insensitive, substring both ways)."""
    p = normalize_name(name).replace(" ", "")
    hits = []
    for real in real_names:
        norm = normalize_name(real)
        for c in {norm.replace(" ", "")} | set(norm.split()):
            if not c:
                continue
            if len(c) < MIN_SUBSTRING_LEN:
                bad = c == p
            else:
                bad = c in p or p in c
            if bad:
                hits.append(real)
                break
    return hits


def _pseudo_word(rng) -> str:
    n_syll = 2 if rng.random() < 0.8 else 3
    parts = [ONSETS[rng.integers(len(ONSETS))] + VOWELS[rng.integers(len(VOWELS))] for _ in range(n_syll)]
    parts[-1] += CODAS[rng.integers(len(CODAS))]
    return "".join(parts).capitalize()


def generate_placebo_names(n: int, seed: int, real_names: list[str], taken: list[str] | None = None,
                           stream: str = "placebo") -> list[str]:
    """Generate ``n`` conflict-free pseudo-word brand names, deterministic in (seed, stream)."""
    rng = rng_for(seed, stream)
    sub, exact = _forbidden_strings(real_names)
    taken_norm = [normalize_name(t).replace(" ", "") for t in (taken or [])]
    out: list[str] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 1000 * max(n, 1):
            raise RuntimeError("could not generate enough conflict-free placebo names")
        cand = _pseudo_word(rng)
        c = cand.lower()
        if not 4 <= len(c) <= 8:
            continue
        if c[0] in FORBIDDEN_INITIALS:  # would collide with the option letters A-D in many tokenizers
            continue
        if c in exact or any(s in c or c in s for s in sub):
            continue
        if any(t in c or c in t for t in taken_norm):
            continue
        out.append(cand)
        taken_norm.append(c)
    return out


def placebo_problem(name: str, real_names: list[str]) -> str | None:
    """Why a placebo name is unusable (None if it is fine)."""
    if normalize_name(name)[:1] in FORBIDDEN_INITIALS:
        return "starts with A-D (letter collision)"
    hits = find_conflicts(name, real_names)
    return f"collides with {hits}" if hits else None


def make_placebo_file(categories_file: str, natural_negatives_file: str, out: str, n_choice: int = 8,
                      n_probe: int = 40, seed: int = 0, repair: bool = False) -> Path:
    """Write ``placebo_names.yaml`` with a choice pool and a probe pool per category.

    ``repair=True`` keeps the existing file and replaces ONLY unusable names (A-D initial, conflict
    with a real brand, or collision with an earlier placebo), so names and review flags that are
    still valid survive; categories missing from the file get new pools. Replacements are
    deterministic in (seed, pool, category, old name).
    """
    cpath, npath, opath = (_resolve(p) for p in (categories_file, natural_negatives_file, out))
    seeds = load_category_seeds(cpath)
    real = all_real_names(seeds, load_natural_negatives(npath))
    if repair:
        return _repair_placebo_file(opath, real, seed, list(seeds.categories), n_choice, n_probe)
    taken: list[str] = []
    body: dict[str, Any] = {"choice": {}, "probe": {}}
    for cat in seeds.categories:
        for pool, n in (("choice", n_choice), ("probe", n_probe)):
            names = generate_placebo_names(n, seed, real, taken=taken, stream=f"{pool}:{cat}")
            taken.extend(names)
            body[pool][cat] = [{"name": nm, "needs_web_check": True} for nm in names]
    meta = {"generator": {"method": "syllable-v2", "seed": seed, "n_choice": n_choice, "n_probe": n_probe}}
    return _write_placebo_yaml(opath, meta, body, len(taken))


def _repair_placebo_file(opath: Path, real: list[str], seed: int, categories: list[str] | None = None,
                         n_choice: int = 8, n_probe: int = 40) -> Path:
    with open(opath, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    pools = load_placebo_file(opath)
    kept: list[str] = []
    for pool in ("choice", "probe"):  # first pass: every valid, mutually non-colliding name is kept
        for cat, entries in pools[pool].items():
            for e in entries:
                c = normalize_name(e["name"]).replace(" ", "")
                if placebo_problem(e["name"], real) is None and not any(k in c or c in k for k in kept):
                    kept.append(c)
                else:
                    e["_replace"] = True
    n_fixed = 0
    for pool in ("choice", "probe"):
        for cat, entries in pools[pool].items():
            for i, e in enumerate(entries):
                if not e.pop("_replace", False):
                    continue
                new = generate_placebo_names(1, seed, real, taken=kept, stream=f"repair:{pool}:{cat}:{e['name']}")[0]
                log.info("placebo %s/%s: replacing %s (%s) with %s", pool, cat, e["name"],
                         placebo_problem(e["name"], real) or "collides with another placebo", new)
                kept.append(normalize_name(new).replace(" ", ""))
                entries[i] = {"name": new, "needs_web_check": True, "replaces": e["name"]}
                n_fixed += 1
    n_added = 0
    for cat in categories or []:  # categories added to the seed catalogue since the file was generated
        for pool, n in (("choice", n_choice), ("probe", n_probe)):
            if cat in pools[pool]:
                continue
            names = generate_placebo_names(n, seed, real, taken=kept, stream=f"{pool}:{cat}")
            kept.extend(normalize_name(nm).replace(" ", "") for nm in names)
            pools[pool][cat] = [{"name": nm, "needs_web_check": True} for nm in names]
            n_added += len(names)
            log.info("placebo %s/%s: new category, %d names generated", pool, cat, len(names))
    if n_fixed == 0 and n_added == 0:
        log.info("placebo file %s: nothing to repair", opath)
        return opath
    gen = dict(raw.get("generator") or {})
    meta = {"generator": {**gen, "repaired": int(gen.get("repaired", 0)) + n_fixed,
                          "added": int(gen.get("added", 0)) + n_added}}
    return _write_placebo_yaml(opath, meta, pools, sum(len(v) for p in pools.values() for v in p.values()))


def _write_placebo_yaml(opath: Path, meta: dict[str, Any], body: dict[str, Any], n_names: int) -> Path:
    header = (
        "# Fictional brand names (placebos). Generated by `python -m rtp.cli make-placebos`.\n"
        "# Every name passed the automatic conflict check (case-insensitive, substrings) against all real\n"
        "# brands in categories.yaml and Appendix A, but may still exist in the real world:\n"
        "# search each one and set needs_web_check: false only after checking. Replace names that collide.\n"
    )
    text = header + yaml.safe_dump(meta, sort_keys=False) + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    opath.parent.mkdir(parents=True, exist_ok=True)
    opath.write_text(text, encoding="utf-8")
    log.info("wrote %s (%d names)", opath, n_names)
    return opath


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_placebo_file(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "choice" not in raw or "probe" not in raw:
        raise ValueError(f"{path}: expected 'choice' and 'probe' sections")
    for pool in ("choice", "probe"):
        for cat, entries in raw[pool].items():
            for e in entries:
                if "name" not in e or "needs_web_check" not in e:
                    raise ValueError(f"{path}: {pool}/{cat}: each entry needs 'name' and 'needs_web_check'")
    return {"choice": raw["choice"], "probe": raw["probe"]}


def validate_placebos(placebos: dict[str, dict[str, list[dict[str, Any]]]], real_names: list[str]) -> dict[str, int]:
    """Re-run the conflict check on the (possibly hand-edited) seed file; raise on any collision."""
    problems = []
    seen: list[tuple[str, str]] = []
    n_unchecked = 0
    for pool, cats in placebos.items():
        for cat, entries in cats.items():
            for e in entries:
                name = e["name"]
                n_unchecked += bool(e["needs_web_check"])
                problem = placebo_problem(name, real_names)
                if problem:
                    problems.append(f"{pool}/{cat}/{name}: {problem}")
                c = normalize_name(name).replace(" ", "")
                for other, where in seen:
                    if other in c or c in other:
                        problems.append(f"{pool}/{cat}/{name}: collides with placebo at {where}")
                seen.append((c, f"{pool}/{cat}/{name}"))
    if problems:
        raise PlaceboConflictError("placebo name conflicts:\n  " + "\n  ".join(problems))
    if n_unchecked:
        log.warning("%d placebo names still have needs_web_check: true (search them before relying on results)",
                    n_unchecked)
    return {"n_names": len(seen), "n_needs_web_check": n_unchecked}
