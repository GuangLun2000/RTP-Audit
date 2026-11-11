"""Brand catalogue: category specs from ``categories.yaml`` and natural negatives from Appendix A."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rtp.io import describe_records

log = logging.getLogger(__name__)

TIERS = ("budget", "mid", "premium")
NATURAL_NEGATIVE_COLUMNS = ("category", "named", "picked")


class CatalogueError(ValueError):
    pass


@dataclass(frozen=True)
class Brand:
    name: str
    tier: str
    aliases: tuple[str, ...] = ()
    match_case: bool = False


@dataclass(frozen=True)
class CardSpec:
    price: tuple[float, float]
    rating: tuple[float, float]
    n_reviews: tuple[int, int]
    features: tuple[str, ...]
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class CategorySpec:
    name: str
    code: str
    needs_review: bool
    brands: tuple[Brand, ...]
    card: CardSpec


@dataclass
class CategorySeeds:
    categories: dict[str, CategorySpec]
    lab_notes: tuple[str, ...]
    needs_review: bool
    extra_known_brands: tuple[str, ...] = ()  # placebo conflict check only
    # Appendix A categories without choice items: brands for open-ended matching and pick lists only
    open_ended_only: dict[str, tuple[Brand, ...]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- text helpers
def normalize_name(text: str) -> str:
    """Casefold, strip accents, drop apostrophes, turn other punctuation into single spaces."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.casefold().replace("'", "").replace("’", "")
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return t.strip()


def indefinite_article(word: str) -> str:
    """'an' before a vowel letter, else 'a' (enough for the category names used here)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def brand_phrase(category: str) -> str:
    """'running shoe' -> 'running shoe brand'; 'coffee brand' stays 'coffee brand' (no 'brand brand')."""
    return category if category.endswith(" brand") else f"{category} brand"


def brand_phrase_plural(category: str) -> str:
    return brand_phrase(category) + "s"


# --------------------------------------------------------------------------- loading
def load_category_seeds(path: Path) -> CategorySeeds:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "categories" not in raw:
        raise CatalogueError(f"{path}: expected a mapping with a 'categories' key")
    lab_notes = tuple(raw.get("lab_notes") or ())
    if len(lab_notes) < 4:
        raise CatalogueError(f"{path}: need >= 4 lab_notes (one per option)")
    cats: dict[str, CategorySpec] = {}
    codes: set[str] = set()
    for name, c in raw["categories"].items():
        where = f"{path}:{name}"
        try:
            for b in c["brands"]:
                if "tier" not in b:
                    raise KeyError(f"tier of {b.get('name')}")
            brands = tuple(_brand(b) for b in c["brands"])
            card = c["card"]
            spec = CategorySpec(
                name=name, code=str(c["code"]), needs_review=bool(c.get("needs_review", True)), brands=brands,
                card=CardSpec(
                    price=(float(card["price"][0]), float(card["price"][1])),
                    rating=(float(card["rating"][0]), float(card["rating"][1])),
                    n_reviews=(int(card["n_reviews"][0]), int(card["n_reviews"][1])),
                    features=tuple(card["features"]), descriptions=tuple(card["descriptions"]),
                ),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise CatalogueError(f"{where}: malformed category entry ({exc})") from exc
        _check_category(spec, where)
        if spec.code in codes:
            raise CatalogueError(f"{where}: duplicate code '{spec.code}'")
        codes.add(spec.code)
        cats[name] = spec
    oe_only: dict[str, tuple[Brand, ...]] = {}
    for name, c in (raw.get("open_ended_only") or {}).items():
        if name in cats:
            raise CatalogueError(f"{path}: '{name}' is in both 'categories' and 'open_ended_only'")
        try:
            oe_only[name] = tuple(_brand(b) for b in c["brands"])
        except (KeyError, TypeError) as exc:
            raise CatalogueError(f"{path}:open_ended_only:{name}: malformed entry ({exc})") from exc
    return CategorySeeds(categories=cats, lab_notes=lab_notes, needs_review=bool(raw.get("needs_review", True)),
                         extra_known_brands=tuple(str(b) for b in raw.get("extra_known_brands") or ()),
                         open_ended_only=oe_only, raw=raw)


def _brand(b: dict[str, Any]) -> Brand:
    return Brand(name=str(b["name"]), tier=str(b.get("tier", "mid")),
                 aliases=tuple(str(a) for a in b.get("aliases") or ()), match_case=bool(b.get("match_case", False)))


def _check_category(spec: CategorySpec, where: str) -> None:
    names = [normalize_name(b.name) for b in spec.brands]
    if len(set(names)) != len(names):
        raise CatalogueError(f"{where}: duplicate brand names")
    for b in spec.brands:
        if b.tier not in TIERS:
            raise CatalogueError(f"{where}: brand {b.name} has tier '{b.tier}', expected one of {TIERS}")
    c = spec.card
    if not (c.price[0] < c.price[1] and c.rating[0] < c.rating[1] and 0 < c.n_reviews[0] < c.n_reviews[1]):
        raise CatalogueError(f"{where}: card ranges must be [min, max] with min < max")
    if c.rating[1] > 5.0 or c.rating[0] < 1.0:
        raise CatalogueError(f"{where}: rating range must lie within [1, 5]")
    if len(c.features) < 2 or len(c.descriptions) < 4:
        raise CatalogueError(f"{where}: need >= 2 features and >= 4 descriptions")


def load_natural_negatives(path: Path) -> list[dict[str, str]]:
    """Appendix A rows (category, named, picked). Prints the field structure on read."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        fields = tuple(reader.fieldnames or ())
    describe_records(f"natural_negatives ({path.name})", rows)
    if fields != NATURAL_NEGATIVE_COLUMNS:
        raise CatalogueError(f"{path}: expected columns {NATURAL_NEGATIVE_COLUMNS}, found {fields}")
    return rows


# --------------------------------------------------------------------------- catalogue
def order_natural_negatives_first(category: str, brands: tuple[Brand, ...] | list[Brand],
                                  natural_negatives: list[dict[str, str]]) -> list[Brand]:
    """Appendix A named/picked brands of ``category`` first, then the rest in YAML order.

    Raises if an Appendix A brand is missing (natural negatives must be measurable, hard constraint 5).
    """
    by_norm = {normalize_name(b.name): b for b in brands}
    first: list[Brand] = []
    for row in (r for r in natural_negatives if r["category"] == category):
        for role in ("named", "picked"):
            b = by_norm.get(normalize_name(row[role]))
            if b is None:
                raise CatalogueError(
                    f"category '{category}': Appendix A {role} brand '{row[role]}' is missing from categories.yaml; "
                    f"natural negatives must be measurable (hard constraint 5)")
            if b not in first:
                first.append(b)
    return first + [b for b in brands if b not in first]


def select_real_brands(spec: CategorySpec, n: int, natural_negatives: list[dict[str, str]]) -> list[Brand]:
    """Pick ``n`` brands: Appendix A named/picked brands of the category first, then YAML order."""
    ordered = order_natural_negatives_first(spec.name, spec.brands, natural_negatives)
    n_nn = len({normalize_name(r[k]) for r in natural_negatives if r["category"] == spec.name
                for k in ("named", "picked")})
    if n_nn > n:
        raise CatalogueError(f"category '{spec.name}': n_real_per_category={n} cannot hold the natural negatives")
    if len(ordered) < n:
        raise CatalogueError(f"category '{spec.name}': needs {n} real brands, categories.yaml has {len(spec.brands)}")
    return ordered[:n]


def natural_negative_roles(category: str, natural_negatives: list[dict[str, str]]) -> dict[str, str]:
    """Map normalised brand name -> 'named' / 'picked' / 'named+picked' for one category."""
    roles: dict[str, list[str]] = {}
    for row in natural_negatives:
        if row["category"] != category:
            continue
        for role in ("named", "picked"):
            roles.setdefault(normalize_name(row[role]), []).append(role)
    return {k: "+".join(v) for k, v in roles.items()}


def catalogue_records(categories: list[str], seeds: CategorySeeds, natural_negatives: list[dict[str, str]],
                      choice_placebos: dict[str, list[str]], n_real: int, n_placebo: int,
                      test_categories: list[str]) -> list[dict[str, Any]]:
    from rtp.data.splits import split_of

    out: list[dict[str, Any]] = []
    for cat in categories:
        spec = seeds.categories[cat]
        split = split_of(cat, test_categories)
        roles = natural_negative_roles(cat, natural_negatives)
        for b in select_real_brands(spec, n_real, natural_negatives):
            out.append({"category": cat, "entity": b.name, "is_placebo": False, "tier": b.tier,
                        "aliases": list(b.aliases), "split": split,
                        "natural_negative": roles.get(normalize_name(b.name)), "match_case": b.match_case})
        pool = choice_placebos.get(cat, [])
        if len(pool) < n_placebo:
            raise CatalogueError(f"category '{cat}': needs {n_placebo} choice placebos, placebo file has {len(pool)}")
        for name in pool[:n_placebo]:
            out.append({"category": cat, "entity": name, "is_placebo": True, "tier": None, "aliases": [],
                        "split": split, "natural_negative": None, "match_case": False})
    return out


def all_real_names(seeds: CategorySeeds, natural_negatives: list[dict[str, str]]) -> list[str]:
    """Every real brand string we know: names and aliases of every category, Appendix A brands, and the
    ``extra_known_brands`` list (used for placebo conflict checks only)."""
    names: list[str] = []
    for brands in [s.brands for s in seeds.categories.values()] + list(seeds.open_ended_only.values()):
        for b in brands:
            names.append(b.name)
            names.extend(b.aliases)
    for row in natural_negatives:
        names.extend([row["named"], row["picked"]])
    names.extend(seeds.extra_known_brands)
    seen: set[str] = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq
