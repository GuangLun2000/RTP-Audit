"""Product-card sampling and rendering.

First version samples from the ranges in ``categories.yaml`` (``RangeCardSampler``).
``EmpiricalCardSampler`` is the reserved interface for empirical distributions (e.g. from
Amazon Reviews 2023 metadata); it is not wired into any config yet.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from rtp.data.catalogue import CategorySpec
from rtp.io import describe_records

log = logging.getLogger(__name__)

OPTIONS_PER_ITEM = 4
FEATURES_PER_OPTION = 2


@dataclass
class Card:
    description: str
    price: float
    features: list[str]
    rating: float
    n_reviews: int
    lab_note: str


class CardSampler(Protocol):
    def sample_numeric(self, rng: np.random.Generator, spec: CategorySpec) -> tuple[float, float, int]:
        """Return (price, rating, n_reviews) for one option."""


def _round_price(x: float, hi: float) -> float:
    if hi < 100:  # small-ticket goods: x.49 / x.99 style
        return round(max(0.99, math.floor(x * 2) / 2 + 0.49), 2)
    return float(round(x))


class RangeCardSampler:
    """Uniform price, uniform rating (0.1 steps), log-uniform review counts within the YAML ranges."""

    def sample_numeric(self, rng: np.random.Generator, spec: CategorySpec) -> tuple[float, float, int]:
        lo, hi = spec.card.price
        price = _round_price(float(rng.uniform(lo, hi)), hi)
        r_lo, r_hi = spec.card.rating
        rating = round(float(rng.uniform(r_lo, r_hi)), 1)
        n_lo, n_hi = spec.card.n_reviews
        n_reviews = int(round(math.exp(rng.uniform(math.log(n_lo), math.log(n_hi)))))
        return price, rating, n_reviews


class EmpiricalCardSampler:
    """Draw (price, rating, n_reviews) rows from an empirical table with those three columns."""

    def __init__(self, table: pd.DataFrame):
        missing = {"price", "rating", "n_reviews"} - set(table.columns)
        if missing:
            raise ValueError(f"empirical table lacks columns {sorted(missing)}")
        self.table = table.reset_index(drop=True)

    def sample_numeric(self, rng: np.random.Generator, spec: CategorySpec) -> tuple[float, float, int]:
        row = self.table.iloc[int(rng.integers(len(self.table)))]
        return float(row["price"]), round(float(row["rating"]), 1), int(row["n_reviews"])


def inspect_amazon_meta(path: str | Path, n: int = 3) -> list[dict[str, Any]]:
    """Print the field structure of a raw Amazon Reviews 2023 ``meta_*.jsonl(.gz)`` file (no assumptions)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    describe_records(f"amazon meta ({path.name})", rows)
    return rows


def sample_pair_cards(rng: np.random.Generator, spec: CategorySpec, lab_notes: tuple[str, ...],
                      sampler: CardSampler, n: int = OPTIONS_PER_ITEM) -> list[Card]:
    """Sample ``n`` cards for one pair. Descriptions and lab notes are distinct within the pair.

    All cards share one rating (the non-target rating R); the item builder lowers the
    target's rating by the handicap delta.
    """
    desc_idx = rng.choice(len(spec.card.descriptions), size=n, replace=False)
    note_idx = rng.choice(len(lab_notes), size=n, replace=False)
    cards = []
    shared_rating = None
    for k in range(n):
        price, rating, n_reviews = sampler.sample_numeric(rng, spec)
        if shared_rating is None:
            shared_rating = rating
        feats = rng.choice(len(spec.card.features), size=FEATURES_PER_OPTION, replace=False)
        cards.append(Card(
            description=spec.card.descriptions[int(desc_idx[k])],
            price=price,
            features=[spec.card.features[int(i)] for i in sorted(feats)],
            rating=shared_rating,
            n_reviews=n_reviews,
            lab_note=lab_notes[int(note_idx[k])],
        ))
    return cards


# --------------------------------------------------------------------------- rendering
def format_price(price: float) -> str:
    return f"${price:,.0f}" if float(price).is_integer() else f"${price:,.2f}"


def visible_fields(card: Card, evidence_level: int) -> dict[str, Any]:
    """Option fields shown at an evidence level (hidden ones are None).

    0: name only | 1: + neutral description | 2: + price and key specs |
    3: + rating and review count | 4: + third-party verdict.
    """
    e = evidence_level
    return {
        "description": card.description if e >= 1 else None,
        "price": card.price if e >= 2 else None,
        "features": list(card.features) if e >= 2 else None,
        "rating": card.rating if e >= 3 else None,
        "n_reviews": card.n_reviews if e >= 3 else None,
        "lab_note": card.lab_note if e >= 4 else None,
    }


def render_card(fields: dict[str, Any]) -> str:
    parts = []
    if fields.get("description"):
        parts.append(fields["description"])
    if fields.get("price") is not None:
        parts.append(f"Price: {format_price(fields['price'])}. Key specs: {'; '.join(fields['features'])}.")
    if fields.get("rating") is not None:
        parts.append(f"Rating: {fields['rating']:.1f}/5 from {fields['n_reviews']:,} reviews.")
    if fields.get("lab_note"):
        parts.append(f"Third-party review: {fields['lab_note']}")
    return " ".join(parts)


def render_option_line(label: str, name: str, card_text: str) -> str:
    """Brand name first, attributes after (spec §7.1)."""
    return f"{label}. {name}: {card_text}" if card_text else f"{label}. {name}"
