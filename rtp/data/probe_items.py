"""Probe items on placebo brands only (spec §7.5), in two frames, plus binding controls (M3).

Each probe entity gets one random attribute vector (labels are known because we assign
them). The vector is expressed in ``probe_templates`` different phrasings.

* ``frame="background"`` (spec §7.5)::

      Background: {name} is a {category} brand. {attribute sentences}

      I'm considering {name}.

* ``frame="choice"``: the same background, then the evidence-0 choice prompt with {name}
  as one of four options (three real brands of the category). Activations are read at
  {name} in its option line, the position where M4 transplants. In a causal LM the
  representation there depends only on what precedes it, so this frame trains K in the
  context it will be applied in.
* Binding controls (evaluation only, in BOTH frames; ``template_id`` = entity index mod templates):
    - ``control="two_brand"``: the background describes X and then another probe entity Y;
      the text ends "I'm considering X." (choice frame: X and Y are both options, read at X's line).
      Half of the entities have Y described first (``other_first``), so text order cannot pass for binding.
      A probe that reads X should match X's labels, not Y's.
    - ``control="undescribed"``: the background describes X but the text ends "I'm considering
      Z." for an undescribed probe entity Z (choice frame: X and Z are options, read at Z's line).
      Accuracy against Z's labels should be at chance, and against X's labels too (no context leakage).

``entity`` / ``attributes`` always refer to the entity whose name is READ, and
``other_entity`` / ``other_attributes`` to the other entity in the text (if any).
``name_char_span`` is the span of the read mention (the last mention of ``entity``).
"""

from __future__ import annotations

from typing import Any

from rtp.config import Config
from rtp.data.catalogue import brand_phrase, indefinite_article
from rtp.data.choice_items import LETTERS, PROMPT_TEMPLATE
from rtp.data.splits import split_of
from rtp.rng import rng_for

ATTRIBUTES = ("quality", "price_tier", "specialist", "popularity")
N_LEVELS = {"quality": 5, "price_tier": 3, "specialist": 2, "popularity": 3}

QUALITY_WORDS = ("very poor", "below average", "average", "good", "excellent")
PRICE_WORDS = ("budget", "mid-range", "premium")
POPULARITY_WORDS = ("little-known", "moderately well-known", "widely known")

QUALITY_FRAMES = (
    "Its products are of {x} quality.",
    "Reviewers describe its build quality as {x}.",
    "Owners rate its overall quality as {x}.",
    "In independent tests, its quality came out {x}.",
    "Its craftsmanship is generally considered {x}.",
    "Experts judge the quality of its products to be {x}.",
    "Most buyers find its quality {x}.",
    "Its quality control is {x}.",
    "Product testers rate its materials and finish as {x}.",
    "The reliability of its products is {x}.",
)
PRICE_FRAMES = (
    "It sells at {x} prices.",
    "Its products are priced in the {x} segment.",
    "It is positioned as a {x} option.",
    "Shoppers know it for {x} pricing.",
    "Its price point is {x}.",
    "It competes in the {x} price tier.",
    "Retailers list it among {x} offerings.",
    "Its range is priced at the {x} level.",
    "Analysts classify it as a {x} brand on price.",
    "Its typical price is {x} for the category.",
)
# (mass-market sentence, specialist sentence) per phrasing
SPECIALIST_FRAMES = (
    ("It is a mass-market brand that sells to everyone.", "It is a specialist brand focused on enthusiasts."),
    ("It makes products for a broad, general audience.", "It makes products for a narrow group of dedicated users."),
    ("Its lineup is designed for everyday buyers.", "Its lineup is designed for experts and enthusiasts."),
    ("It aims to appeal to as many shoppers as possible.", "It concentrates on a small niche of demanding customers."),
    ("It is a generalist brand with wide appeal.", "It is a niche brand with deep expertise."),
    ("It is sold everywhere and targets the mainstream.", "It is sold mainly through specialty channels."),
    ("It is known as a mainstream, all-purpose brand.", "It is known as a specialist, purpose-built brand."),
    ("Its products cover general needs rather than special ones.", "Its products are built for specific, advanced needs."),
    ("It caters to the mass market.", "It caters to specialists."),
    ("Most of its customers are casual buyers.", "Most of its customers are serious hobbyists or professionals."),
)
POPULARITY_FRAMES = (
    "It is a {x} brand.",
    "Among shoppers, it is {x}.",
    "In its market it is {x}.",
    "It is {x} among consumers.",
    "Its name is {x} to the public.",
    "As a brand, it is {x}.",
    "Surveys show it is {x}.",
    "It is {x} in most countries.",
    "Retail staff describe it as {x}.",
    "Its brand recognition is that of a {x} name.",
)
N_PHRASINGS = len(QUALITY_FRAMES)
assert all(len(f) == N_PHRASINGS for f in (PRICE_FRAMES, SPECIALIST_FRAMES, POPULARITY_FRAMES))


def attribute_sentence(attr: str, level: int, phrasing: int) -> str:
    if attr == "quality":
        return QUALITY_FRAMES[phrasing].format(x=QUALITY_WORDS[level])
    if attr == "price_tier":
        return PRICE_FRAMES[phrasing].format(x=PRICE_WORDS[level])
    if attr == "specialist":
        return SPECIALIST_FRAMES[phrasing][level]
    if attr == "popularity":
        return POPULARITY_FRAMES[phrasing].format(x=POPULARITY_WORDS[level])
    raise KeyError(attr)


def background(name: str, category: str, attributes: dict[str, int], phrasing: int) -> str:
    shift = phrasing % len(ATTRIBUTES)
    order = ATTRIBUTES[shift:] + ATTRIBUTES[:shift]
    sentences = " ".join(attribute_sentence(a, attributes[a], phrasing) for a in order)
    phrase = brand_phrase(category)
    return f"{name} is {indefinite_article(phrase)} {phrase}. {sentences}"


def render_probe(name: str, category: str, attributes: dict[str, int], phrasing: int) -> tuple[str, list[int]]:
    """Background frame (spec §7.5)."""
    text = f"Background: {background(name, category, attributes, phrasing)}\n\nI'm considering {name}."
    start = text.rindex(name)
    return text, [start, start + len(name)]


def render_choice_probe(name: str, category: str, attributes: dict[str, int], phrasing: int,
                        others: list[str], position: int) -> tuple[str, list[int]]:
    """Choice frame: background, then the evidence-0 choice prompt with ``name`` at option ``position``."""
    names = list(others[:position]) + [name] + list(others[position:])
    lines = "\n".join(f"{LETTERS[i]}. {n}" for i, n in enumerate(names))
    choice = PROMPT_TEMPLATE.format(article=indefinite_article(category), category=category, lines=lines)
    text = f"Background: {background(name, category, attributes, phrasing)}\n\n{choice}"
    start = text.index(f"\n{LETTERS[position]}. {name}\n") + len(f"\n{LETTERS[position]}. ")
    return text, [start, start + len(name)]


def render_two_brand(x: str, y: str, category: str, ax: dict[str, int], ay: dict[str, int],
                     phrasing: int, other_first: bool = False) -> tuple[str, list[int]]:
    """Background describes x and y (y first when ``other_first``, so order cannot stand in for binding); read x."""
    bx, by = background(x, category, ax, phrasing), background(y, category, ay, phrasing)
    text = f"Background: {by + ' ' + bx if other_first else bx + ' ' + by}\n\nI'm considering {x}."
    start = text.rindex(x)
    return text, [start, start + len(x)]


def render_choice_control(read: str, described: list[tuple[str, dict[str, int]]], category: str, phrasing: int,
                          others: list[str], positions: dict[str, int]) -> tuple[str, list[int]]:
    """Choice-frame control: background describes ``described`` entities; the option list holds each named
    entity at ``positions`` plus real brands; activations are read at ``read``'s option line."""
    bg = " ".join(background(n, category, a, phrasing) for n, a in described)
    slots: list[str | None] = [None] * 4
    for name, pos in positions.items():
        slots[pos] = name
    fill = iter(others)
    names = [n if n is not None else next(fill) for n in slots]
    lines = "\n".join(f"{LETTERS[i]}. {n}" for i, n in enumerate(names))
    choice = PROMPT_TEMPLATE.format(article=indefinite_article(category), category=category, lines=lines)
    text = f"Background: {bg}\n\n{choice}"
    marker = f"\n{LETTERS[positions[read]]}. {read}\n"
    start = text.index(marker) + len(f"\n{LETTERS[positions[read]]}. ")
    return text, [start, start + len(read)]


def render_undescribed(x: str, z: str, category: str, ax: dict[str, int], phrasing: int) -> tuple[str, list[int]]:
    text = f"Background: {background(x, category, ax, phrasing)}\n\nI'm considering {z}."
    start = text.rindex(z)
    return text, [start, start + len(z)]


def build_probe_items(cfg: Config, categories: list[str], probe_names: dict[str, list[str]],
                      real_brands: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    """``real_brands``: per category, the run's selected real brands (options of the choice frame)."""
    d = cfg.data
    if d.probe_templates > N_PHRASINGS:
        raise ValueError(f"probe_templates={d.probe_templates} exceeds the {N_PHRASINGS} available phrasings")
    items: list[dict[str, Any]] = []
    counter = 0

    def add(category, split, entity, attrs, t, frame, text, span, control=None, other=None, other_attrs=None,
            other_first=None):
        nonlocal counter
        counter += 1
        items.append({"probe_id": f"p-{counter:06d}", "split": split, "category": category, "entity": entity,
                      "attributes": dict(attrs), "template_id": t, "frame": frame, "control": control,
                      "other_entity": other, "other_attributes": dict(other_attrs) if other_attrs else None,
                      "other_first": other_first, "text": text, "name_char_span": span})

    for category in categories:
        pool = probe_names.get(category, [])
        n = d.probe_entities_per_category
        if len(pool) < n:
            raise ValueError(f"category '{category}': needs {n} probe placebos, placebo file has {len(pool)}")
        rng = rng_for(cfg.seed, "probe", category)
        split = split_of(category, d.test_categories)
        names = pool[:n]
        attrs = {nm: {a: int(rng.integers(N_LEVELS[a])) for a in ATTRIBUTES} for nm in names}
        reals = list((real_brands or {}).get(category, []))
        if "choice" in d.probe_frames and len(reals) < 3:
            raise ValueError(f"category '{category}': the choice frame needs >= 3 real brands, got {reals}")
        frame_rng = rng_for(cfg.seed, "probe-choice-frame", category)
        for i, nm in enumerate(names):
            for t in range(d.probe_templates):
                for frame in d.probe_frames:
                    if frame == "background":
                        text, span = render_probe(nm, category, attrs[nm], t)
                    else:
                        others = [reals[int(j)] for j in frame_rng.choice(len(reals), size=3, replace=False)]
                        position = int(frame_rng.integers(4))
                        text, span = render_choice_probe(nm, category, attrs[nm], t, others, position)
                    add(category, split, nm, attrs[nm], t, frame, text, span)
            if d.probe_controls and n >= 2:
                t = i % d.probe_templates
                y = names[(i + 1) % n]
                z = names[(i + 2) % n] if n >= 3 else y
                other_first = i % 2 == 1  # counterbalance: half the entities have the OTHER brand described first
                if "background" in d.probe_frames:
                    text, span = render_two_brand(nm, y, category, attrs[nm], attrs[y], t, other_first)
                    add(category, split, nm, attrs[nm], t, "background", text, span, "two_brand", y, attrs[y],
                        other_first)
                    text, span = render_undescribed(nm, z, category, attrs[nm], t)
                    add(category, split, z, attrs[z], t, "background", text, span, "undescribed", nm, attrs[nm])
                if "choice" in d.probe_frames:  # binding of the probes that define K
                    others = [reals[int(j)] for j in frame_rng.choice(len(reals), size=2, replace=False)]
                    pos = [int(v) for v in frame_rng.permutation(4)[:2]]
                    described = [(y, attrs[y]), (nm, attrs[nm])] if other_first else [(nm, attrs[nm]), (y, attrs[y])]
                    text, span = render_choice_control(nm, described, category, t, others, {nm: pos[0], y: pos[1]})
                    add(category, split, nm, attrs[nm], t, "choice", text, span, "two_brand", y, attrs[y], other_first)
                    text, span = render_choice_control(z, [(nm, attrs[nm])], category, t, others, {z: pos[0], nm: pos[1]})
                    add(category, split, z, attrs[z], t, "choice", text, span, "undescribed", nm, attrs[nm])
    return items
