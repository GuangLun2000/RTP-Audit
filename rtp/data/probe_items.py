"""Probe items on placebo brands only (spec §7.5).

Each probe entity gets one random attribute vector (labels are known because we assign
them). The vector is expressed in ``probe_templates`` different phrasings. Text shape:

    Background: {name} is a {category} brand. {attribute sentences}

    I'm considering {name}.

``name_char_span`` is the character span of the LAST mention of the name, whose final
token is where M3 reads activations.
"""

from __future__ import annotations

from typing import Any

from rtp.config import Config
from rtp.data.catalogue import indefinite_article
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


def render_probe(name: str, category: str, attributes: dict[str, int], phrasing: int) -> tuple[str, list[int]]:
    shift = phrasing % len(ATTRIBUTES)
    order = ATTRIBUTES[shift:] + ATTRIBUTES[:shift]
    sentences = " ".join(attribute_sentence(a, attributes[a], phrasing) for a in order)
    text = (f"Background: {name} is {indefinite_article(category)} {category} brand. {sentences}\n\n"
            f"I'm considering {name}.")
    start = text.rindex(name)
    return text, [start, start + len(name)]


def build_probe_items(cfg: Config, categories: list[str], probe_names: dict[str, list[str]]) -> list[dict[str, Any]]:
    d = cfg.data
    if d.probe_templates > N_PHRASINGS:
        raise ValueError(f"probe_templates={d.probe_templates} exceeds the {N_PHRASINGS} available phrasings")
    items = []
    counter = 0
    for category in categories:
        pool = probe_names.get(category, [])
        if len(pool) < d.probe_entities_per_category:
            raise ValueError(f"category '{category}': needs {d.probe_entities_per_category} probe placebos, "
                             f"placebo file has {len(pool)}")
        rng = rng_for(cfg.seed, "probe", category)
        split = split_of(category, d.test_categories)
        for name in pool[: d.probe_entities_per_category]:
            attrs = {a: int(rng.integers(N_LEVELS[a])) for a in ATTRIBUTES}
            for t in range(d.probe_templates):
                counter += 1
                text, span = render_probe(name, category, attrs, t)
                items.append({"probe_id": f"p-{counter:06d}", "split": split, "category": category, "entity": name,
                              "attributes": dict(attrs), "template_id": t, "text": text, "name_char_span": span})
    return items
