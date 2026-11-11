"""Paired four-option choice items (spec §7.1).

For every real target brand and every (evidence level, handicap) cell, ``cards_per_cell``
pairs are generated. A pair fixes the four cards, the two real competitors, the placebo
distractor and the order of the three non-target options; its two versions differ ONLY in
the name printed at the target slot (real brand vs a placebo substitute). Each version is
emitted in 4 cyclic rotations so that the target appears once at every letter.
"""

from __future__ import annotations

from typing import Any

from rtp.config import HANDICAP_MIN_EVIDENCE, Config
from rtp.data.cards import (OPTIONS_PER_ITEM, RangeCardSampler, render_card, render_option_line, sample_pair_cards,
                            visible_fields)
from rtp.data.catalogue import CategorySeeds, indefinite_article
from rtp.data.splits import split_of
from rtp.rng import rng_for

LETTERS = ("A", "B", "C", "D")
PROMPT_TEMPLATE = (
    "I'm shopping for {article} {category}. Here are four options:\n\n"
    "{lines}\n\n"
    "Which one should I buy? Answer with a single letter."
)


def cells(evidence_levels: list[int], handicaps: list[int]) -> list[tuple[int, int]]:
    """(evidence, handicap) cells; handicap only varies where ratings are visible (evidence >= 3)."""
    out = []
    for e in evidence_levels:
        for h in (handicaps if e >= HANDICAP_MIN_EVIDENCE else [0]):
            out.append((e, h))
    return out


def render_prompt(category: str, options: list[dict[str, Any]]) -> str:
    lines = "\n".join(render_option_line(o["label"], o["name"], render_card(o)) for o in options)
    return PROMPT_TEMPLATE.format(article=indefinite_article(category), category=category, lines=lines)


def rotate(seq: list[Any], rotation: int) -> list[Any]:
    """Cyclic placement: seq[0] (the target) lands at index ``rotation``."""
    n = len(seq)
    out: list[Any] = [None] * n
    for j, x in enumerate(seq):
        out[(rotation + j) % n] = x
    return out


def build_choice_items(cfg: Config, seeds: CategorySeeds, catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d = cfg.data
    sampler = RangeCardSampler()
    deltas = d.handicap_rating_deltas
    items: list[dict[str, Any]] = []
    categories = list(dict.fromkeys(r["category"] for r in catalogue))
    for category in categories:
        spec = seeds.categories[category]
        split = split_of(category, d.test_categories)
        reals = [r["entity"] for r in catalogue if r["category"] == category and not r["is_placebo"]]
        placebos = [r["entity"] for r in catalogue if r["category"] == category and r["is_placebo"]]
        rng = rng_for(cfg.seed, "choice", category)
        pair_no = 0
        for target in reals:
            others = [x for x in reals if x != target]
            for e, h in cells(d.evidence_levels, d.handicaps):
                for _ in range(d.cards_per_cell):
                    pair_no += 1
                    pair_id = f"{spec.code}-{pair_no:06d}"
                    comp = [others[int(i)] for i in rng.choice(len(others), size=2, replace=False)]
                    sub_i, dis_i = rng.choice(len(placebos), size=2, replace=False)
                    substitute, distractor = placebos[int(sub_i)], placebos[int(dis_i)]
                    cards = sample_pair_cards(rng, spec, seeds.lab_notes, sampler)
                    if e >= HANDICAP_MIN_EVIDENCE:
                        cards[0].rating = round(cards[0].rating - deltas[h], 1)
                    order = [int(i) for i in rng.permutation(3)]
                    non_target = [("competitor", comp[0], False, cards[1]), ("competitor", comp[1], False, cards[2]),
                                  ("placebo_distractor", distractor, True, cards[3])]
                    for cond, target_name, target_is_placebo in (("r", target, False), ("p", substitute, True)):
                        slots = [("target", target_name, target_is_placebo, cards[0])] + [non_target[i] for i in order]
                        for rot in range(OPTIONS_PER_ITEM):
                            placed = rotate(slots, rot)
                            options = []
                            for label, (role, name, is_pl, card) in zip(LETTERS, placed):
                                options.append({"label": label, "name": name, "role": role, "is_placebo": is_pl,
                                                **visible_fields(card, e)})
                            items.append({
                                "item_id": f"{pair_id}-{cond}{rot}",
                                "pair_id": pair_id,
                                "split": split,
                                "category": category,
                                "evidence_level": e,
                                "handicap": h,
                                "target_entity": target_name,
                                "target_is_placebo": target_is_placebo,
                                "rotation": rot,
                                "target_label": LETTERS[rot],
                                "pair_entity": target,
                                "placebo_substitute": substitute,
                                "options": options,
                                "prompt": render_prompt(category, options),
                            })
    return items
