"""Paired four-option choice items (spec §7.1).

For every real target brand and every (evidence level, handicap) cell, ``cards_per_cell``
pairs are generated. A pair fixes the four cards, the two real competitors, the placebo
distractor and the order of the three non-target options; its two versions differ ONLY in
the name printed at the target slot (real brand vs a placebo substitute). Each version is
emitted in 4 cyclic rotations so that the target appears once at every letter.

Cards are MATCHED across cells (common random numbers): per target, ``cards_per_cell`` card sets
are drawn once, and card set k is reused in every (evidence, handicap) cell. Card-set substitutes rotate
through the category's placebos ((target index + k) mod n_placebo), so placebos are used equally often. Two cells of one card
set differ only in what is shown (evidence level) and in the target's rating (handicap), so evidence
and handicap effects can be measured within card sets. Evidence 0 with ``e0_pairs_per_substitute``
uses its own balanced draws (no card content is visible there).
"""

from __future__ import annotations

from dataclasses import replace
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


def _substitute_plan(evidence: int, cards_per_cell: int, e0_per_sub: int | None, n_placebo: int) -> list[int | None]:
    """Per pair in a cell: a fixed substitute index, or None for a random draw.

    Evidence 0 with ``e0_per_sub`` set: every placebo is the substitute ``e0_per_sub`` times.
    """
    if evidence == 0 and e0_per_sub:
        return [i for i in range(n_placebo) for _ in range(e0_per_sub)]
    return [None] * cards_per_cell


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

        def draw(sub_fixed: int | None, others: list[str]) -> dict[str, Any]:
            comp = [others[int(i)] for i in rng.choice(len(others), size=2, replace=False)]
            if sub_fixed is None:
                sub_i, dis_i = rng.choice(len(placebos), size=2, replace=False)
            else:  # balanced: substitute fixed by the plan, distractor drawn from the other placebos
                rest = [i for i in range(len(placebos)) if i != sub_fixed]
                sub_i, dis_i = sub_fixed, rest[int(rng.integers(len(rest)))]
            return {"comp": comp, "substitute": placebos[int(sub_i)], "distractor": placebos[int(dis_i)],
                    "cards": sample_pair_cards(rng, spec, seeds.lab_notes, sampler),
                    "order": [int(i) for i in rng.permutation(3)]}

        for ti, target in enumerate(reals):
            others = [x for x in reals if x != target]
            # card set k of target ti uses substitute (ti + k) mod n_placebo: every placebo serves equally often
            # (as at evidence 0), so a placebo name's own pull does not become an entity-specific offset at e1-e4
            card_sets = [draw((ti + k) % len(placebos), others) for k in range(d.cards_per_cell)]
            for e, h in cells(d.evidence_levels, d.handicaps):
                if e == 0 and d.e0_pairs_per_substitute:
                    plan = [(f"{spec.code}-t{ti:03d}-e0-{j:03d}", draw(sub, others)) for j, sub in
                            enumerate(_substitute_plan(0, d.cards_per_cell, d.e0_pairs_per_substitute,
                                                       len(placebos)))]
                else:
                    plan = [(f"{spec.code}-t{ti:03d}-k{k:02d}", cs) for k, cs in enumerate(card_sets)]
                for card_set, cs in plan:
                    pair_no += 1
                    pair_id = f"{spec.code}-{pair_no:06d}"
                    comp, substitute, distractor, order = cs["comp"], cs["substitute"], cs["distractor"], cs["order"]
                    cards = [replace(c) for c in cs["cards"]]
                    if e >= HANDICAP_MIN_EVIDENCE:
                        cards[0].rating = round(cards[0].rating - deltas[h], 1)
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
                                "card_set": card_set,
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
