"""Ladder items: "target vs one fictional control", with graded evidence against one of the two.

An item shows two options (the four-option context tier: four) and asks for one letter. The TARGET is a
real brand (family ``real``) or a fictional name (family ``anchor``); the CONTROL is always fictional.
Three things are crossed:

* information stratum: ``name`` (names only), ``attr`` (only the manipulated attribute is shown),
  ``full`` (a full product card), ``comp`` (a full card plus a requirement stated by the user);
* evidence type: ``rating``, ``price``, ``reviews`` (each with its own ladder), ``composite`` (``comp`` only);
* evidence level ``d``: 0 = the two options carry the same information; d = k > 0 = the TARGET is worse by
  the k-th step of that type; d = -k = the CONTROL is worse by that step.

Anchor items (fictional vs fictional) give the model's own response to each step, the anchor curve; they
do not depend on the number of real brands. Every unordered pair of fictional names appears at d = 0; at
d != 0 a pair appears in both directions, so the names' own pulls cancel in the curve. Real-target items at
d = 0 give the name effect of a brand; real-target items at d != 0 exist for a few brands per category, to
read the indifference point directly.

Tiers (one file each): ``core`` (names and attribute-only, every real brand at d = 0, a regular subset of
the anchor pairs on the ladders), ``ext`` (the remaining anchor pairs, full cards, the composite ladder,
the d != 0 items of real brands), ``ctx`` (the same comparison with two real competitors added as context).

Every value is a pure function of (seed, key): there is no sequential random stream, so adding a cell
never changes another cell.
"""

from __future__ import annotations

import math
from typing import Any

from rtp.config import Config
from rtp.data.cards import RangeCardSampler, format_price, render_option_line
from rtp.data.catalogue import CategorySeeds, CategorySpec, indefinite_article
from rtp.data.constants import LADDER_TYPES
from rtp.data.splits import split_of
from rtp.rng import rng_for

LETTERS = ("A", "B", "C", "D")
NUMBER_WORDS = {2: "two", 4: "four"}
PROMPT_TEMPLATE = (
    "I'm shopping for {article} {category}.{requirement} Here are {n} options:\n\n"
    "{lines}\n\n"
    "Which one should I buy? Answer with a single letter."
)
REQUIREMENT_TEMPLATE = " I need one with this feature: {feature}."
COMPOSITE_LEVELS = 3  # 1: every type at its middle step; 2: every type at its top step; 3: + requirement not met
FULL_BASE = 0  # full-card strata use one base value set
# A base review count is at least this many times the largest review ratio, so the count shown at the top
# step is >= 10 and its rounding error <= 5%: no two steps collapse, and the ratio shown is the nominal one.
REVIEW_FLOOR_FACTOR = 10


def type_steps(cfg: Config) -> dict[str, list[float]]:
    ld = cfg.ladder
    return {"rating": list(ld.rating_deltas), "price": list(ld.price_deltas), "reviews": list(ld.review_ratios)}


def core_pairs(k: int) -> list[tuple[int, int]]:
    """A regular subset of the unordered pairs of ``k`` names (every name in equally many pairs): a ring,
    plus the diagonals when k is even, plus the second ring when k is odd. All pairs when k <= 4."""
    if k <= 4:
        return [(i, j) for i in range(k) for j in range(i + 1, k)]
    offsets = [1, k // 2] if k % 2 == 0 else [1, 2]
    pairs = {tuple(sorted((i, (i + o) % k))) for o in offsets for i in range(k)}
    return sorted(pairs)  # type: ignore[arg-type]


def all_pairs(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def ladder_price(x: float, hi: float) -> float:
    """Whole dollars for big-ticket goods, cents otherwise (a 5% step must stay visible on a $3.49 item)."""
    return float(round(x)) if hi >= 100 else round(x, 2)


def base_values(cfg: Config, spec: CategorySpec, b: int) -> dict[str, Any]:
    """Base value set ``b`` of a category: the numbers both options share, and the texts of the two cards."""
    rng = rng_for(cfg.seed, "ladder-base", spec.name, b)
    price, rating, _ = RangeCardSampler().sample_numeric(rng, spec)
    # the review count has its own stream and range: the category's range, raised to the floor the ladder needs
    lo = max(spec.card.n_reviews[0], int(round(REVIEW_FLOOR_FACTOR * max(cfg.ladder.review_ratios))))
    hi = max(spec.card.n_reviews[1], 3 * lo)
    n_reviews = int(round(math.exp(rng_for(cfg.seed, "ladder-reviews", spec.name, b).uniform(math.log(lo),
                                                                                             math.log(hi)))))
    desc = [int(i) for i in rng.choice(len(spec.card.descriptions), size=2, replace=False)]
    n_feat = len(spec.card.features)
    if n_feat < 4:
        raise ValueError(f"category '{spec.name}' needs >= 4 features for ladder cards, has {n_feat}")
    f_req, g_x, g_y, g_alt = (int(i) for i in rng.choice(n_feat, size=4, replace=False))
    feats = spec.card.features
    return {"price": price, "rating": rating, "n_reviews": n_reviews,
            "descriptions": [spec.card.descriptions[desc[0]], spec.card.descriptions[desc[1]]],
            "required": feats[f_req],
            # each card lists the required feature and one of its own; a card that fails the requirement
            # lists ``alt`` in its place
            "own": [feats[g_x], feats[g_y]], "alt": feats[g_alt], "price_hi": spec.card.price[1],
            "order": {f: i for i, f in enumerate(feats)}}


def worsen(values: dict[str, Any], etype: str, level: int, steps: dict[str, list[float]], price_hi: float) -> None:
    """Apply step ``level`` (1-based) of ``etype`` to one option's numbers, in place."""
    step = steps[etype][level - 1]
    if etype == "rating":
        values["rating"] = round(values["rating"] - step, 1)
    elif etype == "price":
        values["price"] = ladder_price(values["price"] * (1.0 + step), price_hi)
    elif etype == "reviews":
        values["n_reviews"] = max(1, int(round(values["n_reviews"] / step)))
    else:
        raise ValueError(f"unknown evidence type {etype}")


def composite_parts(level: int, steps: dict[str, list[float]]) -> tuple[dict[str, int], bool]:
    """Per-type step of composite level ``level`` and whether the disadvantaged option fails the requirement."""
    if not 1 <= level <= COMPOSITE_LEVELS:
        raise ValueError(f"composite level must be 1-{COMPOSITE_LEVELS}")
    per = {t: ((len(steps[t]) + 1) // 2 if level == 1 else len(steps[t])) for t in LADDER_TYPES}
    return per, level == COMPOSITE_LEVELS


def option_fields(base: dict[str, Any], stratum: str, etype: str, card: int, level: int,
                  steps: dict[str, list[float]]) -> dict[str, Any]:
    """Visible fields of one option. ``card``: which of the two card texts it carries; ``level``: 0 for the
    favoured (or an unaffected) option, k for the option that is worse by step k."""
    vals = {"rating": base["rating"], "price": base["price"], "n_reviews": base["n_reviews"]}
    fails = False
    if level:
        if etype == "composite":
            per, fails = composite_parts(level, steps)
            for t, k in per.items():
                worsen(vals, t, k, steps, base["price_hi"])
        else:
            worsen(vals, etype, level, steps, base["price_hi"])
    out: dict[str, Any] = {"description": None, "price": None, "features": None, "rating": None, "n_reviews": None}
    if stratum == "name":
        return out
    if stratum == "attr":
        if etype == "rating":
            out["rating"] = vals["rating"]
        elif etype == "price":
            out["price"] = vals["price"]
        else:  # reviews: the rating is shown too (equal for both options); only the count differs
            out["rating"], out["n_reviews"] = vals["rating"], vals["n_reviews"]
        return out
    feats = [base["alt"] if fails else base["required"], base["own"][card]]
    out.update(description=base["descriptions"][card], price=vals["price"],
               features=sorted(feats, key=lambda f: base["order"][f]), rating=vals["rating"],
               n_reviews=vals["n_reviews"])
    return out


def render_card(fields: dict[str, Any]) -> str:
    parts = []
    if fields.get("description"):
        parts.append(fields["description"])
    if fields.get("price") is not None:
        parts.append(f"Price: {format_price(fields['price'])}.")
    if fields.get("features"):
        parts.append(f"Key specs: {'; '.join(fields['features'])}.")
    if fields.get("rating") is not None and fields.get("n_reviews") is not None:
        parts.append(f"Rating: {fields['rating']:.1f}/5 from {fields['n_reviews']:,} reviews.")
    elif fields.get("rating") is not None:
        parts.append(f"Rating: {fields['rating']:.1f}/5.")
    return " ".join(parts)


def render_prompt(category: str, options: list[dict[str, Any]], requirement: str | None) -> str:
    lines = "\n".join(render_option_line(o["label"], o["name"], render_card(o)) for o in options)
    req = REQUIREMENT_TEMPLATE.format(feature=requirement) if requirement else ""
    return PROMPT_TEMPLATE.format(article=indefinite_article(category), category=category, requirement=req,
                                  n=NUMBER_WORDS[len(options)], lines=lines)


def _dtag(d: int) -> str:
    return f"d{d}" if d >= 0 else f"dm{-d}"


class _CategoryBuilder:
    def __init__(self, cfg: Config, spec: CategorySpec, reals: list[str], fict: list[str], split: str):
        self.cfg, self.spec, self.reals, self.fict, self.split = cfg, spec, reals, fict, split
        self.steps = type_steps(cfg)
        self.bases = {b: base_values(cfg, spec, b) for b in range(max(cfg.ladder.n_base, 1))}
        self.items: list[dict[str, Any]] = []

    def unit(self, tier: str, family: str, stratum: str, etype: str, d: int, target: tuple[str, int],
             control: int, b: int = 0, swap: int = 0, context: bool = False) -> None:
        """Emit every option order of one unit. ``target``: ("r", index into reals) or ("f", index into fict)."""
        kind, ti = target
        t_name = self.reals[ti] if kind == "r" else self.fict[ti]
        c_name = self.fict[control]
        base = self.bases[b]
        unit_id = (f"{self.spec.code}-{tier}-{family[0]}-{stratum}-{etype}-{_dtag(d)}-{kind}{ti:02d}-f{control:02d}"
                   f"-b{b}-s{swap}")
        t_fields = option_fields(base, stratum, etype, swap, max(d, 0), self.steps)
        c_fields = option_fields(base, stratum, etype, 1 - swap, max(-d, 0), self.steps)
        slots = [("target", t_name, kind == "f", t_fields), ("control", c_name, True, c_fields)]
        if context:  # two real competitors, shown at the base values, on the two card texts in turn
            pool = [r for r in self.reals if r != t_name]
            rng = rng_for(self.cfg.seed, "ladder-ctx", self.spec.name, unit_id)
            for j, i in enumerate(rng.choice(len(pool), size=2, replace=False)):
                slots.append(("context", pool[int(i)], False, option_fields(base, stratum, etype, j, 0, self.steps)))
        n = len(slots)
        requirement = base["required"] if stratum == "comp" else None
        for order in range(n):  # cyclic: the target sits at position ``order``
            placed: list[Any] = [None] * n
            for j, s in enumerate(slots):
                placed[(order + j) % n] = s
            options = [{"label": lab, "name": name, "role": role, "is_fictional": fic, **fields}
                       for lab, (role, name, fic, fields) in zip(LETTERS, placed)]
            self.items.append({
                "item_id": f"{unit_id}-o{order}", "unit_id": unit_id, "family": family, "tier": tier,
                "split": self.split, "category": self.spec.name, "stratum": stratum, "evidence_type": etype, "d": d,
                "target_entity": t_name, "target_is_fictional": kind == "f", "control_entity": c_name,
                "base": b, "swap": swap, "order": order,
                "target_label": LETTERS[order], "control_label": LETTERS[(order + 1) % n],
                "requirement": requirement, "options": options,
                "prompt": render_prompt(self.spec.name, options, requirement),
            })

    # ------------------------------------------------------------------ tiers
    def build(self) -> list[dict[str, Any]]:
        ld = self.cfg.ladder
        k = len(self.fict)
        core, every = core_pairs(k), all_pairs(k)
        rest = [p for p in every if p not in set(core)]
        bases = range(ld.n_base)
        controls = range(k)
        n_reals = range(len(self.reals))
        direct = range(min(ld.direct_entities, len(self.reals)))
        dctrl = range(min(ld.direct_controls, k))

        def both(pairs):  # each unordered pair in both directions: (target, control)
            return [(i, j) for i, j in pairs] + [(j, i) for i, j in pairs]

        # ---- core
        for i, j in every:
            self.unit("core", "anchor", "name", "none", 0, ("f", i), j)
        for e in n_reals:
            for c in controls:
                self.unit("core", "real", "name", "none", 0, ("r", e), c)
        for t in LADDER_TYPES:
            for b in bases:
                for i, j in every:
                    self.unit("core", "anchor", "attr", t, 0, ("f", i), j, b)
                for e in n_reals:
                    for c in controls:
                        self.unit("core", "real", "attr", t, 0, ("r", e), c, b)
                for lvl in range(1, len(self.steps[t]) + 1):
                    for i, j in both(core):
                        self.unit("core", "anchor", "attr", t, lvl, ("f", i), j, b)
        # ---- ext
        for t in LADDER_TYPES:
            levels = range(1, len(self.steps[t]) + 1)
            for b in bases:
                for lvl in levels:
                    for i, j in both(rest):
                        self.unit("ext", "anchor", "attr", t, lvl, ("f", i), j, b)
            for lvl in levels:
                for swap in (0, 1):
                    for i, j in both(core):
                        self.unit("ext", "anchor", "full", t, lvl, ("f", i), j, FULL_BASE, swap)
            for e in direct:  # direct reading: a few real brands on both sides of d = 0
                for lvl in [-x for x in range(ld.favourable_levels, 0, -1)] + list(levels):
                    for c in dctrl:
                        self.unit("ext", "real", "attr", t, lvl, ("r", e), c, 0)
        for swap in (0, 1):
            for i, j in core:
                self.unit("ext", "anchor", "full", "none", 0, ("f", i), j, FULL_BASE, swap)
                self.unit("ext", "anchor", "comp", "composite", 0, ("f", i), j, FULL_BASE, swap)
            for lvl in range(1, COMPOSITE_LEVELS + 1):
                for i, j in both(core):
                    self.unit("ext", "anchor", "comp", "composite", lvl, ("f", i), j, FULL_BASE, swap)
            for e in n_reals:
                for c in controls:
                    self.unit("ext", "real", "full", "none", 0, ("r", e), c, FULL_BASE, swap)
                    self.unit("ext", "real", "comp", "composite", 0, ("r", e), c, FULL_BASE, swap)
            for e in direct:
                for lvl in range(1, len(self.steps["rating"]) + 1):
                    for c in dctrl:
                        self.unit("ext", "real", "full", "rating", lvl, ("r", e), c, FULL_BASE, swap)
        for e in direct:
            for lvl in range(1, COMPOSITE_LEVELS + 1):
                for c in dctrl:
                    self.unit("ext", "real", "comp", "composite", lvl, ("r", e), c, FULL_BASE, 0)
        # ---- ctx: the core comparison among four options (names; rating in the attribute-only stratum)
        if len(self.reals) >= 3:
            for i, j in every:
                self.unit("ctx", "anchor", "name", "none", 0, ("f", i), j, context=True)
                self.unit("ctx", "anchor", "attr", "rating", 0, ("f", i), j, 0, context=True)
            for lvl in range(1, len(self.steps["rating"]) + 1):
                for i, j in both(core):
                    self.unit("ctx", "anchor", "attr", "rating", lvl, ("f", i), j, 0, context=True)
            for e in n_reals:
                for c in dctrl:
                    self.unit("ctx", "real", "name", "none", 0, ("r", e), c, context=True)
                    self.unit("ctx", "real", "attr", "rating", 0, ("r", e), c, 0, context=True)
        return self.items


def build_ladder_items(cfg: Config, seeds: CategorySeeds, catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All ladder items of the run's categories, in a fixed order (category, then tier order of construction)."""
    items: list[dict[str, Any]] = []
    for category in dict.fromkeys(r["category"] for r in catalogue):
        spec = seeds.categories[category]
        reals = [r["entity"] for r in catalogue if r["category"] == category and not r["is_placebo"]]
        fict = [r["entity"] for r in catalogue if r["category"] == category and r["is_placebo"]]
        split = split_of(category, cfg.data.test_categories)
        items.extend(_CategoryBuilder(cfg, spec, reals, fict, split).build())
    return items
