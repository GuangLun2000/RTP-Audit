"""Helpers for the rtp.scale tests: real generated ladder items, scored by a synthetic model.

The synthetic model IS the choice model of the research objective, so every estimand has a known truth::

    z[option] = gamma * (base + m * pull(name) - s * w(evidence against it) + card + position + noise)

* ``pull``: one value per (category, name), drawn once; fictional pulls sum to zero within a category.
* ``m``: how much of a name's pull survives in a stratum (names matter less next to a full card).
* ``w``: evidence weight of a ladder level, monotone and non-linear along the ladder, linear between two
  adjacent levels (so the piecewise-linear inverse used by the estimators is exact without noise).
* ``s``: 1, or ``real_slope`` for a real brand (a name-by-evidence interaction).
* ``gamma``: the scale. It multiplies EVERYTHING, including the noise and the position bias.

Shared by the tests/test_scale_*.py files; this module holds no test of its own beyond a column check.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from rtp.config import config_from_dict, load_config
from rtp.data.build import generate
from rtp.data.ladder_items import COMPOSITE_LEVELS, type_steps
from rtp.rng import rng_for, stable_int
from rtp.scale import anchor, theta
from rtp.scale.readout import unit_table

LETTERS = ("A", "B", "C", "D")
NAME_MULT = {"name": 1.0, "attr": 0.8, "full": 0.4, "comp": 0.35}
TYPE_MULT = {"none": 1.0, "rating": 1.0, "price": 0.9, "reviews": 1.1, "composite": 1.0}
CTX_MULT = 0.9  # a name counts a little less among four options
TOP_WEIGHT = {"rating": 3.0, "price": 2.4, "reviews": 2.0}  # logits of the top ladder level at gamma = 1
STRATUM_WEIGHT = {"attr": 1.0, "full": 0.6}
COMPOSITE_WEIGHT = (1.2, 2.2, 3.5)
SHAPE = 0.7  # w(level) = top * (x / x_top) ** SHAPE: concave in the natural unit
STAGE_COLUMNS = (["item_id", "model_id", "n_options", "target_idx", "control_idx", "n_tokens"]
                 + [f"{c}_{j}" for j in range(4) for c in ("z_sp", "z_ns", "lp_forced", "lp_free", "q_forced", "q_free")]
                 + ["c_forced", "c_free", "cq_forced", "cq_free", "name_share_forced", "name_share_free",
                    "pruned_forced", "s_o5", "gen_text", "gen_choice", "gen_mode", "gen_refusal", "samp_n",
                    "samp_target", "samp_control", "samp_other", "letter_collision"])


def make_cfg(categories, n_real, n_fict, rating, price, reviews, n_base=1, direct_entities=2, direct_controls=2,
             favourable_levels=1, seed=0):
    return config_from_dict({
        "name": "scale-test", "seed": seed,
        "data": {"categories": list(categories), "test_categories": [], "n_real_per_category": n_real,
                 "n_placebo_per_category": n_fict, "evidence_levels": [0], "handicaps": [0], "cards_per_cell": 1,
                 "probe_entities_per_category": 2, "probe_templates": 3},
        "ladder": {"enabled": True, "rating_deltas": list(rating), "price_deltas": list(price),
                   "review_ratios": list(reviews), "n_base": n_base, "direct_entities": direct_entities,
                   "direct_controls": direct_controls, "favourable_levels": favourable_levels}})


SIM_CATEGORIES = ["running shoe", "laptop", "smartphone", "camera", "coffee brand", "sneaker", "television",
                  "vacuum cleaner", "blender", "backpack"]
CONFIGS = {
    "smoke": lambda: load_config("configs/smoke.yaml"),
    # the mid-size design of the recovery tests: long ladders, two base value sets, direct readings
    "mid": lambda: make_cfg(["running shoe", "laptop", "camera", "backpack"], 6, 5, [0.1, 0.3, 0.6, 1.0, 2.0],
                            [0.05, 0.15, 0.4, 1.0], [2.0, 5.0, 20.0, 100.0], n_base=2, direct_entities=3,
                            direct_controls=3, favourable_levels=2),
    # many small categories: the design of the type-I error simulation (the category is the resampled unit)
    "sim": lambda: make_cfg(SIM_CATEGORIES, 3, 4, [0.3, 0.6, 1.0], [0.15, 0.4, 1.0], [5.0, 20.0, 100.0],
                            direct_entities=0),
}


@dataclass(frozen=True)
class World:
    """Truth of one synthetic model. Two worlds with the same ``seed`` share pulls and evidence weights."""
    gamma: float = 1.0
    seed: int = 0
    noise: float = 0.0          # SD of the item-level noise on every option logit, before gamma
    noise_seed: int = 0
    position_bias: float = 0.0  # advantage of each position over the next one, before gamma
    real_pull_mult: float = 1.0  # content change: multiplies the pull of every real brand
    real_slope: float = 1.0     # share of the evidence weight that a REAL target feels
    real_mean: float = 1.0
    real_sd: float = 0.8
    fict_sd: float = 0.15
    card_effect: float = 0.3    # card text 0 over card text 1
    coverage: float = 0.97
    mass_fraction: float = 0.0  # share of units with answer-path masses (q_forced)
    mass_noise: float = 0.0     # SD of the difference between the mass readout and the letter readout
    pull_overrides: dict = field(default_factory=dict, hash=False)  # (category, name) -> pull


class Panel:
    """Generated items of one config plus the arrays the scorer needs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.items = generate(cfg)["ladder"]
        self.steps = type_steps(cfg)
        items = self.items
        n = len(items)
        self.n_options = np.array([len(it["options"]) for it in items])
        self.target_idx = np.array([LETTERS.index(it["target_label"]) for it in items])
        self.control_idx = np.array([LETTERS.index(it["control_label"]) for it in items])
        self.names: list[tuple[str, str]] = []  # (category, name)
        self.fictional: list[bool] = []
        index: dict[tuple[str, str], int] = {}
        self.option_name = np.full((n, 4), -1)
        self.option_card = np.zeros((n, 4))  # +1 the option shows card text 0, -1 card text 1, 0 no card
        for i, it in enumerate(items):
            cards = it["stratum"] in ("full", "comp")
            for j, o in enumerate(it["options"]):
                key = (it["category"], o["name"])
                if key not in index:
                    index[key] = len(self.names)
                    self.names.append(key)
                    self.fictional.append(bool(o["is_fictional"]))
                self.option_name[i, j] = index[key]
                if cards and o["role"] == "target":
                    self.option_card[i, j] = 1.0 - 2.0 * it["swap"]
                elif cards and o["role"] == "control":
                    self.option_card[i, j] = 2.0 * it["swap"] - 1.0
        self.name_index = index
        self.stratum = np.array([it["stratum"] for it in items])
        self.etype = np.array([it["evidence_type"] for it in items])
        self.d = np.array([it["d"] for it in items])
        self.real_target = np.array([not it["target_is_fictional"] for it in items])
        self.unit_code, self.unit_ids = pd.factorize(np.array([it["unit_id"] for it in items]), sort=False)
        self.diag_draw = np.array([stable_int("mass", u) / 2.0 ** 64 for u in self.unit_ids])[self.unit_code]
        arm = np.where(self.n_options == 4, CTX_MULT, 1.0)
        self.name_mult = arm * np.array([NAME_MULT[s] * TYPE_MULT[t] for s, t in zip(self.stratum, self.etype)])
        self._units: pd.DataFrame | None = None

    # ------------------------------------------------------------------ truth
    def weights(self, stratum: str, etype: str) -> np.ndarray:
        """w at the ladder levels 1..K of a (stratum, evidence_type), at gamma = 1."""
        if etype == "composite":
            return np.array(COMPOSITE_WEIGHT[:COMPOSITE_LEVELS])
        x = np.array([anchor.level_x(etype, k, self.steps) for k in range(1, len(self.steps[etype]) + 1)])
        return STRATUM_WEIGHT[stratum] * TOP_WEIGHT[etype] * (x / x[-1]) ** SHAPE

    def pulls(self, world: World) -> np.ndarray:
        """Pull of every (category, name), at gamma = 1 and before any content change."""
        out = np.zeros(len(self.names))
        for i, (category, name) in enumerate(self.names):
            rng = rng_for(world.seed, "pull", category, name)
            out[i] = rng.normal(0.0, world.fict_sd) if self.fictional[i] else rng.normal(world.real_mean, world.real_sd)
        fict = np.array(self.fictional)
        for category in {c for c, _ in self.names}:
            members = fict & np.array([c == category for c, _ in self.names])
            out[members] -= out[members].mean()  # fictional pulls sum to zero within a category
        for key, value in world.pull_overrides.items():
            out[self.name_index[key]] = value
        return out

    def true_theta(self, world: World, category: str, entity: str, stratum: str, etype: str, arm: str = "pair",
                   levels: bool = False) -> tuple[float, bool]:
        """(theta, censored) of a real brand: the evidence whose weight equals the brand's name effect."""
        mult = NAME_MULT[stratum] * TYPE_MULT["none" if stratum == "full" else etype] * (CTX_MULT if arm == "ctx" else 1.0)
        delta = mult * world.real_pull_mult * self.pulls(world)[self.name_index[(category, entity)]]
        w = self.weights(stratum, etype)
        x = (np.arange(1, len(w) + 1.0) if levels or etype == "composite"
             else np.array([anchor.level_x(etype, k, self.steps) for k in range(1, len(w) + 1)]))
        value, censored = anchor.invert(x, w, delta)
        return float(value), bool(censored)

    # ------------------------------------------------------------------ scoring
    def logits(self, world: World) -> np.ndarray:
        """[items, 4] option logits (NaN beyond the item's options), everything multiplied by gamma."""
        n = len(self.items)
        pull = self.pulls(world) * np.where(self.fictional, 1.0, world.real_pull_mult)
        shown = self.option_name >= 0
        z = np.where(shown, pull[np.maximum(self.option_name, 0)], 0.0) * self.name_mult[:, None]
        weight = np.zeros(n)
        for stratum, etype in sorted({(s, t) for s, t, d in zip(self.stratum, self.etype, self.d) if d != 0}):
            cell = (self.stratum == stratum) & (self.etype == etype) & (self.d != 0)
            weight[cell] = self.weights(stratum, etype)[np.abs(self.d[cell]) - 1]
        rows = np.arange(n)
        against_target = np.where(self.d > 0, weight * np.where(self.real_target, world.real_slope, 1.0), 0.0)
        z[rows, self.target_idx] -= against_target
        z[rows, self.control_idx] -= np.where(self.d < 0, weight, 0.0)
        z += 0.5 * world.card_effect * self.option_card
        z -= world.position_bias * np.arange(4)[None, :]
        rng = rng_for(world.noise_seed, "item-noise")
        z += 20.0 + rng.normal(0.0, 1.0, size=(n, 1))  # a common level per item: cancels in every difference
        z += world.noise * rng.normal(0.0, 1.0, size=(n, 4))
        return world.gamma * np.where(shown, z, np.nan)

    def score(self, world: World, model_id: str = "synthetic") -> pd.DataFrame:
        """A ``scores`` table with the ladder stage's columns."""
        z = self.logits(world)
        n = len(z)
        finite = np.where(np.isnan(z), -np.inf, z)
        top = finite.max(axis=1, keepdims=True)
        log_soft = z - top - np.log(np.exp(finite - top).sum(axis=1, keepdims=True))
        data: dict[str, object] = {"item_id": [it["item_id"] for it in self.items], "model_id": model_id,
                                   "n_options": self.n_options, "target_idx": self.target_idx,
                                   "control_idx": self.control_idx, "n_tokens": 100}
        sampled = self.diag_draw < world.mass_fraction
        mass = z + world.gamma * world.mass_noise * rng_for(world.noise_seed, "mass-noise").normal(size=z.shape)
        mass = np.where(np.isnan(mass), -np.inf, mass)
        q = np.exp(mass - mass.max(axis=1, keepdims=True))
        q = 0.99 * q / q.sum(axis=1, keepdims=True)
        for j in range(4):
            data[f"z_sp_{j}"] = z[:, j]
            data[f"z_ns_{j}"] = z[:, j] - 15.0 * world.gamma  # " A" is the canonical form
            data[f"lp_forced_{j}"] = np.log(world.coverage) + log_soft[:, j]
            data[f"lp_free_{j}"] = np.log(0.6) + log_soft[:, j]
            data[f"q_forced_{j}"] = np.where(sampled & ~np.isnan(z[:, j]), q[:, j], np.nan)
            data[f"q_free_{j}"] = np.nan
        data.update(c_forced=world.coverage, c_free=0.6, cq_forced=np.where(sampled, 0.99, np.nan), cq_free=np.nan,
                    name_share_forced=np.nan, name_share_free=np.nan, pruned_forced=np.nan, s_o5=np.nan,
                    gen_text=None, gen_choice=None, gen_mode=None, gen_refusal=None, samp_n=np.nan,
                    samp_target=np.nan, samp_control=np.nan, samp_other=np.nan, letter_collision=False)
        return pd.DataFrame(data, index=np.arange(n))[STAGE_COLUMNS]

    # ------------------------------------------------------------------ fast path for simulations
    def unit_values(self, world: World) -> np.ndarray:
        """The unit values L of ``world`` without building a scores table (same numbers as ``unit_table``)."""
        z = self.logits(world)
        rows = np.arange(len(z))
        item = z[rows, self.target_idx] - z[rows, self.control_idx]
        return np.bincount(self.unit_code, weights=item) / np.bincount(self.unit_code)

    def units(self, world: World) -> pd.DataFrame:
        """Unit table of ``world`` through the fast path: metadata from one real ``unit_table`` call, ``L``
        replaced. Only ``L`` is valid; the probability columns are blanked."""
        if self._units is None:
            self._units = unit_table(self.items, self.score(World()))
        blank = {c: np.nan for c in ("order_bias", "p_target", "pick_target", "L_mass")}
        return self._units.assign(L=self.unit_values(world), **blank)

    def fast(self, world: World) -> dict:
        """The part of ``analyze_model`` that the pair comparison needs."""
        units = self.units(world)
        pairs = anchor.anchor_pairs(units)
        curves = anchor.anchor_curves(units, self.steps, pairs=pairs)
        return {"units": units, "pairs": pairs, "curves": curves, "cells": theta.zero_cells(units),
                "entities": theta.entity_table(units, curves, self.steps)}


@functools.lru_cache(maxsize=None)
def panel(name: str) -> Panel:
    return Panel(CONFIGS[name]())


def with_gamma(world: World, gamma: float) -> World:
    return replace(world, gamma=gamma)


def test_synthetic_scores_have_the_stage_columns():
    p = panel("smoke")
    scores = p.score(World(noise=0.2))
    assert list(scores.columns) == STAGE_COLUMNS and len(scores) == len(p.items)
    two = scores[scores["n_options"] == 2]
    assert two[["z_sp_2", "z_sp_3", "lp_forced_2"]].isna().all().all() and two[["z_sp_0", "z_sp_1"]].notna().all().all()
    lp = scores[[f"lp_forced_{j}" for j in range(4)]].to_numpy()
    assert np.allclose(np.nansum(np.exp(lp), axis=1), scores["c_forced"])
