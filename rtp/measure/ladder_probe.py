"""`ladder-probe`: which tokens take the probability that is not on the option letters, and where do the
answers that start with them end?

Cycle C0 found that after the prefill "Answer:" the option letters carry only 0.63 to 0.68 of the probability
on the two-option ladder items, and the ladder stage stores letter logits only. This probe reads a small
balanced sample of ladder items (every kind of item, both option orders): the most likely next tokens at the
forced position and the greedy continuation from there.

It reports per kind of item two statistics that do not move when every logit is multiplied by a constant:

* ``top1_letter``: the share of items whose most likely next token is one of the item's option letters;
* among the items whose most likely token is NOT a letter: where the greedy continuation ends (on the option
  the letter readout picks, on the other option, on no option, or on an opening that declines to choose).

The mean letter mass is reported next to them and does move with scale.

Nothing here is part of a stage identity (like `format-probe`): no stage imports this module, the logits are
the model's own bf16 logits (good enough to rank tokens; the preference level never uses them), and outputs
are overwritten under ``runs/<config>/ladder_probe/<model>/``. Development split only.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rtp import io
from rtp.config import Config
from rtp.data.build import load_ladder_items
from rtp.paths import model_slug, run_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
N_UNITS, TOP_K, MAX_NEW_TOKENS = 40, 20, 32
OUTCOMES = ("picked option", "other option", "declines", "no option")
_DECLINE_RE = re.compile(
    r"(?i)^[\s\"'*(\[]*(?:both\b|neither\b|either\b|it\s+depends|that\s+depends|depends\b|there(?:'s|\s+is)\s+no\b|"
    r"no\s+clear\b|they(?:'re|\s+are)\s+(?:the\s+same|equal|identical|tied)|the\s+same\b|equal(?:ly)?\b|"
    r"i\s+(?:can(?:no|')t|can\s+not|am\s+unable|would\s+need)|unable\b|not\s+enough\b|insufficient\b)")


def cell_of(item: dict[str, Any]) -> str:
    """Kind of item: who the target is, what is shown, and whether the two options show the same evidence."""
    target = "fictional" if item["family"] == "anchor" else "real brand"
    return f"{target} / {item['stratum']} / {'tied' if int(item['d']) == 0 else 'differs'}"


def sample_units(items: Sequence[dict[str, Any]], n_units: int, seed: int) -> list[dict[str, Any]]:
    """Every option order of up to ``n_units`` units per kind of item, drawn with a fixed seed."""
    units: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for it in items:
        units.setdefault(cell_of(it), {}).setdefault(it["unit_id"], []).append(it)
    rng = rng_for(seed, "ladder_probe")
    out: list[dict[str, Any]] = []
    for cell in sorted(units):
        ids = sorted(units[cell])
        for k in rng.permutation(len(ids))[:n_units]:
            out.extend(sorted(units[cell][ids[int(k)]], key=lambda it: int(it["order"])))
    return out


def declines(text: str | None) -> bool:
    """The continuation opens by not choosing ("Both ...", "Either ...", "It depends ..."). The stage's
    answer parser credits such an opening to the first option it names; the probe counts it apart."""
    return bool(text) and bool(_DECLINE_RE.match(text))


def classify(text: str | None, options: Sequence[dict[str, Any]], pick: str) -> tuple[str, str | None]:
    """(outcome, option letter) of one greedy continuation against the letter the readout picks."""
    from rtp.measure.parse import parse_answer, parse_final_answer, restrict_to_options

    if declines(text):
        return "declines", None
    label = restrict_to_options(parse_answer(text, options), options) or \
        restrict_to_options(parse_final_answer(text, options), options)
    if label is None:
        return "no option", None
    return ("picked option" if label == pick else "other option"), label


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Per kind of item and overall: letter mass, the scale-free shares, the tokens that take the rest."""
    def block(g: pd.DataFrame) -> dict[str, Any]:
        off = g[~g["top1_letter"]]
        tokens: dict[str, list[float]] = {}
        for row in g["top_other"]:
            for tok, p, first in json.loads(row):
                entry = tokens.setdefault(tok, [0.0, 0.0])
                entry[0] += p
                entry[1] += float(first)
        ranked = sorted(tokens.items(), key=lambda kv: -kv[1][0])[:10]
        return {"n_items": int(len(g)), "letter_mass": float(g["letter_mass"].mean()),
                "top1_letter": float(g["top1_letter"].mean()),
                "continuation": {k: float((g["outcome"] == k).mean()) for k in OUTCOMES},
                "n_top1_not_letter": int(len(off)),
                "continuation_when_top1_not_letter": {k: float((off["outcome"] == k).mean()) for k in OUTCOMES}
                if len(off) else {},
                "other_tokens": [{"token": tok, "mean_prob": v[0] / len(g), "share_top1": v[1] / len(g)}
                                 for tok, v in ranked]}

    return {"all": block(df), "cells": {cell: block(g) for cell, g in df.groupby("cell", sort=True)}}


def examples(df: pd.DataFrame, per_cell: int = 2) -> list[dict[str, Any]]:
    """Per kind of item, the items with the lowest letter mass: their first tokens and their continuation."""
    out = []
    for cell, g in df.sort_values("letter_mass").groupby("cell", sort=True):
        for r in g.head(per_cell).itertuples(index=False):
            out.append({"cell": cell, "letter_mass": round(float(r.letter_mass), 3),
                        "first_tokens": json.loads(r.top_tokens)[:4], "outcome": r.outcome,
                        "text": (r.gen_text or "").replace("\n", " ")[:110]})
    return out


def markdown(model_id: str, tier: str, res: dict[str, Any], df: pd.DataFrame, prompt_tail: str) -> str:
    def pct(x: Any) -> str:
        return "n/a" if x is None or x != x else f"{100.0 * float(x):.0f}%"

    def split(d: dict[str, float]) -> str:
        return ", ".join(f"{k} {pct(d.get(k))}" for k in OUTCOMES) if d else "n/a"

    lines = [f"# Ladder probe: {model_id} (tier {tier})", "",
             f"{res['all']['n_items']} items, every option order of up to {res['n_units_per_cell']} units per kind of "
             f"item; the {res['top_k']} most likely next tokens after the prefill and a greedy continuation of "
             f"{res['max_new_tokens']} tokens. 'top-1 is a letter' and the continuation shares do not move with "
             f"scale; the letter mass does. 'picked option' = the option whose letter has the higher probability.", "",
             "| kind of item (target / stratum / evidence) | items | letter mass | top-1 is a letter | continuation "
             "ends on | items with another top-1 | their continuation ends on |", "|---|---|---|---|---|---|---|"]
    for cell, b in [("all", res["all"]), *res["cells"].items()]:
        lines.append(f"| {cell} | {b['n_items']} | {b['letter_mass']:.3f} | {pct(b['top1_letter'])} | "
                     f"{split(b['continuation'])} | {b['n_top1_not_letter']} | "
                     f"{split(b['continuation_when_top1_not_letter'])} |")
    lines += ["", "Tokens other than the item's option letters, by mean probability (share of items where the token "
                  "is the most likely one):", ""]
    for cell, b in [("all", res["all"]), *res["cells"].items()]:
        lines.append(f"- **{cell}**: " + "; ".join(f"{t['token']} {t['mean_prob']:.3f} ({pct(t['share_top1'])})"
                                                   for t in b["other_tokens"][:8]))
    lines += ["", "Examples whose most likely token is not a letter (kind | letter mass | first tokens | outcome | "
                  "continuation):", ""]
    off = df[~df["top1_letter"]]
    for cell, g in off.groupby("cell", sort=True):
        for r in g.head(4).itertuples(index=False):
            first = ", ".join(f"{t} {p:.2f}" for t, p in json.loads(r.top_tokens)[:4])
            text = (r.gen_text or "").replace("\n", " ").replace("|", "/")[:90]
            lines.append(f"- {cell} | {r.letter_mass:.2f} | {first} | {r.outcome} | {text}")
    lines += ["", "Rendered prompt tail (forced position):", "", "```", prompt_tail, "```", ""]
    return "\n".join(lines) + "\n"


def run_ladder_probe(cfg: Config, model_id: str, tier: str = "core", n_units: int = N_UNITS, top_k: int = TOP_K,
                     max_new_tokens: int = MAX_NEW_TOKENS, prompt_mode: str = "auto"):
    import torch

    from rtp.infer import generate, last_token_reduce
    from rtp.measure.ladder import letter_forms
    from rtp.measure.parse import render_inputs
    from rtp.models import apply_prompt_mode, load_model

    items = sample_units(load_ladder_items(cfg, tier, "dev"), n_units, cfg.seed)
    if not items:
        raise ValueError(f"no ladder items in tier '{tier}'")
    lm = apply_prompt_mode(load_model(model_id), prompt_mode)
    tok = lm.tokenizer
    forms = letter_forms(tok)
    ids_of = [sorted(set(forms[L]["all"])) for L in LETTERS]
    forced = render_inputs(lm, [it["prompt"] for it in items], prefill=True)
    m = cfg.ladder_measure

    def reduce(logits: Any) -> Any:  # [B, V] -> top-k log-probs, their token ids, log mass of each letter
        logp = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logp, top_k, dim=-1)
        mass = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in ids_of], dim=-1)
        return torch.cat([top.values, top.indices.to(logp.dtype), mass], dim=-1)

    out = last_token_reduce(lm, forced, reduce, m.max_batch_size, m.max_batch_tokens)
    texts = generate(lm, forced, max_new_tokens=max_new_tokens, do_sample=False, max_batch_size=m.max_batch_size,
                     max_batch_tokens=m.max_batch_tokens)
    shown: dict[int, str] = {}

    def show(token_id: int) -> str:
        if token_id not in shown:
            shown[token_id] = repr(tok.decode([token_id]))
        return shown[token_id]

    rows = []
    for i, it in enumerate(items):
        n = len(it["options"])
        own = {t for ids in ids_of[:n] for t in ids}
        probs = np.exp(out[i, :top_k].astype(np.float64))
        token_ids = [int(round(float(x))) for x in out[i, top_k:2 * top_k]]
        mass = np.exp(out[i, 2 * top_k:2 * top_k + n].astype(np.float64))
        pick = LETTERS[int(np.argmax(mass))]
        outcome, label = classify(texts[i][0], it["options"], pick)
        other = [(show(t), float(p), k == 0) for k, (t, p) in enumerate(zip(token_ids, probs)) if t not in own]
        rows.append({"item_id": it["item_id"], "unit_id": it["unit_id"], "cell": cell_of(it), "order": int(it["order"]),
                     "n_options": n, "target_label": it["target_label"], "letter_mass": float(mass.sum()),
                     "top1_letter": token_ids[0] in own, "pick": pick, "pick_is_target": pick == it["target_label"],
                     "top_tokens": json.dumps([(show(t), round(float(p), 4)) for t, p in zip(token_ids, probs)][:10]),
                     "top_other": json.dumps([(s, round(p, 5), first) for s, p, first in other]),
                     "gen_text": texts[i][0], "outcome": outcome, "gen_label": label})
    df = pd.DataFrame(rows)
    res = {"model_id": model_id, "tier": tier, "n_units_per_cell": int(n_units), "top_k": int(top_k),
           "max_new_tokens": int(max_new_tokens), **summarize(df), "examples": examples(df)}
    out_dir = run_dir(cfg.name) / "ladder_probe" / model_slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_parquet(out_dir / "items.parquet", df)
    io.write_json(out_dir / "ladder_probe.json", res)
    io.atomic_write_text(out_dir / "summary.md", markdown(model_id, tier, res, df, tok.decode(forced[0])[-200:]))
    a = res["all"]
    log.info("ladder probe %s (%s): %d items, letter mass %.3f, top-1 is a letter in %.0f%%; continuation ends on %s",
             model_id, tier, a["n_items"], a["letter_mass"], 100 * a["top1_letter"],
             ", ".join(f"{k} {100 * v:.0f}%" for k, v in a["continuation"].items()))
    log.info("tokens that take the rest: %s",
             "; ".join(f"{t['token']} {t['mean_prob']:.3f}" for t in a["other_tokens"][:8]))
    log.info("ladder probe written to %s", out_dir / "summary.md")
    return out_dir
