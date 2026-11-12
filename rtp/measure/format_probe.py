"""`format-probe`: how does a model answer the choice question? A few minutes on a small dev sample.

Run it on every new checkpoint BEFORE the full measure stage. It scores a balanced sample of dev items
(every evidence level x real/placebo version, rotations spread) with the measure-stage estimator and
decodes EVERY sampled item greedily from the free position, then prints per item the coverage at both
positions, the name share, the free and forced argmax and the parsed free answer, plus the tail of one
rendered prompt of each kind (to see the chat template). Nothing here is part of any stage identity;
outputs are overwritten under runs/<config>/format_probe/<model>/.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from rtp import io
from rtp.config import Config
from rtp.data.build import load_choice_items
from rtp.paths import model_slug, run_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)


def sample_items(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """About ``n`` DISTINCT items, balanced over (evidence level, version); rotations cycle within each cell."""
    cells: dict[tuple[int, bool], list[dict[str, Any]]] = {}
    for it in items:
        cells.setdefault((it["evidence_level"], it["target_is_placebo"]), []).append(it)
    per = max(1, -(-n // max(1, len(cells))))
    rng = rng_for(seed, "format_probe")
    out = []
    for key in sorted(cells):
        pool = [cells[key][int(i)] for i in rng.permutation(len(cells[key]))]
        taken: set[str] = set()
        for k in range(min(per, len(pool))):
            cand = next((it for it in pool if it["rotation"] == k % 4 and it["item_id"] not in taken),
                        next(it for it in pool if it["item_id"] not in taken))
            taken.add(cand["item_id"])
            out.append(cand)
    return out


def run_format_probe(cfg: Config, model_id: str, n_items: int = 40, prompt_mode: str = "auto"):
    from rtp.measure.choice import LOW_COVERAGE, letter_token_ids, render_inputs, score_items
    from rtp.measure.name_effect import _pct, consistency
    from rtp.models import apply_prompt_mode, load_model

    items = sample_items(load_choice_items(cfg, "dev"), n_items, cfg.seed)
    lm = apply_prompt_mode(load_model(model_id), prompt_mode)
    letters = letter_token_ids(lm.tokenizer)
    df = score_items(lm, items, letters, {it["item_id"] for it in items}, cfg)
    meta = pd.DataFrame([{"item_id": it["item_id"], "evidence_level": it["evidence_level"],
                          "version": "placebo" if it["target_is_placebo"] else "real",
                          "target_label": it["target_label"]} for it in items])
    df = meta.merge(df, on="item_id", validate="one_to_one")
    out = run_dir(cfg.name) / "format_probe" / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    io.write_parquet(out / "items.parquet", df)
    tok = lm.tokenizer
    free = tok.decode(render_inputs(lm, [items[0]["prompt"]], prefill=False)[0])
    forced = tok.decode(render_inputs(lm, [items[0]["prompt"]], prefill=True)[0])
    cons = consistency(df)
    by_level = df.groupby(["evidence_level", "version"])[["c_free", "c_forced", "name_share_free", "leak_free"]].mean()
    modes = df["gen_mode"].value_counts().to_dict()
    res = {"model_id": model_id, "n_items": int(len(df)), "c_free": float(df["c_free"].mean()),
           "article_free": float(df["article_free"].mean()),
           "c_forced": float(df["c_forced"].mean()), "name_share_free": float(np.nanmean(df["name_share_free"])),
           "modes": {str(k): int(v) for k, v in modes.items()}, "consistency": cons,
           "prompt_tail_free": free[-160:], "prompt_tail_forced": forced[-160:]}
    io.write_json(out / "format_probe.json", res)

    lines = [f"# Format probe: {model_id}", "",
             f"- items: {len(df)} (dev, balanced over evidence level x version); greedy free answers decoded for all",
             f"- mean coverage: free {res['c_free']:.3f}, forced {res['c_forced']:.3f}; name share (free) "
             f"{res['name_share_free']:.3f}; answer modes {res['modes']}; mass of \"A\" + article word removed "
             f"from option A (free) {res['article_free']:.4f}",
             f"- free answer vs O5 argmax: {_pct(cons['agreement'])} (chance {_pct(cons['chance'])}); immediate vs "
             f"free argmax {_pct(cons.get('agreement_immediate'))}; deferred vs forced argmax "
             f"{_pct(cons.get('agreement_deferred'))}", "",
             "Rendered prompt tails (free / forced):", "", "```", free[-160:], "```", "```", forced[-160:], "```", "",
             "| e | version | target | c_free | c_forced | name share | argmax free | argmax forced | mode | parsed "
             "| free answer |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        text = (r.gen_text or "").replace("\n", " ").replace("|", "/")[:70]
        lines.append(f"| {r.evidence_level} | {r.version} | {r.target_label} | {r.c_free:.2f} | {r.c_forced:.2f} | "
                     f"{r.name_share_free:.2f} | {r.argmax_free} | {r.argmax_forced} | {r.gen_mode} | "
                     f"{r.gen_letter or '—'} | {text} |")
    io.atomic_write_text(out / "summary.md", "\n".join(lines) + "\n")

    log.info("format probe %s: c_free %.3f, c_forced %.3f, name share %.3f, modes %s", model_id, res["c_free"],
             res["c_forced"], res["name_share_free"], res["modes"])
    log.info("per evidence level and version:\n%s", by_level.round(3).to_string())
    for r in df.head(12).itertuples(index=False):
        log.info("  e%d %-7s c_free %.2f c_forced %.2f -> free %s / forced %s | %s: %r", r.evidence_level, r.version,
                 r.c_free, r.c_forced, r.argmax_free, r.argmax_forced, r.gen_mode, (r.gen_text or "")[:60])
    if res["c_forced"] < LOW_COVERAGE:
        log.warning("c_forced %.3f < %.2f: even after \"Answer:\" the model rarely names an option; check the "
                    "prompt mode and chat template before measuring", res["c_forced"], LOW_COVERAGE)
    if res["c_free"] < LOW_COVERAGE:
        log.info("note: %.0f%% of the free-position mass defers the answer; O5 imputes it from the forced position "
                 "(deferred agreement above tests that)", 100 * (1 - res["c_free"]))
    log.info("format probe written to %s", out / "summary.md")
    return out
