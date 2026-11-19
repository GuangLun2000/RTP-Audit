"""Choice scoring with rotation averaging (spec §7.1), using answer paths (rtp.measure.answers).

A model may answer a lettered choice question with the letter ("B", " **B**") or with the brand name
("Nike"), immediately or after some prose. Scoring only the letter after a forced "Answer:" (the M2
pilot design) silently dropped name answers and measured a different position from the spec's free
answer. Each item is therefore scored at two positions from one forward pass: the free assistant start
(spec position) and the forced position after the "Answer:" prefill (base models: plain text +
"\nAnswer:", one position). At each position, every answer path (letter forms and the option's
displayed name) is summed into an option probability; the target slot's absent identity is a separate
``identity_leak`` class. ``p_A..p_D`` / ``s_target`` are the O5 estimator: free answers where the model
answers immediately, the deferred rest imputed by its committed answer after "Answer:". The free-only
(O3) and forced-only (O4) estimators, coverage, name channel, leak and imputation-free bounds are stored
per item. ``s_target = log(p / (1 - p))`` with ``p`` clipped to [1e-4, 1 - 1e-4].

The same 2% sample is also re-scored item by item (``s_target_solo``): in bfloat16 a score depends slightly
on batch composition and padding, and the summary reports that numerical noise floor next to the name
effects. The spec §7.1 consistency check decodes the sample greedily from the FREE position. Answers that
name an option within the first few tokens are "immediate"; the others are continued to
``DEFERRED_MAX_TOKENS`` and parsed for a concluding answer ("deferred"). Agreement is reported overall
(against the O5 argmax) and per mode (immediate vs the free argmax, deferred vs the forced argmax, which
tests the O5 imputation directly), each with its chance baseline.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_catalogue, load_choice_items
from rtp.data.choice_items import LETTERS
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.measure.parse import (ANSWER_PREFILL, COMPLETION_SUFFIX, DEFERRED_MAX_TOKENS,  # noqa: F401 (re-exported)
                               is_refusal, letter_token_ids, parse_answer, parse_final_answer, parse_letter,
                               render_inputs, select_gen_check)
from rtp.models import LoadedModel, apply_prompt_mode
from rtp.paths import stage_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

P_CLIP = 1e-4
RESULT_COLUMNS = ["item_id", "model_id", "p_A", "p_B", "p_C", "p_D", "s_target", "gen_letter"]
LOW_COVERAGE = 0.9  # below this mean option-answer coverage at a position, that position's readout is partial


# --------------------------------------------------------------------------- estimators
def s_target(p_target: np.ndarray | float, eps: float = P_CLIP) -> np.ndarray:
    p = np.clip(np.asarray(p_target, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def item_answer_paths(tokenizer: Any, items: Sequence[dict[str, Any]], letter_ids: dict[str, list[int]]):
    """Answer paths per item: displayed option names, plus the target slot's ABSENT identity as the leak class."""
    from rtp.measure.answers import build_item_paths

    cache: dict[str, tuple[int, ...]] = {}
    out = []
    for it in items:
        leak = it["placebo_substitute"] if not it["target_is_placebo"] else it["pair_entity"]
        out.append(build_item_paths(tokenizer, [o["name"] for o in it["options"]], leak, letter_ids, cache))
    return out


def score_items(lm: LoadedModel, items: Sequence[dict[str, Any]], letter_ids: dict[str, list[int]],
                gen_ids: set[str], cfg: Config, patches_for=None,
                case_sensitive: frozenset[str] | set[str] = frozenset()) -> pd.DataFrame:
    """Answer-path scores of ``items`` (see rtp.measure.answers): p_A..p_D and s_target are the primary O5
    estimator; O3 (free), O4 (forced), coverage, channels, identity leak and bounds are stored alongside."""
    from rtp.infer import generate
    from rtp.measure.answers import answer_distributions, estimators

    m = cfg.measure
    prompts = [it["prompt"] for it in items]
    free = render_inputs(lm, prompts, prefill=False)
    forced = render_inputs(lm, prompts, prefill=True)
    paths = item_answer_paths(lm.tokenizer, items, letter_ids)
    res = answer_distributions(lm, free, forced, paths, m.max_batch_size, m.max_batch_tokens, patches_for)
    rows = []
    for it, r in zip(items, res):
        t = LETTERS.index(it["target_label"])
        e = estimators(r["free"], r["forced"], t)
        P = e.pop("P")
        rows.append({"item_id": it["item_id"], "model_id": lm.model_id,
                     "p_A": P[0], "p_B": P[1], "p_C": P[2], "p_D": P[3], "s_target": e.pop("s_target"),
                     "gen_letter": None, "argmax_letter": LETTERS[int(np.argmax(P))],
                     "argmax_free": LETTERS[int(np.argmax(r["free"].q[:4]))],
                     "argmax_forced": LETTERS[int(np.argmax(r["forced"].q[:4]))], **e,
                     "gen_text": None, "gen_mode": None})
    df = pd.DataFrame(rows)
    for col in ("gen_letter", "gen_text", "gen_mode"):
        df[col] = df[col].astype("object")
    df["s_target_solo"] = np.nan
    check = [i for i, it in enumerate(items) if it["item_id"] in gen_ids]
    if check:
        # numerical noise floor: the checked items scored again ALONE (batch of one, no padding). In bfloat16 a
        # score depends slightly on batch composition and padding; the summary compares that with name effects
        solo = answer_distributions(lm, [free[i] for i in check], [forced[i] for i in check], [paths[i] for i in check],
                                    1, m.max_batch_tokens, patches_for)
        for i, r in zip(check, solo):
            df.at[i, "s_target_solo"] = estimators(r["free"], r["forced"], LETTERS.index(items[i]["target_label"]))["s_target"]
        # spec 7.1 check: the model's FREE greedy answer (no prefill), mapped to an option by letter or name;
        # answers that begin with prose are continued to find their final answer ("deferred")
        free_check = [free[i] for i in check]
        short = generate(lm, free_check, max_new_tokens=m.gen_max_new_tokens, do_sample=False,
                         max_batch_size=m.max_batch_size, max_batch_tokens=m.max_batch_tokens)
        deferred = [k for k, (i, t) in enumerate(zip(check, short))
                    if parse_answer(t[0], items[i]["options"], case_sensitive) is None]
        long = generate(lm, [free_check[k] for k in deferred], max_new_tokens=DEFERRED_MAX_TOKENS, do_sample=False,
                        max_batch_size=m.max_batch_size, max_batch_tokens=m.max_batch_tokens) if deferred else []
        long_of = dict(zip(deferred, long))
        for k, (i, t) in enumerate(zip(check, short)):
            opts = items[i]["options"]
            if k in long_of:
                df.at[i, "gen_text"] = long_of[k][0]
                df.at[i, "gen_letter"] = parse_final_answer(long_of[k][0], opts, case_sensitive)
                df.at[i, "gen_mode"] = "deferred"
            else:
                df.at[i, "gen_text"] = t[0]
                df.at[i, "gen_letter"] = parse_answer(t[0], opts, case_sensitive)
                df.at[i, "gen_mode"] = "immediate"
        df["gen_refusal"] = df["gen_text"].map(lambda x: is_refusal(x) if isinstance(x, str) else None).astype("object")
    return df


def case_sensitive_names(catalogue: Sequence[dict[str, Any]]) -> frozenset[str]:
    """Real brands flagged ``match_case`` in the catalogue (ordinary words: Sharp, KIND, Quest, ...)."""
    return frozenset(r["entity"] for r in catalogue if r.get("match_case") and not r["is_placebo"])


# --------------------------------------------------------------------------- stage
def measure_stage_name(split: str) -> str:
    return "measure" if split == "dev" else "measure_test"


def run_measure(cfg: Config, model_id: str, split: str = "dev", unseal: bool = False, force: bool = False,
                prompt_mode: str = "auto"):
    from rtp.measure.name_effect import summarize_measure
    from rtp.models import load_model

    check_split_access(split, unseal)
    items = load_choice_items(cfg, split, unseal)
    catalogue = load_catalogue(cfg, split, unseal)
    sdir = stage_dir(cfg.name, model_id, measure_stage_name(split))
    shash = stages.stage_hash(cfg, "measure", model_id, split, prompt_mode)
    if io.finished_stage(sdir, shash, force):
        log.info("measure already finished for %s with this config and code (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta=stages.stage_meta("measure", model_id, split, prompt_mode))
    stages.check_resume_environment(sdir, force)
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        if not items:
            raise ValueError(f"no choice items in split '{split}'")
        gen_ids = select_gen_check([it["item_id"] for it in items], cfg.measure.gen_check_fraction, cfg.seed)
        case_names = case_sensitive_names(catalogue)
        state: dict[str, Any] = {}

        def process(chunk, idx):
            if "lm" not in state:
                state["lm"] = apply_prompt_mode(load_model(model_id), prompt_mode)
                stages.record_model_revision(sdir, state["lm"].model)
                state["letters"] = letter_token_ids(state["lm"].tokenizer)
                io.write_json(sdir / "letter_tokens.json", state["letters"])
            return score_items(state["lm"], chunk, state["letters"], gen_ids, cfg, case_sensitive=case_names)

        io.run_sharded(items, cfg.measure.shard_size, sdir / "shards", process, desc="measure")
        scores = io.load_shards(sdir / "shards")
        if len(scores) != len(items):
            raise RuntimeError(f"expected {len(items)} scored items, found {len(scores)}")
        io.write_parquet(sdir / "scores.parquet", scores)
        info = summarize_measure(cfg, model_id, split, items, catalogue, scores, sdir,
                                 elapsed=time.time() - t0)
        from rtp.provenance import environment

        io.mark_done(sdir, {**info, "finished_with": environment()})
        log.info("measure finished: %s", sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir
