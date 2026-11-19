"""The ``ladder`` stage: letter logits of paired ladder items (rtp.data.ladder_items).

PRIMARY READOUT. For every item the stage stores the logit of each option's letter at the forced position
(the next token after the "Answer:" prefill), in two single-token forms: " A" (``z_sp``) and "A" (``z_ns``).
The analysis uses the difference between the target's and the control's letter logit in ONE form. A
difference of two logits at one position is multiplied by c when every logit is multiplied by c, which is
what makes the preference level (a ratio of such differences) invariant to a pure change of scale; sums
over answer paths, mixtures of positions and clipped probabilities are not.

Logits are recomputed in float32 from the final hidden state: the model's own head output is bfloat16 on
GPU, which rounds a logit near 20 to a grid of 0.125, more than many of the differences measured here.
Every batch compares the recomputed letter logits with the model's own output (``head_dev`` in the stage
summary): they must agree up to that rounding. A model that transforms its logits after the head (logit
soft-capping, a scaled head, an adapter on the head) fails this check instead of being read wrongly.

DIAGNOSTICS (readout validity, never used for the preference level):
* every item: the probability mass on the option letters at the forced and at the free position
  (``c_forced``, ``c_free``; all single-token letter forms merged);
* a deterministic sample of units (``diag_fraction``, plus every unit containing a name that starts with an
  option-letter token): answer-path masses at both positions, i.e. letters AND names (rtp.measure.answers);
* a smaller sample (``gen_check_fraction``): the greedy free answer, and ``n_samples`` sampled free answers.

Hidden outcomes: nothing here compares models. One model id in, one table out.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_ladder_items
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.measure.answers import NAME_FORMS, answer_distributions, build_item_paths, estimators
from rtp.measure.parse import (DEFERRED_MAX_TOKENS, LETTERS, is_refusal, letter_token_ids, parse_answer,
                               parse_final_answer, render_inputs, restrict_to_options)
from rtp.models import LoadedModel, apply_prompt_mode
from rtp.paths import stage_dir
from rtp.rng import stable_int

log = logging.getLogger(__name__)

MAX_OPTIONS = len(LETTERS)
LOW_COVERAGE = 0.9  # mean letter mass at the forced position below this: the letter readout is partial
# Recomputed vs model logits. ``head_dev`` = |difference| / (|logit| / 64 + 0.05): bfloat16 keeps 8 significant
# bits, so rounding alone stays below |logit| / 256, i.e. head_dev < 0.25 for float32-accumulated products.
HEAD_DEV_WARN = 1.0  # above: logged and reported (bf16 arithmetic differs more than expected on this device)
HEAD_DEV_FAIL = 16.0  # above: the head is not a plain linear map of the hidden state; the readout would be wrong
OPTION_COLUMNS = ("z_sp", "z_ns", "lp_forced", "lp_free", "q_forced", "q_free")


def ladder_stage_name(tier: str, split: str) -> str:
    return f"ladder_{tier}" if split == "dev" else f"ladder_{tier}_test"


# --------------------------------------------------------------------------- letter tokens
def letter_forms(tokenizer: Any, letters: Sequence[str] = LETTERS) -> dict[str, dict[str, Any]]:
    """Per letter: the single token of " A" (``sp``) and of "A" (``ns``), None when it is not one token, and
    every single token whose stripped text is the letter (``all``)."""
    every = letter_token_ids(tokenizer, letters)

    def single(text: str) -> int | None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return int(ids[0]) if len(ids) == 1 else None

    return {L: {"sp": single(" " + L), "ns": single(L), "all": [int(i) for i in every[L]]} for L in letters}


def name_starts_with_letter(tokenizer: Any, name: str, forms: dict[str, dict[str, Any]], letters: Sequence[str],
                            cache: dict[str, bool]) -> bool:
    """True if a spelled-out answer naming ``name`` begins with a token that is also an option letter
    ("Acer" -> "A" + "cer"): such a name answer is credited to that letter by the letter readout."""
    if name not in cache:
        letter_ids = {i for L in letters for i in forms[L]["all"]}
        firsts = {tokenizer(f.format(n=name), add_special_tokens=False)["input_ids"][:1][0] for f in NAME_FORMS
                  if tokenizer(f.format(n=name), add_special_tokens=False)["input_ids"]}
        cache[name] = bool(firsts & letter_ids)
    return cache[name]


# --------------------------------------------------------------------------- primary readout
@torch.inference_mode()
def letter_readout(lm: LoadedModel, free_ids: Sequence[Sequence[int]], forced_ids: Sequence[Sequence[int]],
                   token_ids: Sequence[int], max_batch_size: int, max_batch_tokens: int) -> list[dict[str, Any]]:
    """Per item: float32 logits of ``token_ids`` and the full-vocabulary log-sum-exp, at the free and at the
    forced position. One forward pass per batch when the forced prompt extends the free prompt by the same
    number of tokens for every item of the batch (the usual case); two passes otherwise."""
    from rtp.infer import pad_left, run_batched

    head = lm.model.get_output_embeddings()
    weight = head.weight.detach().float()  # [V, d]; a float32 copy on GPU (bf16 weights), no copy on CPU
    bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    cols = torch.tensor(list(token_ids), dtype=torch.long, device=weight.device)
    pad_id = lm.tokenizer.pad_token_id
    captured: dict[str, torch.Tensor] = {}
    handle = head.register_forward_pre_hook(lambda module, args: captured.__setitem__("h", args[0]))

    def positions(seqs: list[list[int]], keep: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Final hidden states of the last ``keep`` positions, [B, keep, d], float32, and the model's own
        logits at those positions (already scaled by a ``@scale`` variant's output hook)."""
        ids, mask, pos = pad_left(seqs, pad_id, lm.device)
        out = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=keep)
        return captured.pop("h").float(), out.logits

    def read(hidden: torch.Tensor, own: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        z = hidden @ weight.T  # [B, V]
        if bias is not None:
            z = z + bias
        z = z * lm.logit_scale  # variant "@scale=c": the same factor the output hook applies
        ref = own[:, cols].float()
        dev = float(((z[:, cols] - ref).abs() / (ref.abs() / 64.0 + 0.05)).max())
        if dev > HEAD_DEV_FAIL:
            raise RuntimeError(
                f"the letter logits recomputed from the final hidden state differ from the model's own logits "
                f"(head_dev {dev:.1f} > {HEAD_DEV_FAIL:g}): {lm.model_id} does not map its final hidden state to "
                f"logits by its output-embedding matrix alone (logit soft-capping, a scaled head, an adapter on the "
                f"head?). The float32 readout is not valid for it.")
        return (z[:, cols].cpu().numpy().astype(np.float64),
                torch.logsumexp(z, dim=-1).cpu().numpy().astype(np.float64), dev)

    def fn(batch: list[int]) -> list[dict[str, Any]]:
        ns = [len(forced_ids[i]) - len(free_ids[i]) for i in batch]
        prefix = all(list(forced_ids[i][: len(free_ids[i])]) == list(free_ids[i]) for i in batch)
        if prefix and len(set(ns)) == 1:
            h, own = positions([list(forced_ids[i]) for i in batch], ns[0] + 1)
            zf, lf, dev = read(h[:, 0, :], own[:, 0, :])
            zx, lx, dev2 = (zf, lf, dev) if ns[0] == 0 else read(h[:, -1, :], own[:, -1, :])
        else:
            h, own = positions([list(free_ids[i]) for i in batch], 1)
            zf, lf, dev = read(h[:, -1, :], own[:, -1, :])
            h, own = positions([list(forced_ids[i]) for i in batch], 1)
            zx, lx, dev2 = read(h[:, -1, :], own[:, -1, :])
        return [{"z_free": zf[r], "lse_free": float(lf[r]), "z_forced": zx[r], "lse_forced": float(lx[r]),
                 "head_dev": max(dev, dev2)} for r in range(len(batch))]

    try:
        return run_batched([list(x) for x in forced_ids], fn, max_batch_size, max_batch_tokens, desc="letters")
    finally:
        handle.remove()


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def letter_rows(items: Sequence[dict[str, Any]], readout: Sequence[dict[str, Any]], forms: dict[str, dict[str, Any]],
                col_of: dict[int, int], n_tokens: Sequence[int], model_id: str) -> list[dict[str, Any]]:
    rows = []
    for it, r, nt in zip(items, readout, n_tokens):
        n = len(it["options"])
        row: dict[str, Any] = {"item_id": it["item_id"], "model_id": model_id, "n_options": n,
                               "target_idx": LETTERS.index(it["target_label"]),
                               "control_idx": LETTERS.index(it["control_label"]), "n_tokens": int(nt)}
        for j in range(MAX_OPTIONS):
            f = forms[LETTERS[j]] if j < n else None
            row[f"z_sp_{j}"] = float(r["z_forced"][col_of[f["sp"]]]) if f and f["sp"] is not None else np.nan
            row[f"z_ns_{j}"] = float(r["z_forced"][col_of[f["ns"]]]) if f and f["ns"] is not None else np.nan
            idx = [col_of[t] for t in f["all"]] if f else []
            row[f"lp_forced_{j}"] = _logsumexp(r["z_forced"][idx]) - r["lse_forced"] if f else np.nan
            row[f"lp_free_{j}"] = _logsumexp(r["z_free"][idx]) - r["lse_free"] if f else np.nan
            row[f"q_forced_{j}"] = np.nan
            row[f"q_free_{j}"] = np.nan
        row["c_forced"] = float(sum(np.exp(row[f"lp_forced_{j}"]) for j in range(n)))
        row["c_free"] = float(sum(np.exp(row[f"lp_free_{j}"]) for j in range(n)))
        row["head_dev"] = float(r.get("head_dev", np.nan))  # of the item's batch (see HEAD_DEV_WARN)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- samples for the diagnostics
def _unit_fraction(seed: int, stream: str, unit_id: str) -> float:
    return stable_int(seed, stream, unit_id) / 2.0**64


def diag_units(items: Sequence[dict[str, Any]], cfg: Config, collide: set[str]) -> tuple[set[str], set[str]]:
    """(units scored with answer paths, units whose free answers are decoded): deterministic per unit and
    independent of the model. The decoded units are a subset of the answer-path units."""
    m = cfg.ladder_measure
    gen = {it["unit_id"] for it in items if _unit_fraction(cfg.seed, "ladder-gen", it["unit_id"]) < m.gen_check_fraction}
    diag = {it["unit_id"] for it in items
            if _unit_fraction(cfg.seed, "ladder-diag", it["unit_id"]) < m.diag_fraction or it["unit_id"] in collide}
    return diag | gen, gen


def score_items(lm: LoadedModel, items: Sequence[dict[str, Any]], forms: dict[str, dict[str, Any]], cfg: Config,
                collide_cache: dict[str, bool], timing: dict[str, float]) -> pd.DataFrame:
    from rtp.infer import generate

    m = cfg.ladder_measure
    tok = lm.tokenizer
    prompts = [it["prompt"] for it in items]
    free = render_inputs(lm, prompts, prefill=False)
    forced = render_inputs(lm, prompts, prefill=True)
    token_ids = sorted({t for f in forms.values() for t in f["all"] + [x for x in (f["sp"], f["ns"]) if x is not None]})
    col_of = {t: i for i, t in enumerate(token_ids)}
    t0 = time.time()
    readout = letter_readout(lm, free, forced, token_ids, m.max_batch_size, m.max_batch_tokens)
    timing["primary_s"] += time.time() - t0
    df = pd.DataFrame(letter_rows(items, readout, forms, col_of, [len(x) for x in forced], lm.model_id))
    for col in ("cq_forced", "cq_free", "name_share_forced", "name_share_free", "pruned_forced", "s_o5"):
        df[col] = np.nan
    for col in ("gen_text", "gen_choice", "gen_mode"):
        df[col] = pd.Series([None] * len(df), dtype="object")
    df["gen_refusal"] = pd.Series([None] * len(df), dtype="object")
    for col in ("samp_n", "samp_target", "samp_control", "samp_other"):
        df[col] = np.nan
    df["letter_collision"] = False

    collide = set()
    for it in items:
        letters = LETTERS[: len(it["options"])]
        if any(name_starts_with_letter(tok, o["name"], forms, letters, collide_cache) for o in it["options"]):
            collide.add(it["unit_id"])
    df["letter_collision"] = [it["unit_id"] in collide for it in items]
    diag, gen = diag_units(items, cfg, collide)
    didx = [i for i, it in enumerate(items) if it["unit_id"] in diag]
    if didx:
        t0 = time.time()
        letter_ids = {L: f["all"] for L, f in forms.items()}
        cache: dict[str, tuple[int, ...]] = {}
        paths = [build_item_paths(tok, [o["name"] for o in items[i]["options"]], None, letter_ids, cache) for i in didx]
        res = answer_distributions(lm, [free[i] for i in didx], [forced[i] for i in didx], paths,
                                   m.max_batch_size, m.max_batch_tokens)
        for i, r in zip(didx, res):
            n = len(items[i]["options"])
            e = estimators(r["free"], r["forced"], int(df.at[i, "target_idx"]))
            for j in range(n):
                df.at[i, f"q_forced_{j}"] = float(r["forced"].q[j])
                df.at[i, f"q_free_{j}"] = float(r["free"].q[j])
            df.at[i, "cq_forced"], df.at[i, "cq_free"] = e["c_forced"], e["c_free"]
            df.at[i, "name_share_forced"], df.at[i, "name_share_free"] = e["name_share_forced"], e["name_share_free"]
            df.at[i, "pruned_forced"], df.at[i, "s_o5"] = e["pruned_forced"], e["s_target"]
        timing["diag_s"] += time.time() - t0
    gidx = [i for i, it in enumerate(items) if it["unit_id"] in gen]
    if gidx:
        t0 = time.time()
        free_g = [free[i] for i in gidx]
        short = generate(lm, free_g, max_new_tokens=m.gen_max_new_tokens, do_sample=False,
                         max_batch_size=m.max_batch_size, max_batch_tokens=m.max_batch_tokens)
        deferred = [k for k, i in enumerate(gidx) if parse_answer(short[k][0], items[i]["options"]) is None]
        long = generate(lm, [free_g[k] for k in deferred], max_new_tokens=DEFERRED_MAX_TOKENS, do_sample=False,
                        max_batch_size=m.max_batch_size, max_batch_tokens=m.max_batch_tokens) if deferred else []
        long_of = dict(zip(deferred, long))
        for k, i in enumerate(gidx):
            opts = items[i]["options"]
            if k in long_of:
                text, mode = long_of[k][0], "deferred"
                choice = restrict_to_options(parse_final_answer(text, opts), opts)
            else:
                text, mode = short[k][0], "immediate"
                choice = restrict_to_options(parse_answer(text, opts), opts)
            df.at[i, "gen_text"], df.at[i, "gen_mode"], df.at[i, "gen_choice"] = text, mode, choice
            df.at[i, "gen_refusal"] = is_refusal(text)
            if m.n_samples > 0:  # one item per call: the sampling seed then depends on the item only
                seed = stable_int(cfg.seed, "ladder-sample", items[i]["item_id"]) % (2**31 - 1)
                texts = generate(lm, [free_g[k]], max_new_tokens=m.gen_max_new_tokens, do_sample=True,
                                 temperature=1.0, num_return_sequences=m.n_samples, seed=seed,
                                 max_batch_size=m.n_samples, max_batch_tokens=m.max_batch_tokens)[0]
                picks = [restrict_to_options(parse_answer(t, opts), opts) for t in texts]
                df.at[i, "samp_n"] = len(picks)
                df.at[i, "samp_target"] = sum(p == items[i]["target_label"] for p in picks)
                df.at[i, "samp_control"] = sum(p == items[i]["control_label"] for p in picks)
                df.at[i, "samp_other"] = sum(p not in (items[i]["target_label"], items[i]["control_label"])
                                             for p in picks)
        timing["gen_s"] += time.time() - t0
    return df


# --------------------------------------------------------------------------- stage summary
def canonical_form(scores: pd.DataFrame) -> str:
    """The letter form that carries more probability at the forced position, by a fixed rule (never chosen by
    looking at results): the form whose logit is higher on average over items and options."""
    sp = np.stack([scores[f"z_sp_{j}"].to_numpy() for j in range(MAX_OPTIONS)], axis=1)
    ns = np.stack([scores[f"z_ns_{j}"].to_numpy() for j in range(MAX_OPTIONS)], axis=1)
    if np.all(np.isnan(sp)):
        return "ns"
    if np.all(np.isnan(ns)):
        return "sp"
    return "sp" if np.nanmean(sp - ns) >= 0 else "ns"  # two logits at one position: the larger is likelier


def summarize(scores: pd.DataFrame, items: Sequence[dict[str, Any]], model_id: str, tier: str, split: str,
              timing: dict[str, float], elapsed: float) -> dict[str, Any]:
    form = canonical_form(scores)
    t = scores["target_idx"].to_numpy()
    c = scores["control_idx"].to_numpy()
    rows = np.arange(len(scores))
    z = np.stack([scores[f"z_{form}_{j}"].to_numpy() for j in range(MAX_OPTIONS)], axis=1)
    L = z[rows, t] - z[rows, c]
    lp = np.stack([scores[f"lp_forced_{j}"].to_numpy() for j in range(MAX_OPTIONS)], axis=1)
    gen = scores[scores["gen_mode"].notna()]
    agree = float("nan")
    if len(gen):  # the forced pick among ALL options of the item (four in the context tier), not target vs control
        want = [LETTERS[int(np.nanargmax(lp[i, : int(scores.at[i, "n_options"])]))] for i in gen.index]
        agree = float(np.mean([g == w for g, w in zip(gen["gen_choice"], want)]))
    diag = scores[scores["cq_forced"].notna()]
    info: dict[str, Any] = {
        "model_id": model_id, "tier": tier, "split": split, "n_items": int(len(scores)),
        "n_units": int(len({it["unit_id"] for it in items})), "n_tokens": int(scores["n_tokens"].sum()),
        "elapsed_s": round(elapsed, 1), "timing": {k: round(v, 1) for k, v in timing.items()},
        "letter_form": form,
        "c_forced": round(float(scores["c_forced"].mean()), 4), "c_free": round(float(scores["c_free"].mean()), 4),
        "share_low_coverage": round(float((scores["c_forced"] < LOW_COVERAGE).mean()), 4),
        "mean_abs_L": round(float(np.nanmean(np.abs(L))), 4),
        "n_diag": int(len(diag)), "n_gen_checked": int(len(gen)),
        "cq_forced": round(float(diag["cq_forced"].mean()), 4) if len(diag) else None,
        "cq_free": round(float(diag["cq_free"].mean()), 4) if len(diag) else None,
        "name_share_forced": round(float(diag["name_share_forced"].mean()), 4) if len(diag) else None,
        "gen_agreement": None if agree != agree else round(agree, 4),
        "gen_immediate_share": round(float((gen["gen_mode"] == "immediate").mean()), 4) if len(gen) else None,
        "gen_refusal_share": round(float(gen["gen_refusal"].astype(bool).mean()), 4) if len(gen) else None,
        "letter_collision_share": round(float(scores["letter_collision"].mean()), 4),
        "head_dev": round(float(scores["head_dev"].max()), 4),
    }
    info["coverage_warning"] = bool(info["c_forced"] < LOW_COVERAGE)
    info["head_warning"] = bool(info["head_dev"] > HEAD_DEV_WARN)
    return info


def write_summary(sdir, info: dict[str, Any]) -> None:
    lines = [f"# Ladder readout: {info['model_id']} ({info['tier']}, {info['split']})", "",
             f"- items: {info['n_items']} in {info['n_units']} units; prompt tokens: {info['n_tokens']}; "
             f"elapsed {info['elapsed_s']:.0f}s (timing {info['timing']})",
             f"- letter form used by the analysis: `{info['letter_form']}` (sp = \" A\", ns = \"A\")",
             f"- letter mass at the forced position (c_forced): {info['c_forced']}"
             + ("  **WARNING: below 0.9, the letter readout is partial**" if info["coverage_warning"] else ""),
             f"- letter mass at the free position (c_free): {info['c_free']}",
             f"- items with c_forced < {LOW_COVERAGE}: {info['share_low_coverage']:.1%}",
             f"- mean |target - control| letter-logit difference: {info['mean_abs_L']}",
             f"- recomputed vs model logits (head_dev, max over batches; rounding alone gives < 0.25): "
             f"{info['head_dev']}" + ("  **WARNING: above 1, check the numerics of this device**"
                                      if info["head_warning"] else ""), "",
             "## Readout-validity diagnostics (samples; never used for the preference level)", "",
             f"- answer-path sample: {info['n_diag']} items; letters+names mass forced {info['cq_forced']}, "
             f"free {info['cq_free']}; share of the forced mass through names {info['name_share_forced']}",
             f"- decoded sample: {info['n_gen_checked']} items; greedy free answer agrees with the forced letter "
             f"pick (among all options of the item) in {info['gen_agreement']}; immediate answers "
             f"{info['gen_immediate_share']}; refusals "
             f"{info['gen_refusal_share']}",
             f"- units with a name that starts with an option-letter token: {info['letter_collision_share']:.1%} "
             f"(flagged `letter_collision`; always in the answer-path sample)", "",
             "Preference levels, anchor curves and stage comparisons are computed by `python -m rtp.cli analyze` "
             "from `scores.parquet`; this stage only records the readout."]
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- stage
def run_ladder(cfg: Config, model_id: str, tier: str = "core", split: str = "dev", unseal: bool = False,
               force: bool = False, prompt_mode: str = "auto"):
    from rtp.models import is_cached, load_model

    check_split_access(split, unseal)
    items = load_ladder_items(cfg, tier, split, unseal)
    sdir = stage_dir(cfg.name, model_id, ladder_stage_name(tier, split))
    shash = stages.stage_hash(cfg, "ladder", model_id, split, prompt_mode, tier=tier)
    if io.finished_stage(sdir, shash, force):
        log.info("ladder (%s) already finished for %s with this config and code (%s); skipping", tier, model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta={**stages.stage_meta("ladder", model_id, split, prompt_mode), "tier": tier})
    stages.check_resume_environment(sdir, force)
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    timing = {"load_s": 0.0, "primary_s": 0.0, "diag_s": 0.0, "gen_s": 0.0}
    try:
        if not items:
            raise ValueError(f"no ladder items in tier '{tier}', split '{split}'")
        state: dict[str, Any] = {}
        collide_cache: dict[str, bool] = {}

        def process(chunk, idx):
            if "lm" not in state:
                state["cached"] = is_cached(model_id)  # before the load: was the checkpoint on local disk?
                tl = time.time()
                state["lm"] = apply_prompt_mode(load_model(model_id), prompt_mode)
                timing["load_s"] += time.time() - tl
                stages.record_model_revision(sdir, state["lm"].model)
                state["forms"] = letter_forms(state["lm"].tokenizer)
                io.write_json(sdir / "letter_tokens.json", state["forms"])
            return score_items(state["lm"], chunk, state["forms"], cfg, collide_cache, timing)

        n_before = len(stages.finished_shards(sdir))
        io.run_sharded(items, cfg.ladder_measure.shard_size, sdir / "shards", process, desc=f"ladder-{tier}")
        scores = io.load_shards(sdir / "shards")
        if len(scores) != len(items):
            raise RuntimeError(f"expected {len(items)} scored items, found {len(scores)}")
        io.write_parquet(sdir / "scores.parquet", scores)
        info = summarize(scores, items, model_id, tier, split, timing, time.time() - t0)
        info["timing_complete"] = n_before == 0  # False: some shards came from an earlier session
        info["load_cached"] = state.get("cached")  # None: every shard was already done
        write_summary(sdir, info)
        from rtp.provenance import environment

        io.mark_done(sdir, {**info, "finished_with": environment()})
        log.info("ladder (%s) finished: %s", tier, sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir
