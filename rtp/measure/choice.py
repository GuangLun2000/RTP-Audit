"""Option-letter scoring with rotation averaging (spec §7.1).

For each item: render the user message with the model's chat template (base models: plain
text + "\\nAnswer:"), take next-token logits, merge every token whose decoded text minus
whitespace equals a letter (e.g. "A" and " A") with logsumexp, renormalise over A-D, and
compute ``s_target = log(p / (1 - p))`` with ``p`` clipped to [1e-4, 1 - 1e-4]. A fixed 2%
sample is also decoded greedily (<= 4 tokens) to check that the letter argmax agrees with
what the model actually writes.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtp import io
from rtp.config import Config
from rtp.data.build import load_catalogue, load_choice_items
from rtp.data.choice_items import LETTERS
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.models import LoadedModel, apply_prompt_mode, chat_prompt, encode
from rtp.paths import stage_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

P_CLIP = 1e-4
COMPLETION_SUFFIX = "\nAnswer:"  # base (non-chat) checkpoints: plain-text continuation
RESULT_COLUMNS = ["item_id", "model_id", "p_A", "p_B", "p_C", "p_D", "s_target", "gen_letter"]
LOW_LETTER_MASS = 0.05  # below this mean mass on A-D the renormalised letter scores are mostly noise
# A letter counts only in an answer-like position: after an explicit label ("Answer:", "The answer is",
# "Option"), at the very start, or right after "pick/choose/go with/select/recommend". Only a bare leading
# "A" followed by a lowercase word ("A good choice") is read as the English article, not as an answer.
_MARK = r"[\s\"'*_(\[]*"
_LABEL_RE = re.compile(r"^" + _MARK + r"(?i:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice))" + _MARK
                       + r"(?i:is)?" + _MARK + r"[:\-]?" + _MARK + r"([ABCD])(?![A-Za-z0-9])")
_BARE_RE = re.compile(r"^" + _MARK + r"(?:([BCD])|(A)(?!\s+[a-z]))(?![A-Za-z0-9])")
_LETTER_VERB_RE = re.compile(r"(?i:\b(?:pick|choose|go\s+with|select|recommend)\s+(?:option\s+)?)" + _MARK
                             + r"([ABCD])(?![A-Za-z0-9])")


# --------------------------------------------------------------------------- estimators
def letter_token_ids(tokenizer: Any, letters: Sequence[str] = LETTERS) -> dict[str, list[int]]:
    """All non-special token ids whose decoded text, stripped of whitespace, equals each letter."""
    special = set(tokenizer.all_special_ids or [])
    n = len(tokenizer)
    decoded = tokenizer.batch_decode([[i] for i in range(n)])
    out: dict[str, list[int]] = {L: [] for L in letters}
    for tid, text in enumerate(decoded):
        if tid in special:
            continue
        s = text.strip()
        if s in out:
            out[s].append(tid)
    missing = [L for L, ids in out.items() if not ids]
    if missing:
        raise ValueError(f"tokenizer has no single-token form for letters {missing}")
    return out


def letter_logprobs(logits: torch.Tensor, letter_ids: dict[str, list[int]]) -> torch.Tensor:
    """[B, V] logits -> [B, 4] log-probabilities, each the logsumexp over that letter's token variants."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    cols = [torch.logsumexp(logp[:, ids], dim=-1) for ids in letter_ids.values()]
    return torch.stack(cols, dim=-1)


def normalize_letters(letter_lp: np.ndarray) -> np.ndarray:
    """Renormalise per-letter log-probs over the four letters -> probabilities summing to 1."""
    lp = np.asarray(letter_lp, dtype=np.float64)
    m = lp.max(axis=-1, keepdims=True)
    e = np.exp(lp - m)
    return e / e.sum(axis=-1, keepdims=True)


def s_target(p_target: np.ndarray | float, eps: float = P_CLIP) -> np.ndarray:
    p = np.clip(np.asarray(p_target, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def parse_letter(text: str | None) -> str | None:
    """Answer letter of a short generation ('B', ' C.', 'Answer: D', '(A)', "I'd pick C"); None otherwise.

    The article in 'A good choice' is not read as an answer.
    """
    if not text:
        return None
    m = _LABEL_RE.match(text)
    if m:
        return m.group(1)
    m = _BARE_RE.match(text)
    if m:
        return m.group(1) or m.group(2)
    m = _LETTER_VERB_RE.search(text)
    return m.group(1) if m else None


def select_gen_check(item_ids: Sequence[str], fraction: float, seed: int) -> set[str]:
    """Deterministic, model-independent sample of items for the greedy consistency check."""
    n = len(item_ids)
    if n == 0 or fraction <= 0:
        return set()
    k = min(n, max(1, int(round(fraction * n))))
    idx = rng_for(seed, "gen_check").choice(n, size=k, replace=False)
    return {item_ids[int(i)] for i in idx}


# --------------------------------------------------------------------------- model scoring
def render_inputs(lm: LoadedModel, user_texts: Sequence[str]) -> list[list[int]]:
    tok = lm.tokenizer
    if lm.is_chat:
        return [encode(tok, chat_prompt(lm, t), True) for t in user_texts]
    return [encode(tok, t + COMPLETION_SUFFIX, False) for t in user_texts]


def score_items(lm: LoadedModel, items: Sequence[dict[str, Any]], letter_ids: dict[str, list[int]],
                gen_ids: set[str], cfg: Config) -> pd.DataFrame:
    m = cfg.measure
    seqs = render_inputs(lm, [it["prompt"] for it in items])

    def reduce(logits: torch.Tensor) -> torch.Tensor:
        llp = letter_logprobs(logits, letter_ids)
        mass = torch.logsumexp(llp, dim=-1, keepdim=True)  # log total prob on any letter token
        return torch.cat([llp, mass], dim=-1)

    from rtp.infer import generate, last_token_reduce

    red = last_token_reduce(lm, seqs, reduce, m.max_batch_size, m.max_batch_tokens)
    lp = red[:, :4]
    p = normalize_letters(lp)
    tgt_idx = np.array([LETTERS.index(it["target_label"]) for it in items])
    df = pd.DataFrame({
        "item_id": [it["item_id"] for it in items],
        "model_id": lm.model_id,
        "p_A": p[:, 0], "p_B": p[:, 1], "p_C": p[:, 2], "p_D": p[:, 3],
        "s_target": s_target(p[np.arange(len(items)), tgt_idx]),
        "gen_letter": pd.Series([None] * len(items), dtype="object"),
        "argmax_letter": [LETTERS[i] for i in p.argmax(axis=1)],
        "letter_mass": np.exp(red[:, 4]),
        "gen_text": pd.Series([None] * len(items), dtype="object"),
    })
    check = [i for i, it in enumerate(items) if it["item_id"] in gen_ids]
    if check:
        texts = generate(lm, [seqs[i] for i in check], max_new_tokens=m.gen_max_new_tokens, do_sample=False,
                         max_batch_size=m.max_batch_size, max_batch_tokens=m.max_batch_tokens)
        for i, t in zip(check, texts):
            df.at[i, "gen_text"] = t[0]
            df.at[i, "gen_letter"] = parse_letter(t[0])
    return df


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
    stage_hash = cfg.stage_hash("measure", model_id, split, prompt_mode)
    if io.finished_stage(sdir, stage_hash, force):
        log.info("measure already finished for %s with this config (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), stage_hash, force=force,
                         extra_meta={"model_id": model_id, "split": split, "stage": "measure",
                                     "prompt_mode": prompt_mode})
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        if not items:
            raise ValueError(f"no choice items in split '{split}'")
        gen_ids = select_gen_check([it["item_id"] for it in items], cfg.measure.gen_check_fraction, cfg.seed)
        state: dict[str, Any] = {}

        def process(chunk, idx):
            if "lm" not in state:
                state["lm"] = apply_prompt_mode(load_model(model_id), prompt_mode)
                state["letters"] = letter_token_ids(state["lm"].tokenizer)
                io.write_json(sdir / "letter_tokens.json", state["letters"])
            return score_items(state["lm"], chunk, state["letters"], gen_ids, cfg)

        io.run_sharded(items, cfg.measure.shard_size, sdir / "shards", process, desc="measure")
        scores = io.load_shards(sdir / "shards")
        if len(scores) != len(items):
            raise RuntimeError(f"expected {len(items)} scored items, found {len(scores)}")
        io.write_parquet(sdir / "scores.parquet", scores)
        info = summarize_measure(cfg, model_id, split, items, catalogue, scores, sdir,
                                 elapsed=time.time() - t0)
        io.mark_done(sdir, info)
        log.info("measure finished: %s", sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir
