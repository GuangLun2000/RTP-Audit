"""Batched inference helpers: length-bucketed batching, left padding, OOM halving, generation.

Left padding is used everywhere so the last position is the next-token position. Explicit
``position_ids`` are passed so that padded and unpadded runs see identical positions.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from rtp.models import LoadedModel

log = logging.getLogger(__name__)


def make_batches(lengths: Sequence[int], max_batch_size: int, max_batch_tokens: int) -> list[list[int]]:
    """Group indices into batches sorted by length so that ``batch_size * max_len <= max_batch_tokens``."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (len(cur) + 1 > max_batch_size or new_max * (len(cur) + 1) > max_batch_tokens):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def pad_left(seqs: Sequence[Sequence[int]], pad_id: int, device: Any = "cpu"):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, s in enumerate(seqs):
        if len(s):
            ids[row, width - len(s):] = torch.tensor(list(s), dtype=torch.long)
            mask[row, width - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_batched(seqs: Sequence[Sequence[int]], fn: Callable[[list[int]], list[Any]],
                max_batch_size: int, max_batch_tokens: int, desc: str = "batches") -> list[Any]:
    """Apply ``fn(list_of_indices) -> list_of_results`` over length-bucketed batches.

    On CUDA OOM the failing batch is split in half and retried once; a second OOM propagates.
    Results are returned in the original order of ``seqs``.
    """
    from tqdm.auto import tqdm

    out: list[Any] = [None] * len(seqs)
    batches = make_batches([len(s) for s in seqs], max_batch_size, max_batch_tokens)
    for batch in tqdm(batches, desc=desc, leave=False, mininterval=5.0):
        oom = False
        try:
            res = fn(batch)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not _is_oom(exc) or len(batch) == 1:
                raise
            oom = True  # retry OUTSIDE the except clause: the traceback pins the failed batch's tensors
        if oom:
            log.warning("OOM on a batch of %d; halving and retrying once", len(batch))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            half = len(batch) // 2
            res = fn(batch[:half]) + fn(batch[half:])
        for i, r in zip(batch, res):
            out[i] = r
    return out


@torch.inference_mode()
def last_token_reduce(lm: LoadedModel, seqs: Sequence[Sequence[int]],
                      reduce: Callable[[torch.Tensor], torch.Tensor],
                      max_batch_size: int, max_batch_tokens: int) -> np.ndarray:
    """Run a forward pass and reduce the last-position logits [B, V] -> [B, k] with ``reduce``."""
    pad_id = lm.tokenizer.pad_token_id

    def fn(batch: list[int]) -> list[np.ndarray]:
        ids, mask, pos = pad_left([seqs[i] for i in batch], pad_id, lm.device)
        logits = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits
        red = reduce(logits[:, -1, :].float())
        return list(red.cpu().numpy())

    return np.stack(run_batched(seqs, fn, max_batch_size, max_batch_tokens, desc="forward"))


@torch.inference_mode()
def continuation_logprobs(lm: LoadedModel, prefixes: Sequence[Sequence[int]],
                          continuations: Sequence[Sequence[int]],
                          max_batch_size: int, max_batch_tokens: int) -> np.ndarray:
    """Sum of log p(continuation tokens | prefix) for each (prefix, continuation) pair."""
    seqs = [list(p) + list(c) for p, c in zip(prefixes, continuations)]
    cont_len = [len(c) for c in continuations]
    if min(cont_len) < 1:
        raise ValueError("every continuation needs at least one token")
    pad_id = lm.tokenizer.pad_token_id

    def fn(batch: list[int]) -> list[float]:
        ids, mask, pos = pad_left([seqs[i] for i in batch], pad_id, lm.device)
        k = max(cont_len[i] for i in batch)
        logits = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=k + 1).logits
        logp = torch.log_softmax(logits.float(), dim=-1)  # [B, k+1, V]; position j predicts token j+1
        res = []
        for row, i in enumerate(batch):
            c = cont_len[i]
            targets = ids[row, -c:]
            preds = logp[row, -c - 1:-1, :]
            res.append(float(preds.gather(-1, targets[:, None]).sum()))
        return res

    return np.asarray(run_batched(seqs, fn, max_batch_size, max_batch_tokens, desc="score"), dtype=np.float64)


@torch.inference_mode()
def generate(lm: LoadedModel, seqs: Sequence[Sequence[int]], max_new_tokens: int, do_sample: bool,
             temperature: float = 1.0, num_return_sequences: int = 1, seed: int = 0,
             max_batch_size: int = 32, max_batch_tokens: int = 32768) -> list[list[str]]:
    """Generate continuations; returns ``num_return_sequences`` decoded strings per prompt.

    Sampling uses the raw model distribution (no top-k/top-p/repetition penalty) so that
    model-specific generation defaults do not leak into the measurement.
    """
    from transformers import GenerationConfig

    tok = lm.tokenizer
    eos = lm.model.generation_config.eos_token_id
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample, num_return_sequences=num_return_sequences,
        pad_token_id=tok.pad_token_id, eos_token_id=eos,
        **({"temperature": temperature, "top_k": 0, "top_p": 1.0} if do_sample else {}),
    )
    eff_bs = max(1, max_batch_size // max(1, num_return_sequences))

    def fn(batch: list[int]) -> list[list[str]]:
        ids, mask, _ = pad_left([seqs[i] for i in batch], tok.pad_token_id, lm.device)
        if do_sample:
            torch.manual_seed(seed + batch[0])
        # use_model_defaults=False: since transformers 4.50, values equal to the global defaults (do_sample=False,
        # temperature=1.0, top_p=1.0, ...) would otherwise be replaced by the checkpoint's generation_config
        # (e.g. OLMo 3: do_sample=True, temperature=0.6), silently turning greedy decoding into sampling.
        out = lm.model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg,
                                use_model_defaults=False)
        new = out[:, ids.shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        n = num_return_sequences
        return [texts[j * n:(j + 1) * n] for j in range(len(batch))]

    return run_batched(seqs, fn, eff_bs, max_batch_tokens, desc="generate")
