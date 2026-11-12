"""Answer-path estimator: which option does the model recommend, counting letter AND name answers.

For every option j the ANSWER PATHS are token sequences that start an answer naming it: letter forms
("A", " A", "**A", " **A", "(A", " (A" and every single token whose stripped text is the letter) and
the option's DISPLAYED name in four forms (name, " name", "**name", " **name"). The target slot's
other identity (the placebo substitute in the real version, the real brand in the placebo version)
is scored as a separate class, ``identity_leak``, and credited to no option.

Path probabilities are computed on a prefix trie: a prefix is expanded (one forward step) only while
its mass is >= ``EPS``; the mass of pruned prefixes is reported as an error bound. EVENT masses make
the classes disjoint: a path's mass minus the mass of its minimal extensions that belong to another
class (letter "D" excludes "D"+"ell" when Dell is an option), and a letter path of option A is
multiplied by 1 - P(next token = space + lowercase word) (the article in "A good choice" is not an
answer). q_j = sum of option j's event masses; coverage c = sum_j q_j.

Two positions come from ONE forward pass: the free assistant start x_free (spec 7.1) and the forced
position after the "Answer:" prefill, x_forced = x_free + suffix tokens (verified per item; otherwise two
passes). Multi-token paths are extended through the KV cache (falls back to full forward passes).

Estimators per item (target index t):
  O3 free:   q_free / c_free
  O4 forced: q_forced / c_forced
  O5 (primary): P = q_free + (1 - c_free) * q_forced / c_forced  -- immediate answers at the spec position,
                 the deferred mass imputed by the committed answer after "Answer:" (an assumption: it is
                 reported, never validated by construction; see the bounds).
  Imputation-free bounds: P_t in [q_free_t, q_free_t + 1 - c_free].
Base (non-chat) checkpoints have one position (text + "\\nAnswer:"); there all three coincide.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from rtp.models import LoadedModel

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
EPS = 1e-4
LETTER_FORMS = ("{L}", " {L}", "**{L}", " **{L}", "({L}", " ({L}")
NAME_FORMS = ("{n}", " {n}", "**{n}", " **{n}")
N_CLASSES = 5  # options A-D + identity leak
LEAK = 4


@dataclass
class ItemPaths:
    """Answer paths of one item: per class a list of (token path, kind) with kind 'letter' | 'name'."""
    classes: list[list[tuple[tuple[int, ...], str]]]
    a_letter: set[tuple[int, ...]] = field(default_factory=set)  # option-A letter paths (article rule)


@lru_cache(maxsize=None)
def _article_ids(tok_key: int, tokenizer: Any) -> frozenset[int]:
    decoded = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    return frozenset(i for i, s in enumerate(decoded) if re.match(r"^ [a-z]", s))


def article_ids(tokenizer: Any) -> frozenset[int]:
    return _article_ids(id(tokenizer), tokenizer)


def build_item_paths(tokenizer: Any, names: Sequence[str], leak_name: str | None, letter_ids: dict[str, list[int]],
                     cache: dict[str, tuple[int, ...]]) -> ItemPaths:
    def toks(s: str) -> tuple[int, ...]:
        if s not in cache:
            cache[s] = tuple(tokenizer(s, add_special_tokens=False)["input_ids"])
        return cache[s]

    classes: list[list[tuple[tuple[int, ...], str]]] = [[] for _ in range(N_CLASSES)]
    for j, L in enumerate(LETTERS):
        seqs = {toks(f.format(L=L)) for f in LETTER_FORMS} | {(i,) for i in letter_ids[L]}
        classes[j].extend((s, "letter") for s in seqs if s)
        classes[j].extend((toks(f.format(n=names[j])), "name") for f in NAME_FORMS if toks(f.format(n=names[j])))
    if leak_name:
        classes[LEAK].extend((toks(f.format(n=leak_name)), "name") for f in NAME_FORMS if toks(f.format(n=leak_name)))
    for c in range(N_CLASSES):  # drop paths that extend a shorter path of the same class (already counted)
        uniq = sorted(set(classes[c]), key=lambda pk: (len(pk[0]), pk[0]))
        kept: list[tuple[tuple[int, ...], str]] = []
        for p, k in uniq:
            if not any(p[: len(q)] == q for q, _ in kept):
                kept.append((p, k))
        classes[c] = kept
    a_letter = {p for p, k in classes[0] if k == "letter"}
    return ItemPaths(classes=classes, a_letter=a_letter)


def needed_nodes(paths: ItemPaths) -> tuple[set[tuple[int, ...]], dict[tuple[int, ...], set[int]]]:
    """Internal trie nodes (proper prefixes, plus option-A letter paths for the article rule) and, per node,
    the next tokens whose probability is needed."""
    children: dict[tuple[int, ...], set[int]] = {}
    for cls in paths.classes:
        for p, _ in cls:
            for k in range(1, len(p)):
                children.setdefault(p[:k], set()).add(p[k])
    for p in paths.a_letter:
        children.setdefault(p, set())
    return set(children), children


def event_masses(paths: ItemPaths, logmass: dict[tuple[int, ...], float], article: dict[tuple[int, ...], float],
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(q per class [5], name-channel mass per class [5]) from path log-masses (missing = pruned = 0)."""
    allp = [(c, p, k) for c, cls in enumerate(paths.classes) for p, k in cls]
    q = np.zeros(N_CLASSES)
    qn = np.zeros(N_CLASSES)
    for c, p, kind in allp:
        if p not in logmass:
            continue
        m = float(np.exp(logmass[p]))
        ext = [r for c2, r, _ in allp if c2 != c and len(r) > len(p) and r[: len(p)] == p]
        minimal = [r for r in ext if not any(len(p) < len(s) < len(r) and r[: len(s)] == s for s in ext)]
        m -= sum(float(np.exp(logmass[r])) for r in minimal if r in logmass)
        if c == 0 and p in paths.a_letter and p in article:
            m *= 1.0 - article[p]
        m = max(m, 0.0)
        q[c] += m
        if kind == "name":
            qn[c] += m
    return q, qn


@dataclass
class PositionResult:
    q: np.ndarray        # [5] event mass per class (A-D, leak)
    q_name: np.ndarray   # [5] of which through name paths
    pruned: float        # mass of pruned frontier prefixes (upper bound on missed answer mass)


def batched_tries(all_paths: Sequence[ItemPaths], root_lps: np.ndarray, expand_batch) -> list[PositionResult]:
    """Breadth-first trie evaluation for MANY items at one position. ``expand_batch(todo)`` takes a list of
    (row, node) and returns, per entry, (dict next-token -> logprob for the needed children, article mass).
    All items' nodes of one depth are expanded together, so the number of forward passes is the trie depth."""
    n = len(all_paths)
    meta = [needed_nodes(p) for p in all_paths]
    logmass: list[dict[tuple[int, ...], float]] = [{} for _ in range(n)]
    article: list[dict[tuple[int, ...], float]] = [{} for _ in range(n)]
    pruned = [0.0] * n
    for r, paths in enumerate(all_paths):
        for node in {p[:1] for cls in paths.classes for p, _ in cls}:
            logmass[r][node] = float(root_lps[r][node[0]])
    depth = 1
    while True:
        todo: list[tuple[int, tuple[int, ...]]] = []
        any_frontier = False
        for r in range(n):
            internal, _ = meta[r]
            for node in internal:
                if len(node) != depth or node not in logmass[r]:
                    continue
                any_frontier = True
                mass = float(np.exp(logmass[r][node]))
                if mass >= EPS:
                    todo.append((r, node))
                else:
                    pruned[r] += mass
        if not any_frontier:
            break
        if todo:
            for (r, node), (child_lp, art) in zip(todo, expand_batch(todo, [meta[r][1].get(node, set())
                                                                           for r, node in todo])):
                article[r][node] = art
                for t, lp in child_lp.items():
                    logmass[r][node + (t,)] = logmass[r][node] + lp
        depth += 1
    out = []
    for r, paths in enumerate(all_paths):
        q, qn = event_masses(paths, logmass[r], article[r])
        out.append(PositionResult(q=q, q_name=qn, pruned=pruned[r]))
    return out


# --------------------------------------------------------------------------- model-facing
def _select_cache(cache: Any, rows: torch.Tensor, keep_len: int | None):
    """A new DynamicCache holding ``rows`` (repeats allowed) of ``cache``, optionally cropped to keep_len."""
    from transformers import DynamicCache

    sel = []
    for k, v in cache.to_legacy_cache():
        k, v = k[rows], v[rows]
        if keep_len is not None:
            k, v = k[:, :, :keep_len], v[:, :, :keep_len]
        sel.append((k, v))
    return DynamicCache.from_legacy_cache(tuple(sel))


def _expand_rows(lm: LoadedModel, cache: Any, mask: torch.Tensor, prefixes: list[list[int]], keep_len: int | None,
                 todo: list[tuple[int, tuple[int, ...]]], needs: list[set[int]], art_ids: torch.Tensor,
                 chunk: int, allow_fallback: bool = True) -> list[tuple[dict[int, float], float]]:
    """Next-token log-probs (needed children only) and article mass after each (row, node), batched.

    All nodes in ``todo`` have the same length (one trie depth). The prefix comes from the KV cache; if the
    cache cannot be sliced (e.g. sliding-window caches), the exact but slower full-sequence path is used.
    """
    from rtp.infer import pad_left

    out: list[tuple[dict[int, float], float]] = []
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        depth = len(part[0][1])
        node_tok = torch.tensor([list(nd) for _, nd in part], dtype=torch.long, device=lm.device)
        try:
            rows = torch.tensor([r for r, _ in part], dtype=torch.long, device=mask.device)
            c = _select_cache(cache, rows, keep_len)
            m = mask[rows][:, :keep_len] if keep_len is not None else mask[rows]
            m = torch.cat([m, torch.ones(len(part), depth, dtype=m.dtype, device=m.device)], dim=1)
            p = (m.cumsum(-1) - 1)[:, -depth:]
            o = lm.model(input_ids=node_tok, attention_mask=m, position_ids=p, past_key_values=c, use_cache=True,
                         logits_to_keep=1)
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:  # a full forward would silently drop the prefix patches
                raise RuntimeError(f"KV-cache extension failed for a patched run ({exc}); cannot score it") from exc
            log.debug("cache extension failed (%s); using full forward passes", exc)
            seqs = [prefixes[r] + list(nd) for r, nd in part]
            ids, mm, pp = pad_left(seqs, lm.tokenizer.pad_token_id, lm.device)
            o = lm.model(input_ids=ids, attention_mask=mm, position_ids=pp, logits_to_keep=1)
        lp = torch.log_softmax(o.logits[:, -1, :].float(), dim=-1)
        art = lp[:, art_ids.to(lp.device)].exp().sum(dim=-1).cpu().numpy()
        lpc = lp.cpu().numpy()
        for j in range(len(part)):
            out.append(({t: float(lpc[j, t]) for t in needs[s + j]}, float(art[j])))
    return out


@torch.inference_mode()
def answer_distributions(lm: LoadedModel, free_ids: Sequence[Sequence[int]], forced_ids: Sequence[Sequence[int]],
                         paths: Sequence[ItemPaths], max_batch_size: int, max_batch_tokens: int,
                         patches_for=None, node_chunk: int = 64) -> list[dict[str, PositionResult]]:
    """Per item {"free": PositionResult, "forced": PositionResult}.

    ``patches_for(batch_indices, padded_len, left_offsets)`` may return patches (see rtp.patching) that are
    applied during the PREFIX forward only; extensions read the patched KV cache (used by `locate`/M4).
    """
    from rtp.infer import pad_left, run_batched
    from rtp.patching import patched

    tok = lm.tokenizer
    art_ids = torch.tensor(sorted(article_ids(tok)), dtype=torch.long)
    pad_id = tok.pad_token_id

    def run_group(batch: list[int], seqs: list[list[int]], ns: int) -> list[dict[str, PositionResult]]:
        ids, mask, pos = pad_left(seqs, pad_id, lm.device)
        L = ids.shape[1]
        offsets = [L - len(x) for x in seqs]
        patches = patches_for(batch, L, offsets) if patches_for else None
        with patched(lm.model, patches):
            out = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                           logits_to_keep=ns + 1)
        logp = torch.log_softmax(out.logits.float(), dim=-1).cpu().numpy()
        cache = out.past_key_values
        res: dict[str, list[PositionResult]] = {}
        for position, k, keep in (("forced", -1, None), ("free", 0, L - ns if ns else None)):
            if position == "free" and ns == 0:
                res["free"] = res["forced"]
                continue
            prefixes = [x if keep is None else x[: len(x) - ns] for x in seqs]

            def expand_batch(todo, needs, keep=keep, prefixes=prefixes):  # extensions are never patched
                return _expand_rows(lm, cache, mask, prefixes, keep, todo, needs, art_ids, node_chunk,
                                    allow_fallback=not patches)
            res[position] = batched_tries([paths[i] for i in batch], logp[:, k, :], expand_batch)
        return [{"free": res["free"][r], "forced": res["forced"][r]} for r in range(len(batch))]

    def fn(batch: list[int]) -> list[dict[str, PositionResult]]:
        ns_list = [len(forced_ids[i]) - len(free_ids[i]) for i in batch]
        ok = [list(forced_ids[i][: len(free_ids[i])]) == list(free_ids[i]) for i in batch]
        if all(ok) and len(set(ns_list)) == 1:
            return run_group(batch, [list(forced_ids[i]) for i in batch], ns_list[0])
        out = []  # rare: the suffix is not a token extension; score the two positions separately
        for i in batch:
            fr = run_group([i], [list(free_ids[i])], 0)[0]["forced"]
            fo = run_group([i], [list(forced_ids[i])], 0)[0]["forced"]
            out.append({"free": fr, "forced": fo})
        return out

    return run_batched([list(x) for x in forced_ids], fn, max_batch_size, max_batch_tokens, desc="answers")


# --------------------------------------------------------------------------- estimators
def logit(p: np.ndarray | float, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def estimators(free: PositionResult, forced: PositionResult, target: int) -> dict[str, float]:
    """O5 (primary), O3, O4 for one item, plus coverage, channels, leak, bounds."""
    qf, qx = free.q[:4], forced.q[:4]
    cf, cx = float(qf.sum()), float(qx.sum())
    o3 = qf / cf if cf > 0 else np.full(4, 0.25)
    o4 = qx / cx if cx > 0 else np.full(4, 0.25)
    p = qf + (1.0 - min(cf, 1.0)) * o4
    p = p / p.sum() if p.sum() > 0 else np.full(4, 0.25)
    lo, hi = float(qf[target]), float(qf[target] + max(0.0, 1.0 - cf))
    return {"P": p, "s_target": float(logit(p[target])), "s_target_free": float(logit(o3[target])),
            "s_target_forced": float(logit(o4[target])), "c_free": cf, "c_forced": cx,
            "s_target_lo": float(logit(lo)), "s_target_hi": float(logit(hi)),
            "target_mass_free": float(qf[target]), "target_name_free": float(free.q_name[target]),
            "name_share_free": float(free.q_name[:4].sum() / cf) if cf > 0 else float("nan"),
            "name_share_forced": float(forced.q_name[:4].sum() / cx) if cx > 0 else float("nan"),
            "leak_free": float(free.q[LEAK]), "leak_forced": float(forced.q[LEAK]),
            "pruned_free": free.pruned, "pruned_forced": forced.pruned}
