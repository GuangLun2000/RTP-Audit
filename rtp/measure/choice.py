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

The spec §7.1 consistency check decodes a fixed 2% sample greedily from the FREE position. Answers that
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
from rtp.models import LoadedModel, apply_prompt_mode, chat_prompt, encode
from rtp.paths import stage_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

P_CLIP = 1e-4
COMPLETION_SUFFIX = "\nAnswer:"  # base (non-chat) checkpoints: plain-text continuation
ANSWER_PREFILL = "Answer:"  # chat checkpoints: assistant turn prefilled so the next token is the letter
RESULT_COLUMNS = ["item_id", "model_id", "p_A", "p_B", "p_C", "p_D", "s_target", "gen_letter"]
LOW_COVERAGE = 0.9  # below this mean option-answer coverage at a position, that position's readout is partial
# A letter counts only in an answer-like position: after an explicit label ("Answer:", "The answer is",
# "Option"), at the very start, or right after "pick/choose/go with/select/recommend". A bare leading "A"
# followed by a lowercase word ("A good choice") is read as the English article, unless that word is an
# answer continuation ("A is the best", "A would suit you"; the same list as the estimator's article rule).
_MARK = r"[\s\"'*_(\[]*"
_LABEL_RE = re.compile(r"^" + _MARK + r"(?i:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice))" + _MARK
                       + r"(?i:is)?" + _MARK + r"[:\-]?" + _MARK + r"([ABCD])(?![A-Za-z0-9])")


def _continuations() -> str:
    from rtp.measure.answers import ANSWER_CONTINUATIONS

    return "|".join(sorted(ANSWER_CONTINUATIONS, key=len, reverse=True))


_ARTICLE_A = r"A\s+(?!(?:" + _continuations() + r")\b)[a-z]"  # "A good ..." but not "A is ..."
_BARE_RE = re.compile(r"^" + _MARK + r"(?:([BCD])|(?!" + _ARTICLE_A + r")(A))(?![A-Za-z0-9])")
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


def _ascii(text: str) -> str:
    """Accents stripped ("L'Oréal" -> "L'Oreal"), curly apostrophes straightened."""
    t = unicodedata.normalize("NFKD", text.replace("\u2019", "'"))
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", _ascii(text).lower().replace("'", "")).strip()


def _name_hit(name: str, text: str, case_sensitive: bool, anywhere: bool) -> int | None:
    """Start of the first whole-word occurrence of ``name`` in ``text`` (a trailing "s" tolerated), or None.
    Case-sensitive names (catalogue ``match_case``: ordinary words such as Sharp, KIND) match as written or in
    capitals only, as in open-ended matching."""
    def keep(x: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", " ", _ascii(x).replace("'", "")).strip()

    forms = [keep(name), keep(name).upper()] if case_sensitive else [_norm(name)]
    hay = keep(text) if case_sensitive else _norm(text)
    best = None
    for f in dict.fromkeys(forms):
        if not f:
            continue
        pat = re.escape(f) + r"s?(?![0-9A-Za-z])"
        m = re.search(r"(?<![0-9A-Za-z])" + pat, hay) if anywhere else re.match(pat, hay)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def parse_answer(text: str | None, options: Sequence[dict[str, Any]],
                 case_sensitive: frozenset[str] | set[str] = frozenset()) -> str | None:
    """Option letter of a FREE answer: the option whose brand name the answer starts with ("A&W is ..."), else an
    answer letter, else the option whose name appears as a whole word (earliest wins), else the option whose
    name the WHOLE short answer is a truncated prefix of (generation stops after a few tokens: "Bev" ->
    "Bevodel", "New Bal" -> "New Balance"; a prose opening such as "The best choice" is not a prefix).
    """
    if not text:
        return None
    t = _norm(text)
    for o in options:  # a name at the very start wins over a letter reading of its first character
        if _name_hit(o["name"], text, o["name"] in case_sensitive, anywhere=False) is not None:
            return o["label"]
    letter = parse_letter(_ascii(text))
    if letter or not t:
        return letter
    best = None
    for o in options:
        pos = _name_hit(o["name"], text, o["name"] in case_sensitive, anywhere=True)
        if pos is not None and (best is None or pos < best[0]):
            best = (pos, o["label"])
    if best:
        return best[1]
    joined = t.replace(" ", "")
    cands = [o["label"] for o in options if len(joined) >= 3 and _norm(o["name"]).replace(" ", "").startswith(joined)]
    return cands[0] if len(cands) == 1 else None


def select_gen_check(item_ids: Sequence[str], fraction: float, seed: int) -> set[str]:
    """Deterministic, model-independent sample of items for the greedy consistency check."""
    n = len(item_ids)
    if n == 0 or fraction <= 0:
        return set()
    k = min(n, max(1, int(round(fraction * n))))
    idx = rng_for(seed, "gen_check").choice(n, size=k, replace=False)
    return {item_ids[int(i)] for i in idx}


# --------------------------------------------------------------------------- model scoring
def render_inputs(lm: LoadedModel, user_texts: Sequence[str], prefill: bool = True) -> list[list[int]]:
    """Scored prompts (``prefill=True``: chat turn prefilled with "Answer:"). ``prefill=False`` renders the
    plain generation prompt used for the free-answer consistency check (base models: unchanged)."""
    tok = lm.tokenizer
    if lm.is_chat:
        return [encode(tok, chat_prompt(lm, t, assistant_prefix=ANSWER_PREFILL if prefill else None), True)
                for t in user_texts]
    return [encode(tok, t + COMPLETION_SUFFIX, False) for t in user_texts]


DEFERRED_MAX_TOKENS = 48  # free answers that start with prose are continued this far to find the final answer
# A concluding cue in a deferred answer: "the answer is C", "the best choice would be Nike", "I recommend B",
# "I'd go with D". A bare "Option B" is not a cue (models enumerate the options before concluding).
_STRONG_CUE = (r"(?i:\b(?:answer(?:\s+is)?|(?:option|choice|pick)\s+(?:is|would\s+be|will\s+be)|"
               r"recommend(?:ation)?(?:\s+is)?|(?:i\s+would|i'd|i\s+will|i'll)\s+(?:choose|pick|go\s+with|select)|"
               r"choose|go\s+with|select)\b)[\s:*\"'(\[\-]*(?i:option\s+|the\s+)?")
_STRONG_LETTER = re.compile(_STRONG_CUE + r"([ABCD])(?![A-Za-z0-9])")
_BARE_LETTER = re.compile(r"(?<![A-Za-z0-9'])([ABCD])(?![A-Za-z0-9'])")


def _name_patterns(options: Sequence[dict[str, Any]],
                   case_sensitive: frozenset[str] | set[str] = frozenset()) -> list[tuple[str, str]]:
    out = []
    for o in options:
        words = [w for w in re.split(r"[^0-9A-Za-z]+", _ascii(o["name"])) if w]
        if not words:  # "Levi's" matches "Levis" / "Levi's"; "New Balance" matches "New-Balance"
            continue
        if o["name"] in case_sensitive:  # as written or in capitals ("Sharp", "SHARP"; not "a sharp picture")
            body = "(?:" + "|".join(r"[^0-9A-Za-z]*".join(map(re.escape, ws)) for ws in
                                    dict.fromkeys([tuple(words), tuple(w.upper() for w in words)])) + ")"
        else:
            body = r"(?i:" + r"[^0-9A-Za-z]*".join(map(re.escape, words)) + ")"
        out.append((o["label"], r"(?<![0-9A-Za-z])" + body + r"(?![0-9A-Za-z])"))
    return out


def parse_final_answer(text: str | None, options: Sequence[dict[str, Any]],
                       case_sensitive: frozenset[str] | set[str] = frozenset()) -> str | None:
    """Option of a longer (deferred) free answer ("Based on the reviews, the best choice is **C**").

    The LAST option named after a concluding cue (letter or brand name) wins. Without a cue, the answer is
    accepted only if the text mentions exactly ONE option (by name or bare letter; the article in "A good
    choice" is not a letter); an answer that compares several options without concluding is unparsed (None),
    which the consistency check counts as disagreement.
    """
    if not text:
        return None
    t = _ascii(text)
    names = _name_patterns(options, case_sensitive)
    spans = [m.span() for _, pat in names for m in re.finditer(pat, t)]

    def in_name(pos: int) -> bool:  # the "A" of "A&W" is part of a name, not a letter answer
        return any(a <= pos < b for a, b in spans)

    strong = [(m.end(), m.group(1)) for m in _STRONG_LETTER.finditer(t) if not in_name(m.start(1))]
    for label, pat in names:
        strong += [(m.end(), label) for m in re.finditer(_STRONG_CUE + pat, t)]
    if strong:
        return max(strong)[1]
    mentioned = {m.group(1) for m in _BARE_LETTER.finditer(t)
                 if not in_name(m.start()) and not (m.group(1) == "A" and re.match(_ARTICLE_A, t[m.start():]))}
    mentioned |= {label for label, pat in names if re.search(pat, t)}
    return mentioned.pop() if len(mentioned) == 1 else None


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
    check = [i for i, it in enumerate(items) if it["item_id"] in gen_ids]
    if check:
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
