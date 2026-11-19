"""Answer parsing and prompt rendering shared by the choice stages.

Kept apart from ``rtp.measure.choice`` so that the ladder stage (rtp.measure.ladder) can use the same
parsers without importing the legacy four-option measurement code.

A letter counts only in an answer-like position; free answers may name an option by letter or by brand
name, immediately or after some prose (see ``parse_answer`` / ``parse_final_answer``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from rtp.models import LoadedModel, chat_prompt, encode
from rtp.rng import rng_for

LETTERS = ("A", "B", "C", "D")
COMPLETION_SUFFIX = "\nAnswer:"  # base (non-chat) checkpoints: plain-text continuation
ANSWER_PREFILL = "Answer:"  # chat checkpoints: assistant turn prefilled so the next token is the letter
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


# A free answer that OPENS by declining to choose ("I'm sorry, but I can't provide a recommendation", "As an AI, I
# don't have personal preferences"). Tulu-3-8B-SFT does this often at evidence 0. O5 treats such mass like any
# deferred answer (imputed from the forced position), so the refusal rate is reported per level and version.
_REFUSAL_RE = re.compile(r"(?i)^[\s\"'*]*(?:i'?m\s+sorry|i\s+am\s+sorry|sorry,|i\s+(?:can(?:no|')t|can\s+not|am\s+unable|"
                         r"won't|do\s+not\s+have\s+personal|don't\s+have\s+personal)|as\s+an\s+ai\b|unfortunately,?\s+i\b)")


def is_refusal(text: str | None) -> bool:
    return bool(text) and _REFUSAL_RE.match(_ascii(text).replace("\u2019", "'")) is not None


def restrict_to_options(label: str | None, options: Sequence[dict[str, Any]]) -> str | None:
    """A parsed answer counts only if it is one of the item's option labels (two-option items: "C" is no answer)."""
    return label if label is not None and any(o["label"] == label for o in options) else None


# --------------------------------------------------------------------------- brand matching in free text
def _norm_keep_case(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("'", "").replace("’", "")
    return re.sub(r"[^0-9A-Za-z]+", " ", t).strip()


class BrandMatcher:
    """Map a free-text answer to a brand: whole-word, accent/punctuation-insensitive, case-insensitive
    unless the brand is flagged ``match_case`` (then the listed spellings and their ALL-CAPS forms match). A
    trailing "s" is tolerated so that possessives and
    plurals ("Nike's" -> "Nikes", "Reeboks") match. Earliest match wins; ties go to the longest variant."""

    def __init__(self, entries: list[tuple[str, list[str], bool]]):
        self.patterns: list[tuple[re.Pattern, str, bool, int]] = []
        for entity, variants, match_case in entries:
            spellings = [entity, *variants]
            if match_case:  # case-sensitive brands still match when written in capitals ("VISA", "CHASE")
                spellings += [x.upper() for x in spellings]
            for v in dict.fromkeys(spellings):
                nv = _norm_keep_case(v)
                if not nv:
                    continue
                key = nv if match_case else nv.lower()
                pat = re.compile(r"(?<![0-9A-Za-z])" + re.escape(key) + r"s?(?![0-9A-Za-z])")
                self.patterns.append((pat, entity, match_case, len(key)))

    def match(self, text: str) -> str | None:
        cs = _norm_keep_case(text or "")
        ci = cs.lower()
        best: tuple[int, int, str] | None = None
        for pat, entity, match_case, length in self.patterns:
            m = pat.search(cs if match_case else ci)
            if m and (best is None or (m.start(), -length) < (best[0], best[1])):
                best = (m.start(), -length, entity)
        return best[2] if best else None
