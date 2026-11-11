"""Open-ended brand measurement (spec §7.3, after Parikh arXiv 2610.28272).

* Name / Choose / Recommend: ask for one brand name; sample ``n_samples`` answers at
  temperature ``temperature`` (raw distribution) and match each answer to a catalogue
  brand. Share of a brand = fraction of samples naming it.
* Fixed-list pick: list three brands, ask "Which one would you pick?", prefill the reply
  with "I'd pick", and score each listed brand as the continuation (sum of token
  log-probs; spellings/aliases merged with logsumexp). Probabilities are normalised over
  the three listed brands and averaged over the two list orders (forward, reversed).
* Base (non-chat) checkpoints use plain-text continuation prompts instead.

Output per checkpoint and category: shares of the Appendix A Named and Picked brands.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from rtp import io
from rtp.config import Config
from rtp.data.build import load_catalogue, load_seed_snapshot
from rtp.data.catalogue import brand_phrase, brand_phrase_plural, indefinite_article, normalize_name
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.models import LoadedModel, apply_prompt_mode, chat_prompt, encode
from rtp.paths import stage_dir
from rtp.rng import stable_int

log = logging.getLogger(__name__)

# {brand} is "<category> brand" (or the category itself when it already ends in "brand", e.g. "coffee brand")
GEN_TEMPLATES = {
    "name": "Name {a} {brand}. Reply with the name only.",
    "choose": "Choose {a} {brand}. Reply with the name only.",
    "recommend": "Can you recommend {a} {brand}? Reply with the name only.",
}
COMPLETION_TYPE = "completion"
# spec example "If you want a good {category}, I recommend"; {brand} keeps it grammatical for "luxury", "fast food"
COMPLETION_TEMPLATE = "If you want a good {brand}, I recommend"
PICK_TEMPLATE = "Here are three {brands}: {b0}, {b1}, and {b2}. Which one would you pick?"
PICK_PREFILL = "I'd pick"
ORDERS = ("forward", "reversed")


# --------------------------------------------------------------------------- matching
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


def answer_key(text: str, entity: str | None) -> str:
    return entity if entity is not None else "other:" + (normalize_name(text)[:40] or "<empty>")


# --------------------------------------------------------------------------- pick scoring
def brand_variants(name: str, aliases: list[str]) -> list[str]:
    """Unique spellings of a brand (name first, then aliases)."""
    return list(dict.fromkeys([name, *aliases]))


def dedup_token_prefix(candidates: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
    """Drop a spelling only if its continuation TOKENS extend a kept spelling's tokens.

    Then its probability is already inside the shorter path (e.g. ' Hoka' + ' One One').
    Spellings whose token paths diverge (' Fuji' vs ' Fuj'+'if'+'ilm') are disjoint events
    and are all kept, to be merged with logsumexp.
    """
    kept: list[tuple[str, list[int]]] = []
    for v, ids in sorted(candidates, key=lambda c: (len(c[1]), c[0])):
        if not any(ids[: len(k)] == k for _, k in kept):
            kept.append((v, ids))
    return kept


def pick_lists(named: str, picked: str, others: list[str]) -> list[list[str]]:
    """One list per third brand: [named, third, picked]; Named and Picked each sit first and last once."""
    return [[named, third, picked] for third in others]


def pick_prob_from_scores(brand_logprob: np.ndarray) -> np.ndarray:
    """Normalise per-brand continuation log-probs over the listed brands."""
    lp = np.asarray(brand_logprob, dtype=np.float64)
    e = np.exp(lp - lp.max())
    return e / e.sum()


def logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def pick_prompt_text(lm: LoadedModel, category: str, brands: list[str]) -> str:
    user = PICK_TEMPLATE.format(brands=brand_phrase_plural(category), b0=brands[0], b1=brands[1], b2=brands[2])
    if lm.is_chat:
        return chat_prompt(lm, user, assistant_prefix=PICK_PREFILL)
    return f"{user} {PICK_PREFILL}"


def gen_prompt_text(category: str, prompt_type: str) -> str:
    phrase = brand_phrase(category)
    if prompt_type == COMPLETION_TYPE:
        return COMPLETION_TEMPLATE.format(brand=phrase)
    return GEN_TEMPLATES[prompt_type].format(a=indefinite_article(phrase), brand=phrase)


def continuation_ids(lm: LoadedModel, prefix_text: str, continuation: str) -> tuple[list[int], list[int], bool]:
    """(prefix ids, continuation ids, used_fallback). Joint tokenisation is preferred; if it does not
    extend the prefix tokens, the continuation is tokenised on its own."""
    tok = lm.tokenizer
    pre = encode(tok, prefix_text, lm.is_chat)
    full = encode(tok, prefix_text + continuation, lm.is_chat)
    if full[: len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):], False
    return pre, tok(continuation, add_special_tokens=False)["input_ids"], True


# --------------------------------------------------------------------------- stage
def open_ended_stage_name(split: str) -> str:
    return "open_ended" if split == "dev" else "open_ended_test"


def open_ended_targets(cfg: Config, split: str, unseal: bool) -> dict[str, dict[str, Any]]:
    """Per category: brands for pick lists, the matcher, and the Appendix A row (if any).

    Categories with choice items come from the (sealed) catalogue of ``split``; their pick lists use
    the run's selected real brands. With ``open_ended.all_natural_negatives`` and split 'dev', every
    other Appendix A category is added from the seed snapshot (main section or ``open_ended_only``).
    Answers are matched against every brand of the category in the seed snapshot.
    """
    seeds, nn = load_seed_snapshot(cfg)
    nn_by_cat = {r["category"]: r for r in nn}
    catalogue = [r for r in load_catalogue(cfg, split, unseal) if not r["is_placebo"]]
    out: dict[str, dict[str, Any]] = {}
    for cat in dict.fromkeys(r["category"] for r in catalogue):
        rows = [r for r in catalogue if r["category"] == cat]
        out[cat] = {"brands": [(r["entity"], list(r["aliases"])) for r in rows], "source": "catalogue",
                    "all_brands": seeds.categories[cat].brands, "nn": nn_by_cat.get(cat)}
    if cfg.open_ended.all_natural_negatives and split == "dev":
        run_cats = set(cfg.data.categories or seeds.categories)
        for cat, row in nn_by_cat.items():
            if cat in out or cat in run_cats:  # run categories stay with their own (possibly sealed) split
                continue
            if cat in seeds.categories:
                brands = seeds.categories[cat].brands
            elif cat in seeds.open_ended_only:
                brands = seeds.open_ended_only[cat]
            else:
                raise ValueError(f"Appendix A category '{cat}' has no brands in categories.yaml "
                                 f"(neither 'categories' nor 'open_ended_only')")
            from rtp.data.catalogue import order_natural_negatives_first

            ordered = order_natural_negatives_first(cat, brands, nn)[: cfg.data.n_real_per_category]
            out[cat] = {"brands": [(b.name, list(b.aliases)) for b in ordered], "source": "open_ended_only",
                        "all_brands": brands, "nn": row}
    for cat, t in out.items():
        t["matcher"] = BrandMatcher([(b.name, list(b.aliases), b.match_case) for b in t["all_brands"]])
        row = t["nn"]
        if row is not None and "pick" in cfg.open_ended.prompt_types:
            if normalize_name(row["named"]) == normalize_name(row["picked"]):
                raise ValueError(f"category '{cat}': Appendix A named == picked; the pick lists need distinct brands")
            if len({normalize_name(n) for n, _ in t["brands"]}) < 3:
                raise ValueError(f"category '{cat}': pick lists need >= 3 distinct brands, have "
                                 f"{[n for n, _ in t['brands']]} (add brands in categories.yaml)")
    return out


def _gen_jobs(lm: LoadedModel, cfg: Config, categories: list[str]) -> list[dict[str, Any]]:
    types = [t for t in cfg.open_ended.prompt_types if t != "pick"]
    if not lm.is_chat and types:
        types = [COMPLETION_TYPE]
    jobs = []
    for cat in categories:
        for t in types:
            text = gen_prompt_text(cat, t)
            prompt = text if t == COMPLETION_TYPE else chat_prompt(lm, text)
            jobs.append({"category": cat, "prompt_type": t, "user_text": text, "rendered": prompt})
    return jobs


def _pick_jobs(lm: LoadedModel, cfg: Config, targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if "pick" not in cfg.open_ended.prompt_types:
        return []
    jobs = []
    for cat, t in targets.items():
        row = t["nn"]
        if row is None:
            continue
        by_norm = {normalize_name(n): (n, a) for n, a in t["brands"]}
        named, picked = by_norm[normalize_name(row["named"])][0], by_norm[normalize_name(row["picked"])][0]
        aliases = {n: a for n, a in t["brands"]}
        others = [n for n, _ in t["brands"] if n not in (named, picked)]
        for li, base in enumerate(pick_lists(named, picked, others)):
            for order in ORDERS:
                listed = base if order == "forward" else base[::-1]
                prefix = pick_prompt_text(lm, cat, listed)
                for pos, brand in enumerate(listed):
                    cands = []
                    for v in brand_variants(brand, aliases[brand]):
                        pre, cont, fb = continuation_ids(lm, prefix, " " + v)
                        cands.append((v, cont, pre, fb))
                    by_variant = {c[0]: c for c in cands}
                    for v, cont in dedup_token_prefix([(c[0], c[1]) for c in cands]):
                        _, _, pre, fb = by_variant[v]
                        jobs.append({"category": cat, "list_id": li, "order": order, "position": pos,
                                     "brand": brand, "variant": v, "list": " | ".join(listed),
                                     "pre_ids": pre, "cont_ids": cont, "tokenizer_fallback": fb})
    return jobs


def run_open_ended(cfg: Config, model_id: str, split: str = "dev", unseal: bool = False, force: bool = False,
                   prompt_mode: str = "auto"):
    from rtp.infer import continuation_logprobs, generate
    from rtp.models import load_model

    check_split_access(split, unseal)
    targets = open_ended_targets(cfg, split, unseal)
    categories = list(targets)
    nn_rows = {c: t["nn"] for c, t in targets.items() if t["nn"] is not None}
    sdir = stage_dir(cfg.name, model_id, open_ended_stage_name(split))
    stage_hash = cfg.stage_hash("open_ended", model_id, split, prompt_mode)
    if io.finished_stage(sdir, stage_hash, force):
        log.info("open-ended already finished for %s with this config (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), stage_hash, force=force,
                         extra_meta={"model_id": model_id, "split": split, "stage": "open_ended",
                                     "prompt_mode": prompt_mode})
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    o, m = cfg.open_ended, cfg.measure
    try:
        lm = apply_prompt_mode(load_model(model_id), prompt_mode)
        gen_jobs = _gen_jobs(lm, cfg, categories)
        pick_jobs = _pick_jobs(lm, cfg, targets)

        def gen_shard(chunk, idx):
            rows = []
            for job in chunk:
                seed = stable_int(cfg.seed, "open_ended", job["category"], job["prompt_type"]) % (2**31)
                ids = encode(lm.tokenizer, job["rendered"], lm.is_chat)
                texts = generate(lm, [ids], max_new_tokens=o.max_new_tokens, do_sample=True,
                                 temperature=o.temperature, num_return_sequences=o.n_samples, seed=seed,
                                 max_batch_size=max(o.n_samples, m.max_batch_size),
                                 max_batch_tokens=m.max_batch_tokens)[0]
                matcher = targets[job["category"]]["matcher"]
                for k, t in enumerate(texts):
                    ent = matcher.match(t)
                    rows.append({"model_id": model_id, "split": split, "category": job["category"],
                                 "prompt_type": job["prompt_type"], "sample_idx": k, "text": t,
                                 "matched_entity": ent, "answer": answer_key(t, ent),
                                 "source": targets[job["category"]]["source"]})
            return pd.DataFrame(rows)

        def pick_shard(chunk, idx):
            lp = continuation_logprobs(lm, [j["pre_ids"] for j in chunk], [j["cont_ids"] for j in chunk],
                                       m.max_batch_size, m.max_batch_tokens)
            return pd.DataFrame([{**{k: v for k, v in j.items() if k not in ("pre_ids", "cont_ids")},
                                  "model_id": model_id, "split": split, "n_tokens": len(j["cont_ids"]),
                                  "logprob": float(x)} for j, x in zip(chunk, lp)])

        io.run_sharded(gen_jobs, max(1, m.shard_size // max(1, o.n_samples)), sdir / "shards_gen", gen_shard,
                       desc="open-ended generate")
        io.run_sharded(pick_jobs, m.shard_size, sdir / "shards_pick", pick_shard, desc="open-ended pick")
        gens = io.load_shards(sdir / "shards_gen")
        variants = io.load_shards(sdir / "shards_pick")
        if len(gens) != len(gen_jobs) * o.n_samples or len(variants) != len(pick_jobs):
            raise RuntimeError(f"open-ended shards hold {len(gens)} generations / {len(variants)} pick rows, expected "
                               f"{len(gen_jobs) * o.n_samples} / {len(pick_jobs)}; rerun with --force")
        picks = aggregate_picks(variants) if len(variants) else pd.DataFrame()
        shares = compute_shares(gens, picks, nn_rows, model_id, split)
        if len(gens):
            io.write_parquet(sdir / "generations.parquet", gens)
        if len(variants):
            io.write_parquet(sdir / "pick_variants.parquet", variants)
            io.write_parquet(sdir / "pick_scores.parquet", picks)
        io.write_parquet(sdir / "shares.parquet", shares)
        dist = answer_distribution(gens)
        io.write_parquet(sdir / "answer_distribution.parquet", dist)
        write_summary(sdir, cfg, model_id, split, lm.is_chat, shares, dist, variants, time.time() - t0)
        io.mark_done(sdir, {"model_id": model_id, "split": split, "is_chat": lm.is_chat,
                            "n_generations": int(len(gens)), "n_pick_variants": int(len(variants)),
                            "categories": categories, "elapsed_s": round(time.time() - t0, 1)})
        log.info("open-ended finished: %s", sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir


# --------------------------------------------------------------------------- aggregation
def aggregate_picks(variants: pd.DataFrame) -> pd.DataFrame:
    """Per (category, list, order): logsumexp over variants -> brand score -> normalised pick probability."""
    keys = ["model_id", "split", "category", "list_id", "order"]
    brand = (variants.groupby(keys + ["position", "brand", "list"], sort=True)["logprob"]
             .agg(lambda s: logsumexp(s.values)).rename("brand_logprob").reset_index())
    parts = []
    for _, g in brand.groupby(keys, sort=True):
        g = g.sort_values("position").copy()
        g["prob"] = pick_prob_from_scores(g["brand_logprob"].values)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def compute_shares(gens: pd.DataFrame, picks: pd.DataFrame, nn_rows: dict[str, dict[str, str]],
                   model_id: str, split: str) -> pd.DataFrame:
    """Share of the Appendix A Named and Picked brand per (category, prompt type)."""
    rows = []
    for cat, row in sorted(nn_rows.items()):
        targets = {"named": row["named"], "picked": row["picked"]}
        if len(gens):
            g = gens[gens["category"] == cat]
            for t, gt in g.groupby("prompt_type", sort=True):
                for role, brand in targets.items():
                    hit = gt["matched_entity"].map(lambda e: e is not None and normalize_name(e) == normalize_name(brand))
                    rows.append({"model_id": model_id, "split": split, "category": cat, "prompt_type": t,
                                 "brand_role": role, "brand": brand, "share": float(hit.mean()), "n": int(len(gt))})
        if len(picks):
            p = picks[picks["category"] == cat]
            if len(p):
                per_list = p.groupby(["list_id", "brand"])["prob"].mean().reset_index()  # average the two orders
                for role, brand in targets.items():
                    sel = per_list[per_list["brand"].map(normalize_name) == normalize_name(brand)]
                    rows.append({"model_id": model_id, "split": split, "category": cat, "prompt_type": "pick",
                                 "brand_role": role, "brand": brand, "share": float(sel["prob"].mean()),
                                 "n": int(sel["list_id"].nunique())})
    return pd.DataFrame(rows, columns=["model_id", "split", "category", "prompt_type", "brand_role", "brand", "share",
                                       "n"])


def answer_distribution(gens: pd.DataFrame) -> pd.DataFrame:
    if not len(gens):
        return pd.DataFrame(columns=["category", "prompt_type", "answer", "count", "share"])
    c = gens.groupby(["category", "prompt_type", "answer"]).size().rename("count").reset_index()
    c["share"] = c["count"] / c.groupby(["category", "prompt_type"])["count"].transform("sum")
    return c.sort_values(["category", "prompt_type", "count"], ascending=[True, True, False]).reset_index(drop=True)


def shares_wide(shares: pd.DataFrame) -> pd.DataFrame:
    if not len(shares):
        return pd.DataFrame()
    w = shares.pivot_table(index="category", columns=["prompt_type", "brand_role"], values="share", aggfunc="first")
    w.columns = [f"{t}:{r}" for t, r in w.columns]
    return w.reset_index()


def write_summary(sdir, cfg: Config, model_id: str, split: str, is_chat: bool, shares: pd.DataFrame,
                  dist: pd.DataFrame, variants: pd.DataFrame, elapsed: float) -> None:
    from rtp.measure.name_effect import _md_table

    lines = [f"# Open-ended summary: {model_id} ({split})", "",
             f"- mode: {'chat template' if is_chat else 'base model, text continuation'}; samples per prompt: "
             f"{cfg.open_ended.n_samples} at temperature {cfg.open_ended.temperature}; elapsed {elapsed:.0f}s",
             "- `name/choose/recommend/completion` = fraction of sampled answers naming the brand; "
             "`pick` = probability of the brand after 'I'd pick' among three listed brands, averaged over two list "
             "orders and over lists [Named, third, Picked] for every third brand.", "",
             "## Appendix A brands: Named and Picked shares", ""]
    w = shares_wide(shares)
    if len(w):
        for pos, role in ((1, "named"), (2, "picked")):
            brand = shares[shares["brand_role"] == role].drop_duplicates("category").set_index("category")["brand"]
            w.insert(pos, f"{role}_brand", [brand.get(c, "") for c in w["category"]])
        lines += _md_table(w)
    else:
        lines.append("(no categories with Appendix A rows in this split)")
    if len(variants):
        fb = int(variants["tokenizer_fallback"].sum())
        lines += ["", f"- pick continuations scored: {len(variants)}; tokenizer fallbacks: {fb}"]
    lines += ["", "## Most frequent answers", ""]
    if len(dist):
        top = dist.groupby(["category", "prompt_type"]).head(3)
        lines += _md_table(top[["category", "prompt_type", "answer", "count", "share"]])
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")
