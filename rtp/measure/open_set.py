"""The ``open-set`` stage: which brand does the model name when asked an open question?

For every category of the split and every prompt type ("Name a laptop brand. Reply with the name only."):

* CLOSED-SET DISTRIBUTION. The probability that the reply starts with each catalogue brand (its name or an
  alias, in the usual surface forms), computed exactly on an answer-path trie from the model's next-token
  distributions (rtp.measure.answers). ``coverage`` is the total mass on the category's brands; the rest is
  everything else the model might say (another brand, a preamble, a refusal). This is a distribution over
  brands, not a sample: its concentration has no sampling bias, and a counterfactual such as "the previous
  stage, sharpened by c" can be computed from it brand by brand.
* SAMPLED ANSWERS at each temperature of ``open_set.temperatures`` (``n_samples`` each), matched to brands as
  in the legacy open-ended stage. The grid is fixed in the config, so no GPU job depends on a scale
  multiplier estimated elsewhere; it shows what decoding at another temperature does to concentration.
  Token-level temperature is not the brand-level power transform (multi-token names, shared first tokens,
  preambles), which is why both are measured.

Base (non-chat) checkpoints get the plain-text completion prompt instead of the chat prompts.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_catalogue
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.measure.answers import answer_distributions, build_name_paths
from rtp.measure.parse import BrandMatcher
from rtp.models import LoadedModel, apply_prompt_mode, chat_prompt, encode
from rtp.paths import stage_dir
from rtp.rng import stable_int

log = logging.getLogger(__name__)

TEMPLATES = {
    "name": "Name {a} {brand}. Reply with the name only.",
    "choose": "Choose {a} {brand}. Reply with the name only.",
    "recommend": "Can you recommend {a} {brand}? Reply with the name only.",
}
COMPLETION_TYPE = "completion"
COMPLETION_TEMPLATE = "If you want a good {brand}, I recommend"


# The prompt wording is built HERE, inside the stage's own code: generator modules (rtp.data.*) are not part
# of a model stage's identity, so a prompt helper imported from there could change without invalidating
# finished open-set results.
def brand_phrase(category: str) -> str:
    """'running shoe' -> 'running shoe brand'; 'coffee brand' stays 'coffee brand'."""
    return category if category.endswith(" brand") else f"{category} brand"


def indefinite_article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def open_set_stage_name(split: str) -> str:
    return "open_set" if split == "dev" else "open_set_test"


def n_eff(p: np.ndarray) -> float:
    """Effective number of options, exp(entropy), of a distribution (normalised here; zeros allowed)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if not len(p):
        return float("nan")
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def jobs_for(lm: LoadedModel, cfg: Config, catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    types = list(cfg.open_set.prompt_types) if lm.is_chat else [COMPLETION_TYPE]
    jobs = []
    for cat in dict.fromkeys(r["category"] for r in catalogue):
        brands = [r for r in catalogue if r["category"] == cat and not r["is_placebo"]]
        phrase = brand_phrase(cat)
        for t in types:
            if t == COMPLETION_TYPE:
                text = COMPLETION_TEMPLATE.format(brand=phrase)
                rendered = text
            else:
                text = TEMPLATES[t].format(a=indefinite_article(phrase), brand=phrase)
                rendered = chat_prompt(lm, text)
            jobs.append({"category": cat, "prompt_type": t, "user_text": text,
                         "ids": encode(lm.tokenizer, rendered, lm.is_chat), "brands": brands})
    return jobs


def closed_set(lm: LoadedModel, jobs: list[dict[str, Any]], cfg: Config) -> pd.DataFrame:
    o = cfg.open_set
    cache: dict[str, tuple[int, ...]] = {}
    paths = [build_name_paths(lm.tokenizer, [[b["entity"], *b["aliases"]] for b in j["brands"]], cache) for j in jobs]
    ids = [j["ids"] for j in jobs]
    # one position: the start of the reply (free == forced when the two prompts are the same)
    res = answer_distributions(lm, ids, ids, paths, o.max_batch_size, o.max_batch_tokens, eps=o.eps)
    rows = []
    for j, r in zip(jobs, res):
        q = r["free"].q
        cov = float(q.sum())
        for b, mass in zip(j["brands"], q):
            rows.append({"model_id": lm.model_id, "category": j["category"], "prompt_type": j["prompt_type"],
                         "entity": b["entity"], "p": float(mass), "coverage": cov, "pruned": float(r["free"].pruned)})
    return pd.DataFrame(rows)


def sampled(lm: LoadedModel, jobs: list[dict[str, Any]], cfg: Config) -> pd.DataFrame:
    from rtp.infer import generate

    o = cfg.open_set
    rows = []
    for j in jobs:
        matcher = BrandMatcher([(b["entity"], list(b["aliases"]), bool(b.get("match_case"))) for b in j["brands"]])
        for temp in o.temperatures:
            seed = stable_int(cfg.seed, "open_set", j["category"], j["prompt_type"], repr(float(temp))) % (2**31 - 1)
            texts = generate(lm, [j["ids"]], max_new_tokens=o.max_new_tokens, do_sample=True, temperature=float(temp),
                             num_return_sequences=o.n_samples, seed=seed, max_batch_size=o.n_samples,
                             max_batch_tokens=o.max_batch_tokens)[0]
            for k, t in enumerate(texts):
                rows.append({"model_id": lm.model_id, "category": j["category"], "prompt_type": j["prompt_type"],
                             "temperature": float(temp), "sample_idx": k, "text": t, "matched_entity": matcher.match(t)})
    return pd.DataFrame(rows, columns=["model_id", "category", "prompt_type", "temperature", "sample_idx", "text",
                                       "matched_entity"])


def summarize(closed: pd.DataFrame, samples: pd.DataFrame) -> dict[str, Any]:
    by = closed.groupby(["category", "prompt_type"], sort=False)
    per = by.agg(coverage=("coverage", "first"), top1=("p", lambda x: float(x.max() / x.sum()) if x.sum() > 0 else np.nan),
                 n_eff=("p", lambda x: n_eff(x.to_numpy()))).reset_index()
    info: dict[str, Any] = {"n_prompts": int(len(per)), "coverage": round(float(per["coverage"].mean()), 4),
                            "n_eff": {t: round(float(g["n_eff"].mean()), 3) for t, g in per.groupby("prompt_type")},
                            "top1": {t: round(float(g["top1"].mean()), 3) for t, g in per.groupby("prompt_type")}}
    if len(samples):
        matched = samples[samples["matched_entity"].notna()]
        info["sample_match_rate"] = round(float(len(matched) / len(samples)), 4)
        ne: dict[str, float] = {}
        for temp, g in matched.groupby("temperature"):
            vals = [n_eff(x["matched_entity"].value_counts().to_numpy()) for _, x in g.groupby(["category", "prompt_type"])]
            ne[f"{temp:g}"] = round(float(np.mean(vals)), 3)
        info["sampled_n_eff_by_temperature"] = ne  # plug-in estimate from n_samples draws: biased low
    return info


def write_summary(sdir, info: dict[str, Any], model_id: str, split: str, per: pd.DataFrame) -> None:
    lines = [f"# Open-set brand distribution: {model_id} ({split})", "",
             f"- prompts: {info['n_prompts']}; mean mass on the category's catalogue brands (coverage): "
             f"{info['coverage']}",
             f"- closed-set N_eff (exp entropy over the catalogue brands, mean over categories): {info['n_eff']}",
             f"- closed-set top-1 share: {info['top1']}"]
    if "sampled_n_eff_by_temperature" in info:
        lines += [f"- sampled answers matched to a catalogue brand: {info['sample_match_rate']:.1%}",
                  f"- sampled N_eff by temperature (plug-in, biased low at small samples): "
                  f"{info['sampled_n_eff_by_temperature']}"]
    lines += ["", "## Per category and prompt type", "", "| category | prompt | coverage | N_eff | top brand | its share |",
              "|---|---|---|---|---|---|"]
    for r in per.itertuples(index=False):
        lines.append(f"| {r.category} | {r.prompt_type} | {r.coverage:.3f} | {r.n_eff:.2f} | {r.top_entity} | "
                     f"{r.top_share:.3f} |")
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")


def run_open_set(cfg: Config, model_id: str, split: str = "dev", unseal: bool = False, force: bool = False,
                 prompt_mode: str = "auto"):
    from rtp.models import load_model

    check_split_access(split, unseal)
    catalogue = load_catalogue(cfg, split, unseal)
    sdir = stage_dir(cfg.name, model_id, open_set_stage_name(split))
    shash = stages.stage_hash(cfg, "open_set", model_id, split, prompt_mode)
    if io.finished_stage(sdir, shash, force):
        log.info("open-set already finished for %s with this config and code (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta=stages.stage_meta("open_set", model_id, split, prompt_mode))
    stages.check_resume_environment(sdir, force)
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        if not any(not r["is_placebo"] for r in catalogue):
            raise ValueError(f"no real brands in split '{split}'")
        tl = time.time()
        lm = apply_prompt_mode(load_model(model_id), prompt_mode)
        load_s = time.time() - tl
        stages.record_model_revision(sdir, lm.model)
        jobs = jobs_for(lm, cfg, catalogue)
        closed = closed_set(lm, jobs, cfg)
        closed.insert(1, "split", split)
        samples = sampled(lm, jobs, cfg) if cfg.open_set.n_samples > 0 else sampled(lm, [], cfg)
        io.write_parquet(sdir / "closed_set.parquet", closed)
        io.write_parquet(sdir / "samples.parquet", samples)
        info = summarize(closed, samples)
        top = closed.sort_values("p", ascending=False).groupby(["category", "prompt_type"], sort=False).head(1)
        per = closed.groupby(["category", "prompt_type"], sort=False).agg(
            coverage=("coverage", "first"), n_eff=("p", lambda x: n_eff(x.to_numpy())), total=("p", "sum")).reset_index()
        per = per.merge(top[["category", "prompt_type", "entity", "p"]].rename(
            columns={"entity": "top_entity", "p": "top_p"}), on=["category", "prompt_type"])
        per["top_share"] = per["top_p"] / per["total"].where(per["total"] > 0)
        info.update({"model_id": model_id, "split": split, "elapsed_s": round(time.time() - t0, 1),
                     "timing": {"load_s": round(load_s, 1)}})
        info["coverage_warning"] = bool(info["coverage"] < 0.5)
        write_summary(sdir, info, model_id, split, per)
        from rtp.provenance import environment

        io.mark_done(sdir, {**info, "finished_with": environment()})
        log.info("open-set finished: %s", sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir
