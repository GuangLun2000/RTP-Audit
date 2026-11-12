"""M3 readiness: WHERE is the name effect carried? Full activation transplants (spec §7.6 "full" type only;
the K / random / complement transplants are M4). Hard constraint 1: one model path, never a base model.

Unit = one evidence-0 dev pair at one rotation: the real version (target slot shows the real brand) and
the placebo version (same cards, a placebo name). Both prompts are identical except for the target name, so
every token AFTER the name is aligned between them. For each unit, layer and site the stage runs

* knock-in (spec): the placebo run with the REAL run's residual stream written in at the site;
  recovery = (s_patched - s_placebo) / (s_real - s_placebo);
* knock-out: the real run with the PLACEBO run's residual written in;
  recovery = (s_real - s_patched) / (s_real - s_placebo).

Sites (hidden-state index l = residual after block l-1; layer 0 is never scanned): ``name_last`` (last
token of the target name: the spec's transplant site), ``post_name`` (every token after the name up to the
free answer position), ``final`` (the free answer position only), and ``band`` (``name_last`` at
``localize.band`` consecutive layers at once). Scores are the measure stage's O5 ``s_target`` (answer paths
at the free and forced positions; patches apply to the prompt, the readout continuation is never patched).

Aggregates per (direction, site, layer): pooled recovery = sum(ne * delta) / sum(ne^2) over units (a
least-squares slope, sign-invariant, weighted by |name effect|) with a bootstrap CI over entities; the
between-entity R^2 of entity-mean deltas against entity-mean name effects (what M4's ranking needs); and
pooled recovery per token-count class of the real name (1, 2, 3+ tokens).

Layer choice (reported, not used here): the spec rule is the scanned layer with the highest pooled knock-in
recovery at ``name_last`` (warning below 0.8). A pre-registered alternative (DECISIONS.md) is the deepest
layer whose pooled recovery is >= 0.9 and >= 0.8 in every token class with >= 3 entities.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_choice_items
from rtp.log import add_file_log
from rtp.paths import stage_dir
from rtp.rng import rng_for

log = logging.getLogger(__name__)

SITES = ("name_last", "post_name", "final")
SPEC_RECOVERY_WARN = 0.8
ALT_POOLED_MIN, ALT_CLASS_MIN, ALT_CLASS_MIN_ENTITIES = 0.9, 0.8, 3
VERSIONS = ("real", "placebo")


# --------------------------------------------------------------------------- units
def select_units(items: list[dict[str, Any]], cfg: Config) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(real item, placebo item) per selected (pair, rotation): evidence 0, ``pairs_per_entity`` pairs per entity
    spread over the placebo substitutes (one pair per substitute in turn), at ``localize.rotations``."""
    lz = cfg.localize
    by_pair: dict[str, dict[tuple[bool, int], dict[str, Any]]] = {}
    for it in items:
        if it["evidence_level"] == 0 and it["rotation"] in lz.rotations:
            by_pair.setdefault(it["pair_id"], {})[(it["target_is_placebo"], it["rotation"])] = it
    by_entity: dict[tuple[str, str], dict[str, list[str]]] = {}
    for pid, versions in by_pair.items():
        it = next(iter(versions.values()))
        by_entity.setdefault((it["category"], it["pair_entity"]), {}).setdefault(it["placebo_substitute"], []).append(pid)
    units = []
    for (cat, ent), subs in sorted(by_entity.items()):
        rng = rng_for(cfg.seed, "localize", cat, ent)
        order = sorted(subs)
        pools = {s: sorted(subs[s]) for s in order}
        chosen: list[str] = []
        while len(chosen) < lz.pairs_per_entity and any(pools.values()):
            for s in order:
                if pools[s] and len(chosen) < lz.pairs_per_entity:
                    chosen.append(pools[s].pop(int(rng.integers(len(pools[s])))))
        for pid in sorted(chosen):
            for rot in lz.rotations:
                v = by_pair[pid]
                if (False, rot) in v and (True, rot) in v:
                    units.append((v[(False, rot)], v[(True, rot)]))
    return units


def prepare_unit(lm, real: dict[str, Any], plac: dict[str, Any]) -> dict[str, Any] | None:
    """Token ids (free and forced), the name's last token and the aligned suffix length of both versions;
    None if the text after the name does not tokenise identically (the unit is then skipped)."""
    from rtp.detect.activations import locate, target_name_span
    from rtp.measure.choice import ANSWER_PREFILL, render_inputs

    out: dict[str, Any] = {}
    for v, it in zip(VERSIONS, (real, plac)):
        span = target_name_span(it)
        forced, nl, n_tok = locate(lm, it["prompt"], span, prefill=ANSWER_PREFILL)
        if forced != render_inputs(lm, [it["prompt"]], prefill=True)[0]:
            raise RuntimeError(f"{it['item_id']}: located prompt differs from the measured prompt")
        if lm.is_chat:
            free, nl_free, _ = locate(lm, it["prompt"], span, prefill=None)
            if nl_free != nl or forced[: len(free)] != free:
                return None
        else:
            free = forced
        out[v] = {"item": it, "free": free, "forced": forced, "name_last": nl, "n_name_tokens": n_tok}
    r, p = out["real"], out["placebo"]
    if r["free"][r["name_last"] + 1:] != p["free"][p["name_last"] + 1:]:
        return None
    out["n_post"] = len(r["free"]) - r["name_last"] - 1
    out["unit_id"] = f"{real['pair_id']}-r{real['rotation']}"
    return out


def configurations(n_blocks: int, stride: int, band: int) -> list[tuple[str, int, tuple[int, ...]]]:
    """(site, first layer, layers) of every transplant: three sites at every scanned layer, plus bands."""
    scan = list(range(stride, n_blocks, stride))
    cfgs = [(site, L, (L,)) for L in scan for site in SITES]
    cfgs += [("band", L, tuple(range(L, L + band))) for L in scan if L + band - 1 <= n_blocks - 1]
    return cfgs


# --------------------------------------------------------------------------- scoring
@torch.inference_mode()
def donor_states(lm, units: list[dict[str, Any]], layers: list[int]) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Residual states of each unit's two versions at the name's last token and every post-name position."""
    from rtp.infer import pad_left

    seqs = [(u, v) for u in range(len(units)) for v in VERSIONS]
    ids, mask, pos = pad_left([units[u][v]["free"] for u, v in seqs], lm.tokenizer.pad_token_id, lm.device)
    out = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True, logits_to_keep=1)
    hs = torch.stack([out.hidden_states[lv] for lv in layers], dim=1)  # [B, n_layers, T, d]
    res = {}
    for row, (u, v) in enumerate(seqs):
        un = units[u]
        off = ids.shape[1] - len(un[v]["free"])
        nl = off + un[v]["name_last"]
        res[(u, v)] = {"name_last": hs[row, :, nl, :].clone(), "post": hs[row, :, nl + 1: nl + 1 + un["n_post"], :].clone()}
    return res


def score_rows(lm, units: list[dict[str, Any]], rows: list[dict[str, Any]], letters, layers: list[int], cfg: Config,
               ) -> list[dict[str, float]]:
    """O5 scores of ``rows`` = {unit, receiver, donor (None = clean), site, layers}."""
    from rtp.measure.answers import answer_distributions, estimators
    from rtp.measure.choice import LETTERS, item_answer_paths

    lz = cfg.localize
    donors = donor_states(lm, units, layers) if any(r["donor"] for r in rows) else {}
    li = {lv: k for k, lv in enumerate(layers)}
    paths_of = {}
    for u, un in enumerate(units):
        for v in VERSIONS:
            paths_of[(u, v)] = item_answer_paths(lm.tokenizer, [un[v]["item"]], letters)[0]

    def patches_for(batch: list[int], padded_len: int, offsets: list[int]):
        patches: dict[int, list] = {}
        for r, gi in enumerate(batch):
            row = rows[gi]
            if row["donor"] is None:
                continue
            un, recv = units[row["unit"]], row["receiver"]
            d = donors[(row["unit"], row["donor"])]
            nl = un[recv]["name_last"]
            for lv in row["layers"]:
                k = li[lv]
                if row["site"] in ("name_last", "band"):
                    sites = [(nl, d["name_last"][k])]
                elif row["site"] == "post_name":
                    sites = [(nl + 1 + j, d["post"][k, j]) for j in range(un["n_post"])]
                else:  # final: the free answer position
                    sites = [(nl + un["n_post"], d["post"][k, un["n_post"] - 1])]
                patches.setdefault(lv, []).extend((r, offsets[r] + p, vec) for p, vec in sites)
        return patches

    free = [units[r["unit"]][r["receiver"]]["free"] for r in rows]
    forced = [units[r["unit"]][r["receiver"]]["forced"] for r in rows]
    paths = [paths_of[(r["unit"], r["receiver"])] for r in rows]
    res = answer_distributions(lm, free, forced, paths, lz.max_batch_size, lz.max_batch_tokens, patches_for)
    out = []
    for r, x in zip(rows, res):
        it = units[r["unit"]][r["receiver"]]["item"]
        e = estimators(x["free"], x["forced"], LETTERS.index(it["target_label"]))
        out.append({k: e[k] for k in ("s_target", "s_target_free", "s_target_forced", "c_free", "c_forced")})
    return out


def unit_rows(n_units: int, cfgs) -> list[dict[str, Any]]:
    rows = []
    for u in range(n_units):
        rows += [{"unit": u, "receiver": v, "donor": None, "direction": "clean", "site": "none", "layer": -1,
                  "layers": ()} for v in VERSIONS]
        for site, L, lays in cfgs:
            rows.append({"unit": u, "receiver": "placebo", "donor": "real", "direction": "in", "site": site,
                         "layer": L, "layers": lays})
            rows.append({"unit": u, "receiver": "real", "donor": "placebo", "direction": "out", "site": site,
                         "layer": L, "layers": lays})
    return rows


# --------------------------------------------------------------------------- analysis
def token_class(n: int) -> str:
    return "1" if n <= 1 else ("2" if n == 2 else "3+")


def unit_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per unit and transplant: name effect ne = s_real - s_placebo (clean runs) and the recovered part delta."""
    clean = df[df["direction"] == "clean"].pivot_table(index="unit_id", columns="receiver", values="s_target")
    clean = clean.rename(columns={"real": "s_real_clean", "placebo": "s_placebo_clean"})
    t = df[df["direction"] != "clean"].join(clean[["s_real_clean", "s_placebo_clean"]], on="unit_id")
    t["delta"] = np.where(t["direction"] == "in", t["s_target"] - t["s_placebo_clean"],
                          t["s_real_clean"] - t["s_target"])
    t["ne"] = t["s_real_clean"] - t["s_placebo_clean"]
    return t


def _pooled(ne: np.ndarray, delta: np.ndarray) -> float:
    den = float((ne ** 2).sum())
    return float((ne * delta).sum() / den) if den > 0 else float("nan")


def recovery_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Pooled recovery per (direction, site, layer) with an entity bootstrap CI, between-entity R^2 and per
    token class. The bootstrap resamples entities with replacement (the same draws for every row)."""
    t = unit_deltas(df)
    ents = sorted(t["entity"].unique())
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ents), np.full(len(ents), 1 / len(ents)), size=n_boot) if ents else None
    rows = []
    for (direction, site, layer), g in t.groupby(["direction", "site", "layer"], sort=True):
        g = g.assign(nd=g["ne"] * g["delta"], n2=g["ne"] ** 2)
        per = g.groupby("entity")[["nd", "n2"]].sum().reindex(ents, fill_value=0.0)
        den = counts @ per["n2"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = (counts @ per["nd"].to_numpy()) / den
        boots = boots[np.isfinite(boots) & (den > 0)]
        em = g.groupby("entity")[["ne", "delta"]].mean()
        ss = float(((em["ne"] - em["ne"].mean()) ** 2).sum())
        r2 = 1 - float(((em["ne"] - em["delta"]) ** 2).sum()) / ss if ss > 0 and len(em) >= 3 else float("nan")
        row = {"direction": direction, "site": site, "layer": int(layer), "layers": str(g["layers"].iloc[0]),
               "pooled_recovery": _pooled(g["ne"].to_numpy(), g["delta"].to_numpy()),
               "ci_low": float(np.quantile(boots, 0.025)) if len(boots) > 10 else float("nan"),
               "ci_high": float(np.quantile(boots, 0.975)) if len(boots) > 10 else float("nan"),
               "r2_between_entities": r2, "n_units": int(len(g)), "n_entities": int(g["entity"].nunique())}
        for cls, gc in g.groupby("token_class"):
            row[f"pooled_tok{cls}"] = _pooled(gc["ne"].to_numpy(), gc["delta"].to_numpy())
            row[f"n_entities_tok{cls}"] = int(gc["entity"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_layers(rec: pd.DataFrame) -> dict[str, Any]:
    """Spec rule (argmax of pooled knock-in recovery at name_last) and the pre-registered alternative."""
    nl = rec[(rec["direction"] == "in") & (rec["site"] == "name_last")].sort_values("layer")
    if nl.empty or not np.isfinite(nl["pooled_recovery"]).any():
        return {"spec_layer": None, "spec_recovery": None, "spec_warning": True, "alt_layer": None}
    best = nl.loc[nl["pooled_recovery"].idxmax()]
    alt = None
    classes = [c[len("pooled_tok"):] for c in nl.columns if c.startswith("pooled_tok")]
    for _, r in nl.iloc[::-1].iterrows():
        ok_cls = all(not (r.get(f"n_entities_tok{c}", 0) >= ALT_CLASS_MIN_ENTITIES) or
                     r.get(f"pooled_tok{c}", np.nan) >= ALT_CLASS_MIN for c in classes)
        if r["pooled_recovery"] >= ALT_POOLED_MIN and ok_cls:
            alt = int(r["layer"])
            break
    return {"spec_layer": int(best["layer"]), "spec_recovery": float(best["pooled_recovery"]),
            "spec_r2_between": float(best["r2_between_entities"]),
            "spec_warning": bool(best["pooled_recovery"] < SPEC_RECOVERY_WARN), "alt_layer": alt}


# --------------------------------------------------------------------------- stage
def run_localize(cfg: Config, model_id: str, force: bool = False, prompt_mode: str = "auto") -> Path:
    """Localisation scan on dev evidence-0 pairs of ``model_id`` (dev only; method development)."""
    from rtp.measure.choice import letter_token_ids
    from rtp.models import apply_prompt_mode, get_blocks, load_model
    from rtp.progress import silenced

    split = "dev"
    lz = cfg.localize
    items = load_choice_items(cfg, split)
    sdir = stage_dir(cfg.name, model_id, "localize")
    shash = stages.stage_hash(cfg, "localize", model_id, split, prompt_mode)
    if io.finished_stage(sdir, shash, force):
        log.info("localize already finished for %s with this config and code (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta=stages.stage_meta("localize", model_id, split, prompt_mode))
    stages.check_resume_environment(sdir, force)
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        pairs = select_units(items, cfg)
        if not pairs:
            raise ValueError("localize: no evidence-0 dev pairs at the configured rotations")
        state: dict[str, Any] = {}

        def process(chunk, idx):
            if "lm" not in state:
                state["lm"] = apply_prompt_mode(load_model(model_id), prompt_mode)
                stages.record_model_revision(sdir, state["lm"].model)
                state["letters"] = letter_token_ids(state["lm"].tokenizer)
                n_blocks = len(get_blocks(state["lm"].model))
                state["cfgs"] = configurations(n_blocks, lz.layer_stride, lz.band)
                state["layers"] = sorted({lv for _, _, lays in state["cfgs"] for lv in lays})
                log.info("localize: %d blocks, %d transplant configurations per direction and unit, layers %s",
                         n_blocks, len(state["cfgs"]), state["layers"])
            lm = state["lm"]
            out = []
            prepared = [(r, p, prepare_unit(lm, r, p)) for r, p in chunk]
            skipped = [r["pair_id"] for r, _, u in prepared if u is None]
            if skipped:
                log.warning("localize: %d units skipped (text after the name tokenises differently): %s",
                            len(skipped), skipped[:5])
            units = [u for _, _, u in prepared if u is not None]
            for s in range(0, len(units), lz.units_per_batch):
                part = units[s:s + lz.units_per_batch]
                rows = unit_rows(len(part), state["cfgs"])
                with silenced():
                    scores = score_rows(lm, part, rows, state["letters"], state["layers"], cfg)
                for row, sc in zip(rows, scores):
                    un = part[row["unit"]]
                    out.append({"unit_id": un["unit_id"], "pair_id": un["real"]["item"]["pair_id"],
                                "rotation": un["real"]["item"]["rotation"], "category": un["real"]["item"]["category"],
                                "entity": un["real"]["item"]["pair_entity"],
                                "placebo_name": un["placebo"]["item"]["target_entity"],
                                "n_name_tokens": un["real"]["n_name_tokens"],
                                "token_class": token_class(un["real"]["n_name_tokens"]),
                                "receiver": row["receiver"], "direction": row["direction"], "site": row["site"],
                                "layer": row["layer"], "layers": ",".join(map(str, row["layers"])), **sc})
            return pd.DataFrame(out)

        io.run_sharded(pairs, lz.shard_size, sdir / "shards", process, desc="localize")
        df = io.load_shards(sdir / "shards")
        io.write_parquet(sdir / "transplants.parquet", df)
        rec = recovery_table(df, lz.n_boot, cfg.seed)
        io.write_parquet(sdir / "recovery.parquet", rec)
        choice = choose_layers(rec)
        n_units = int(df["unit_id"].nunique())
        write_localize_summary(sdir, model_id, cfg, rec, choice, n_units, len(pairs), time.time() - t0)
        from rtp.provenance import environment

        warnings = []
        if choice["spec_warning"]:
            warnings.append(f"localize: best full knock-in recovery at the name's last token is "
                            f"{choice['spec_recovery']} (< {SPEC_RECOVERY_WARN}) at layer {choice['spec_layer']}")
        io.mark_done(sdir, {"model_id": model_id, "n_units": n_units, "n_pairs_selected": len(pairs), **choice,
                            "warnings": warnings, "finished_with": environment(),
                            "elapsed_s": round(time.time() - t0, 1)})
        log.info("localize finished: %s (spec layer %s, recovery %s)", sdir, choice["spec_layer"],
                 choice["spec_recovery"])
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir


def write_localize_summary(sdir: Path, model_id: str, cfg: Config, rec: pd.DataFrame, choice: dict[str, Any],
                           n_units: int, n_selected: int, elapsed: float) -> None:
    from rtp.measure.name_effect import _md_table

    lz = cfg.localize
    lines = [f"# Localisation summary: {model_id} (dev, evidence 0)", "",
             f"- units (pair x rotation): {n_units} of {n_selected} selected ({lz.pairs_per_entity} pairs per entity, "
             f"rotations {lz.rotations}); layer stride {lz.layer_stride}; band {lz.band} layers; elapsed "
             f"{elapsed:.0f}s",
             "- knock-in = placebo run with the real run's residual written in at the site (spec); knock-out = the "
             "reverse. pooled_recovery = sum(ne*delta)/sum(ne^2) over units (1 = the whole name effect moves with "
             "the site); r2_between_entities = how much of the between-entity variation of name effects the "
             "transplant reproduces (what M4 ranks on).", "",
             "## Layer choice", "",
             f"- spec rule (highest pooled knock-in recovery at name_last): layer {choice['spec_layer']}, recovery "
             f"{_fmt(choice['spec_recovery'])}, between-entity R^2 {_fmt(choice.get('spec_r2_between'))}",
             f"- pre-registered alternative (deepest layer with pooled >= {ALT_POOLED_MIN} and >= {ALT_CLASS_MIN} in "
             f"every token class with >= {ALT_CLASS_MIN_ENTITIES} entities): {choice['alt_layer']}"]
    if choice["spec_warning"]:
        lines.append(f"- **WARNING: the best knock-in recovery at the name's last token is below {SPEC_RECOVERY_WARN}; "
                     f"a transplant at that site cannot carry the full name effect (spec §7.6). Look at post_name / "
                     f"band below to see where the effect lives.**")
    cols = ["direction", "site", "layer", "pooled_recovery", "ci_low", "ci_high", "r2_between_entities"] + \
        [c for c in rec.columns if c.startswith("pooled_tok")] + ["n_units", "n_entities"]
    for direction, title in (("in", "Knock-in (placebo run <- real states)"),
                             ("out", "Knock-out (real run <- placebo states)")):
        sub = rec[rec["direction"] == direction].sort_values(["site", "layer"])
        lines += ["", f"## {title}", ""] + _md_table(sub[[c for c in cols if c in sub]])
    io.atomic_write_text(sdir / "summary.md", "\n".join(lines) + "\n")


def _fmt(v: Any) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
