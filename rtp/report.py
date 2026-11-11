"""``report``: one Markdown report across every model measured under a config.

Reads finished stage dirs under ``$RTP_WORK_DIR/runs/{config}/``. Test-split stages are
included only with ``--unseal``. Each model is summarised on its own (calibration is
within-model); the cross-model tables only line results up side by side.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rtp import io, registry
from rtp.config import Config
from rtp.data.splits import check_split_access
from rtp.measure.choice import measure_stage_name
from rtp.measure.name_effect import _md_table
from rtp.measure.open_ended import open_ended_stage_name, shares_wide
from rtp.paths import run_dir

log = logging.getLogger(__name__)

STAGE_ORDER = {"base": 0, "sft": 1, "dpo": 2, "final": 3, "instruct": 3}


def _model_sort_key(model_id: str) -> tuple:
    spec = registry.get_model_spec(model_id)
    return (spec.family if spec else "~", STAGE_ORDER.get(spec.stage, 9) if spec else 9, model_id)


def _finished(cfg: Config, rdir: Path, stage: str, split: str, stale: list[str]) -> list[tuple[str, Path]]:
    """Finished stage dirs produced by THIS config; others are recorded in ``stale`` and left out."""
    kind = "measure" if stage.startswith("measure") else "open_ended"
    out = []
    for d in sorted(rdir.glob(f"*/{stage}")):
        if not io.is_done(d):
            continue
        meta = io.read_json(d / io.STAGE_META)
        model_id = meta.get("model_id", d.parent.name)
        expected = cfg.stage_hash(kind, model_id, split, meta.get("prompt_mode", "auto"))
        if meta.get("stage_hash") != expected:
            stale.append(f"{model_id} / {stage}")
            continue
        out.append((model_id, d))
    return sorted(out, key=lambda x: _model_sort_key(x[0]))


def build_report(cfg: Config, unseal: bool = False) -> str:
    rdir = run_dir(cfg.name)
    splits = ["dev"] + (["test"] if unseal else [])
    if unseal:
        check_split_access("test", unseal)
    lines = [f"# Report: {cfg.name}", "", f"- data hash `{cfg.data_hash}`; splits: {', '.join(splits)}",
             "- name effects are calibrated within each model; columns from different models are shown side by side "
             "only for reading convenience.", ""]
    stale: list[str] = []
    for split in splits:
        measured = _finished(cfg, rdir, measure_stage_name(split), split, stale)
        opened = _finished(cfg, rdir, open_ended_stage_name(split), split, stale)
        lines += [f"## Split: {split}", ""]
        if not measured and not opened:
            lines += ["(no finished stages)", ""]
            continue
        if measured:
            lines += ["### Behavioural measurement", "", "| model | items | pairs | consistency | warning | letter mass | ladder slope |",
                      "|---|---|---|---|---|---|---|"]
            ne_cols = []
            for model_id, d in measured:
                info = io.read_json(d / io.DONE_FILE)
                c = info.get("consistency", {})
                agree = c.get("agreement")
                agree_s = f"{agree:.1%} of {c.get('n_checked')}" if agree == agree and agree is not None else "—"
                slope = info.get("ladder_slope")
                slope_s = f"{slope:+.3f}" if isinstance(slope, (int, float)) and slope == slope else "—"
                mass = info.get("mean_letter_mass")
                mass_s = "—" if mass is None else (f"**{mass:.3f}**" if info.get("letter_mass_warning") else f"{mass:.3f}")
                lines.append(f"| {model_id} | {info.get('n_items')} | {info.get('n_pairs')} | {agree_s} | "
                             f"{'**yes**' if info.get('consistency_warning') else 'no'} | {mass_s} | {slope_s} |")
                t = pd.read_parquet(d / "entity_evidence.parquet")
                keep = [c for c in t.columns if c.startswith("ne_e")]
                t = t.set_index(["category", "entity", "natural_negative"])[keep]
                t.columns = [f"{model_id.split('/')[-1]}:{c}" for c in t.columns]
                ne_cols.append(t)
            lines += ["", "### Name effect per entity (evidence 0 and 4; per model)", ""]
            wide = pd.concat(ne_cols, axis=1).reset_index()
            show = ["category", "entity", "natural_negative"] + [c for c in wide.columns
                                                                 if c.endswith(":ne_e0") or c.endswith(":ne_e4")]
            lines += _md_table(wide[show]) + [""]
        if opened:
            lines += ["### Open-ended: Appendix A Named / Picked shares", ""]
            for model_id, d in opened:
                shares = pd.read_parquet(d / "shares.parquet")
                lines += [f"**{model_id}**", ""]
                w = shares_wide(shares)
                lines += (_md_table(w) if len(w) else ["(no Appendix A categories)"]) + [""]
            all_sh = pd.concat([pd.read_parquet(d / "shares.parquet") for _, d in opened], ignore_index=True)
            pk = all_sh[(all_sh["prompt_type"] == "pick")]
            if len(pk):
                lines += ["### Pick probability of the Named vs Picked brand across checkpoints", ""]
                order = [m for m, _ in opened]
                piv = pk.pivot_table(index=["category", "brand_role", "brand"], columns="model_id", values="share")
                piv = piv[[m for m in order if m in piv.columns]].reset_index()
                lines += _md_table(piv) + [""]
    if stale:
        lines += ["## Excluded stale stages", "",
                  "These finished stages were produced with a different configuration or different seed files and "
                  "are NOT reported. Re-run them with --force:", ""] + [f"- {s}" for s in stale] + [""]
    return "\n".join(lines) + "\n"


def run_report(cfg: Config, unseal: bool = False) -> Path:
    text = build_report(cfg, unseal=unseal)
    out = run_dir(cfg.name) / "report.md"
    io.atomic_write_text(out, text)
    log.info("report written to %s", out)
    print(text)
    return out
