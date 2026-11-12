"""M3 activation extraction (spec §7.4). Hard constraint 1: one model path, never a base model.

Read with ``output_hidden_states=True`` (hooks are reserved for M4 transplant):

* probe items: the last token of the READ name mention (``name_char_span``);
* choice items (``extract.choice_evidence_levels`` x ``extract.choice_rotations``): the last
  token of the TARGET option's name ("name") and the last input token ("last").

Prompts are rendered exactly as they are measured: chat models through the chat template
(choice items with the "Answer:" prefill), base models as plain text. The name is located by
finding the user text inside the rendered string and mapping its character span to tokens
with ``return_offsets_mapping=True``. Layers are hidden-state indices 0 (embeddings), stride,
2*stride, ... below n_blocks (index n_blocks is the post-final-norm state and is excluded). Arrays are float16, ``[n_items, n_layers, d_model]``,
written in shards with completion markers; the expected size is logged before writing.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtp import io, stages
from rtp.config import Config
from rtp.data.build import load_choice_items, load_probe_items
from rtp.data.splits import check_split_access
from rtp.log import add_file_log
from rtp.measure.choice import ANSWER_PREFILL, COMPLETION_SUFFIX
from rtp.models import LoadedModel, apply_prompt_mode, chat_prompt, get_blocks
from rtp.paths import stage_dir

log = logging.getLogger(__name__)

SETS = {"probe": ("name",), "choice": ("name", "last")}


class TokenLocateError(RuntimeError):
    pass


def extract_stage_name(split: str) -> str:
    return "extract" if split == "dev" else "extract_test"


def selected_layers(n_blocks: int, stride: int) -> list[int]:
    """Hidden-state indices 0, stride, 2*stride, ... < n_blocks (0 = embeddings; index i = residual stream after
    block i-1). Index n_blocks is excluded: transformers returns the FINAL-NORM output there, not the last block's
    residual stream, so it is not a valid M4 transplant site."""
    return list(range(0, n_blocks, stride))


def target_name_span(item: dict[str, Any]) -> list[int]:
    """Character span of the target option's name inside a choice item's user prompt."""
    marker = f"\n{item['target_label']}. {item['target_entity']}"
    pos = item["prompt"].find(marker)
    if pos < 0:
        raise TokenLocateError(f"{item['item_id']}: target option line not found")
    start = pos + len(f"\n{item['target_label']}. ")
    return [start, start + len(item["target_entity"])]


def render(lm: LoadedModel, user_text: str, prefill: str | None) -> tuple[str, int]:
    """(rendered prompt, character offset of ``user_text`` inside it)."""
    if not lm.is_chat:
        return user_text + (COMPLETION_SUFFIX if prefill else ""), 0
    rendered = chat_prompt(lm, user_text, assistant_prefix=prefill)
    off = rendered.find(user_text)
    if off < 0:
        raise TokenLocateError("the chat template altered the user text; cannot locate the name")
    return rendered, off


def locate(lm: LoadedModel, user_text: str, span: list[int], prefill: str | None) -> tuple[list[int], int, int]:
    """Token ids of the rendered prompt, index of the name's last token, and number of name tokens."""
    tok = lm.tokenizer
    if not getattr(tok, "is_fast", False):
        raise TokenLocateError("a fast tokenizer (offset mapping) is required to locate names")
    rendered, off = render(lm, user_text, prefill)
    enc = tok(rendered, add_special_tokens=not lm.is_chat, return_offsets_mapping=True)
    s, e = off + span[0], off + span[1]
    covering = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    if not covering:
        raise TokenLocateError(f"no token covers characters {s}-{e}")
    return list(enc["input_ids"]), max(covering), len(covering)


def estimate_bytes(n_items: int, n_layers: int, d_model: int, n_positions: int) -> int:
    return n_items * n_layers * d_model * n_positions * 2  # float16


@torch.inference_mode()
def hidden_at(lm: LoadedModel, seqs: list[list[int]], positions: list[dict[str, int]], layers: list[int],
              max_batch_size: int, max_batch_tokens: int) -> list[dict[str, np.ndarray]]:
    """Per sequence: {position name -> float16 array [n_layers, d]} from output_hidden_states."""
    from rtp.infer import pad_left, run_batched

    pad_id = lm.tokenizer.pad_token_id

    def fn(batch: list[int]) -> list[dict[str, np.ndarray]]:
        ids, mask, pos = pad_left([seqs[i] for i in batch], pad_id, lm.device)
        out = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                       logits_to_keep=1)
        hs = torch.stack([out.hidden_states[layer] for layer in layers], dim=1)  # [B, L, T, d]
        res = []
        for row, i in enumerate(batch):
            pad = ids.shape[1] - len(seqs[i])
            res.append({name: hs[row, :, pad + p, :].float().cpu().numpy().astype(np.float16)
                        for name, p in positions[i].items()})
        return res

    return run_batched(seqs, fn, max_batch_size, max_batch_tokens, desc="extract")


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def prepare_records(lm: LoadedModel, set_name: str, items: list[dict[str, Any]]):
    """Token ids, positions and an index row for each item of one set."""
    seqs, positions, rows = [], [], []
    for it in items:
        if set_name == "probe":
            ids, last_name, n_name = locate(lm, it["text"], it["name_char_span"], prefill=None)
            positions.append({"name": last_name})
            rows.append({"id": it["probe_id"], "category": it["category"], "entity": it["entity"],
                         "frame": it["frame"], "control": it["control"], "other_entity": it["other_entity"],
                         "template_id": it["template_id"], "other_first": it.get("other_first"),
                         **{f"y_{a}": v for a, v in it["attributes"].items()},
                         **{f"other_{a}": v for a, v in (it["other_attributes"] or {}).items()},
                         "name_tokens": n_name, "n_tokens": len(ids)})
        else:
            ids, last_name, n_name = locate(lm, it["prompt"], target_name_span(it), prefill=ANSWER_PREFILL)
            positions.append({"name": last_name, "last": len(ids) - 1})
            rows.append({"id": it["item_id"], "pair_id": it["pair_id"], "category": it["category"],
                         "pair_entity": it["pair_entity"], "target_entity": it["target_entity"],
                         "target_is_placebo": it["target_is_placebo"], "evidence_level": it["evidence_level"],
                         "rotation": it["rotation"], "name_tokens": n_name, "n_tokens": len(ids)})
        seqs.append(ids)
    return seqs, positions, pd.DataFrame(rows)


def run_extract(cfg: Config, model_id: str, split: str = "dev", unseal: bool = False, force: bool = False,
                prompt_mode: str = "auto") -> Path:
    from rtp.models import load_model

    check_split_access(split, unseal)
    x = cfg.extract
    probes = load_probe_items(cfg, split, unseal)
    choice = [it for it in load_choice_items(cfg, split, unseal)
              if it["evidence_level"] in x.choice_evidence_levels and it["rotation"] in x.choice_rotations]
    if not choice:
        log.warning("extract: no choice items at evidence %s / rotations %s in this data; the M4-readiness "
                    "diagnostics of the probe stage (transfer, familiarity, K capture) will be missing",
                    x.choice_evidence_levels, x.choice_rotations)
    sdir = stage_dir(cfg.name, model_id, extract_stage_name(split))
    shash = stages.stage_hash(cfg, "extract", model_id, split, prompt_mode)
    if io.finished_stage(sdir, shash, force):
        log.info("extract already finished for %s with this config and code (%s); skipping", model_id, sdir)
        return sdir
    io.prepare_stage_dir(sdir, cfg.to_yaml(), shash, force=force,
                         extra_meta=stages.stage_meta("extract", model_id, split, prompt_mode))
    stages.check_resume_environment(sdir, force)
    handler = add_file_log(sdir / "log.txt")
    t0 = time.time()
    try:
        lm = apply_prompt_mode(load_model(model_id), prompt_mode)
        stages.record_model_revision(sdir, lm.model)
        n_blocks = len(get_blocks(lm.model))
        layers = selected_layers(n_blocks, x.layer_stride)
        d_model = int(lm.model.config.hidden_size)
        total = sum(estimate_bytes(len(recs), len(layers), d_model, len(SETS[name]))
                    for name, recs in (("probe", probes), ("choice", choice)))
        log.info("extract: %d probe + %d choice items, %d layers x d=%d -> about %.2f GB of float16 activations",
                 len(probes), len(choice), len(layers), d_model, total / 2**30)
        io.write_json(sdir / "layers.json", {"layers": layers, "n_blocks": n_blocks, "d_model": d_model,
                                             "is_chat": lm.is_chat, "estimated_bytes": total,
                                             "layer_meaning": "hidden_states index; i = residual stream after block "
                                                              "i-1; 0 = embeddings; post-final-norm excluded"})
        for set_name, recs in (("probe", probes), ("choice", choice)):
            out = sdir / set_name
            out.mkdir(parents=True, exist_ok=True)
            n_shards = -(-len(recs) // x.shard_size) if recs else 0
            for idx in range(n_shards):
                marker = out / f"shard_{idx:05d}.done"
                if marker.exists():
                    log.info("extract %s: shard %d/%d already done", set_name, idx + 1, n_shards)
                    continue
                chunk = recs[idx * x.shard_size: (idx + 1) * x.shard_size]
                seqs, positions, index = prepare_records(lm, set_name, chunk)
                acts = hidden_at(lm, seqs, positions, layers, x.max_batch_size, x.max_batch_tokens)
                for pos_name in SETS[set_name]:
                    _save_npy(out / f"shard_{idx:05d}.{pos_name}.npy", np.stack([a[pos_name] for a in acts]))
                io.write_parquet(out / f"shard_{idx:05d}.index.parquet", index)
                io.atomic_write_text(marker, f"{len(chunk)}\n")
                log.info("extract %s: shard %d/%d written (%d items)", set_name, idx + 1, n_shards, len(chunk))
                from rtp import events

                events.emit("shard", f"extract {set_name} shard {idx + 1}/{n_shards}", shard=idx, n_shards=n_shards)
        from rtp.provenance import environment

        io.mark_done(sdir, {"model_id": model_id, "split": split, "layers": layers, "d_model": d_model,
                            "n_probe": len(probes), "n_choice": len(choice), "finished_with": environment(),
                            "elapsed_s": round(time.time() - t0, 1)})
        io.atomic_write_text(sdir / "summary.md",
                             f"# Extract: {model_id} ({split})\n\n- layers (hidden-state indices): {layers}\n"
                             f"- probe items: {len(probes)}; choice items: {len(choice)} (evidence "
                             f"{x.choice_evidence_levels}, rotations {x.choice_rotations})\n"
                             f"- about {total / 2**30:.2f} GB float16\n")
        log.info("extract finished: %s", sdir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return sdir


def load_activations(sdir: Path, set_name: str, position: str = "name") -> tuple[pd.DataFrame, np.ndarray]:
    """(index, float16 array [n, n_layers, d]) concatenated over finished shards of one set."""
    out = Path(sdir) / set_name
    shards = sorted(p for p in out.glob("shard_*.done"))
    if not shards:
        raise FileNotFoundError(f"no finished {set_name} shards in {out}")
    idx = [pd.read_parquet(out / f"{p.stem}.index.parquet") for p in shards]
    arr = [np.load(out / f"{p.stem}.{position}.npy") for p in shards]
    return pd.concat(idx, ignore_index=True), np.concatenate(arr, axis=0)
