"""Cost model for the ``ladder`` and ``open-set`` stages: predicted GPU seconds per job.

Seconds = k * (prompt tokens * S_PER_TOKEN + items * S_PER_ITEM) + model load. The two rates are priors
fitted on the first pilot (RTX PRO 6000, bf16, legacy four-option items: R^2 0.75-0.80); the per-item rate
of the new single-pass readout is expected to be lower, so the prior errs on the long side. ``k`` is
recalibrated from every finished stage of the run (actual scoring seconds / prior), so the estimates
converge after the first job of a campaign. The token count of a tier is taken from a finished stage when
one exists (measured), otherwise from the prompt characters of the built items.
"""

from __future__ import annotations

from typing import Any

from rtp import io, registry
from rtp.config import Config
from rtp.models import is_cached
from rtp.paths import data_dir, run_dir

S_PER_TOKEN = 6.3e-5
S_PER_ITEM = 7.4e-3
CHARS_PER_TOKEN = 3.6  # prompt characters per token, before any stage has measured the real count
FRAME_TOKENS = 30  # chat template and prefill tokens per item (Tulu 3: 12, Olmo 3: 38)
LOAD_FIRST_S = 300.0  # a checkpoint that is not on local disk: download (16 GB) and load
LOAD_AGAIN_S = 60.0  # a checkpoint already in the local cache
OPEN_SET_S = 240.0
ANALYZE_S = 30.0


def _done_infos(cfg: Config) -> list[dict[str, Any]]:
    out = []
    for done in sorted(run_dir(cfg.name).glob(f"*/ladder_*/{io.DONE_FILE}")):
        try:
            out.append(io.read_json(done))
        except Exception:  # noqa: BLE001 - a broken file must not break planning
            continue
    return out


def prior_seconds(n_tokens: float, n_items: float) -> float:
    return n_tokens * S_PER_TOKEN + n_items * S_PER_ITEM


def calibration(cfg: Config) -> dict[str, Any]:
    """Multiplier k (median of actual / prior over finished stages with complete timing), the measured
    tokens per (split, tier), and the measured load times of a cached and of an uncached checkpoint."""
    ratios, loads, cold = [], [], []
    tokens: dict[str, int] = {}
    for info in _done_infos(cfg):
        key = f"{info.get('split')}/{info.get('tier')}"
        tokens.setdefault(key, int(info.get("n_tokens", 0)))
        if not info.get("timing_complete"):
            continue
        t = info.get("timing", {})
        actual = float(t.get("primary_s", 0.0)) + float(t.get("diag_s", 0.0)) + float(t.get("gen_s", 0.0))
        prior = prior_seconds(float(info["n_tokens"]), float(info["n_items"]))
        if prior > 0 and actual > 0:
            ratios.append(actual / prior)
        if t.get("load_s") and info.get("load_cached") is not None:
            (loads if info["load_cached"] else cold).append(float(t["load_s"]))
    ratios.sort()
    loads.sort()
    cold.sort()
    return {"k": ratios[len(ratios) // 2] if ratios else 1.0, "n_stages": len(ratios), "tokens": tokens,
            "load_again_s": loads[len(loads) // 2] if loads else LOAD_AGAIN_S,
            "load_first_s": cold[len(cold) // 2] if cold else LOAD_FIRST_S}


def tier_size(cfg: Config, tier: str, split: str, calib: dict[str, Any] | None = None) -> tuple[int, float, bool]:
    """(items, prompt tokens, measured?) of a tier file."""
    stats = io.read_json(data_dir(cfg.name) / "panel_stats.json").get(f"{split}/{tier}")
    if stats is None:
        raise KeyError(f"no ladder items for {split}/{tier} in the built data of '{cfg.name}'")
    measured = (calib or {}).get("tokens", {}).get(f"{split}/{tier}")
    if measured:
        return stats["n_items"], float(measured), True
    return stats["n_items"], stats["prompt_chars"] / CHARS_PER_TOKEN + stats["n_items"] * FRAME_TOKENS, False


def estimate(cfg: Config, stage: str, model_id: str, tier: str | None = None, split: str = "dev",
             calib: dict[str, Any] | None = None, seen: set[str] | None = None) -> dict[str, Any]:
    """Predicted seconds of one job. ``seen``: checkpoints loaded earlier in the same plan (cached by then)."""
    calib = calib or calibration(cfg)
    base = registry.base_model_id(model_id)
    cached = base in (seen or set()) or is_cached(model_id)
    load = calib["load_again_s"] if cached else calib.get("load_first_s", LOAD_FIRST_S)
    if stage == "ladder":
        n_items, n_tokens, measured = tier_size(cfg, tier or "core", split, calib)
        work = calib["k"] * prior_seconds(n_tokens, n_items)
        return {"seconds": work + load, "work_s": work, "load_s": load, "n_items": n_items,
                "n_tokens": int(n_tokens), "tokens_measured": measured}
    if stage == "open_set":
        return {"seconds": OPEN_SET_S + load, "work_s": OPEN_SET_S, "load_s": load}
    raise ValueError(f"no cost model for stage '{stage}'")
