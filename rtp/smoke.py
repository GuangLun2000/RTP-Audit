"""``python -m rtp.cli smoke``: small end-to-end run on the smoke model.

Steps: M0 load + one forward pass -> M1 build data -> seal check -> M2 measure ->
M2 open-ended -> report. Starts from a clean ``runs/<config name>/`` unless ``--resume``.
"""

from __future__ import annotations

import logging
import platform
import shutil
import time

from rtp import registry
from rtp.config import load_config
from rtp.paths import run_dir

log = logging.getLogger(__name__)

SMOKE_BUDGET_SECONDS = 600  # spec §6 M2: smoke end-to-end within 10 minutes (on GPU)


class SmokeFailure(RuntimeError):
    pass


def hardware_summary() -> str:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"CUDA {props.name} ({props.total_memory / 2**30:.0f} GiB), torch {torch.__version__}"
    return f"CPU only ({platform.processor() or platform.machine()}), torch {torch.__version__}"


def check_forward(lm) -> None:
    """M0 acceptance: one forward pass with finite logits through the chat template."""
    import torch

    from rtp.models import encode, format_chat, get_blocks

    text = format_chat(lm.tokenizer, "Say hello.") if lm.is_chat else "Hello"
    ids = torch.tensor([encode(lm.tokenizer, text, lm.is_chat)], device=lm.device)
    with torch.inference_mode():
        logits = lm.model(input_ids=ids).logits
    if not torch.isfinite(logits).all():
        raise SmokeFailure("forward pass produced non-finite logits")
    log.info("forward OK: %d prompt tokens, vocab %d, %d decoder blocks", ids.shape[1], logits.shape[-1],
             len(get_blocks(lm.model)))


def run_smoke(config_path: str = "configs/smoke.yaml", resume: bool = False) -> int:
    t_start = time.time()
    cfg = load_config(config_path)
    model_id = registry.SMOKE_MODEL
    hw = hardware_summary()
    log.info("smoke: config=%s model=%s hardware=%s", cfg.name, model_id, hw)

    rdir = run_dir(cfg.name)
    if not resume and rdir.exists():
        if rdir.name != cfg.name or rdir.parent.name != "runs":
            raise SmokeFailure(f"refusing to delete unexpected path {rdir}")
        log.info("smoke: removing previous outputs in %s", rdir)
        shutil.rmtree(rdir)

    timings: dict[str, float] = {}

    def step(name: str, fn):
        t0 = time.time()
        out = fn()
        timings[name] = time.time() - t0
        log.info("smoke step '%s' finished in %.1fs", name, timings[name])
        return out

    from rtp.models import load_model

    lm = step("load model", lambda: load_model(model_id))
    step("forward", lambda: check_forward(lm))

    total = time.time() - t_start
    log.info("SMOKE PASSED in %.1fs on %s", total, hw)
    for name, sec in timings.items():
        log.info("  %-14s %7.1fs", name, sec)
    return 0
