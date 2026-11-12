"""``python -m rtp.cli smoke``: small end-to-end run on the smoke model.

Steps: M0 load + one forward pass -> M1 build data -> seal check -> M2 measure ->
M2 open-ended -> M3 extract -> M3 probe -> report. Starts from a clean ``runs/<config name>/``
unless ``--resume``.
"""

from __future__ import annotations

import logging
import platform
import shutil
import time

from rtp import registry
from rtp.config import load_config
from rtp.paths import model_slug, run_dir, work_dir

log = logging.getLogger(__name__)

SMOKE_BUDGET_SECONDS = 600  # spec §6 M2: smoke end-to-end within 10 minutes (on GPU)
SMOKE_MARKER = "_SMOKE_RUN"


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

    from rtp.models import chat_prompt, encode, get_blocks

    text = chat_prompt(lm, "Say hello.") if lm.is_chat else "Hello"
    ids = torch.tensor([encode(lm.tokenizer, text, lm.is_chat)], device=lm.device)
    with torch.inference_mode():
        logits = lm.model(input_ids=ids).logits
    if not torch.isfinite(logits).all():
        raise SmokeFailure("forward pass produced non-finite logits")
    log.info("forward OK: %d prompt tokens, vocab %d, %d decoder blocks", ids.shape[1], logits.shape[-1],
             len(get_blocks(lm.model)))


def check_data(cfg, ddir) -> None:
    """M1 acceptance inside smoke: the test split refuses to open, and regeneration is byte-identical."""
    import hashlib

    from rtp.data.build import CHOICE, generate, load_choice_items
    from rtp.data.splits import SealedSplitError
    from rtp.io import dumps_record

    try:
        load_choice_items(cfg, "test")
    except SealedSplitError:
        pass
    else:
        raise SmokeFailure("reading the test split without unseal did not raise")
    again = generate(cfg)["choice"]
    blob = "".join(dumps_record(r) + "\n" for r in again if r["split"] == "dev").encode("utf-8")
    if hashlib.sha256(blob).hexdigest() != hashlib.sha256((ddir / "dev" / CHOICE).read_bytes()).hexdigest():
        raise SmokeFailure("regenerating the data with the same seed gave different bytes")
    log.info("data OK: test split sealed, regeneration byte-identical")


def run_smoke(config_path: str = "configs/smoke.yaml", resume: bool = False, overrides: list[str] | None = None) -> int:
    from rtp.config import apply_overrides

    t_start = time.time()
    cfg = apply_overrides(load_config(config_path), overrides)
    model_id = registry.SMOKE_MODEL
    hw = hardware_summary()
    log.info("smoke: config=%s model=%s hardware=%s", cfg.name, model_id, hw)

    rdir = run_dir(cfg.name)
    bookkeeping = {"events.jsonl", "status.md", "doctor.json", "pytest_last.txt", "feedback", "report.md"}
    if rdir.exists() and not (rdir / SMOKE_MARKER).exists() and {p.name for p in rdir.iterdir()} <= bookkeeping:
        (rdir / SMOKE_MARKER).write_text("created by `python -m rtp.cli smoke`; smoke may delete this directory\n")
    if rdir.exists() and not (rdir / SMOKE_MARKER).exists():
        # never adopt (and later delete) a run dir that smoke did not create, not even with --resume
        raise SmokeFailure(
            f"refusing to use {rdir}: it was not created by `smoke` (no {SMOKE_MARKER} marker) and may hold real "
            f"results. Use a config with another name (e.g. configs/smoke.yaml) or remove the directory yourself.")
    if not resume and rdir.exists():
        # any directory other than smoke's own data/model dirs and bookkeeping may hold real results: never delete
        foreign = sorted(d.name for d in rdir.iterdir()
                         if d.is_dir() and d.name not in ("data", "feedback", model_slug(model_id)))
        if rdir.resolve().parent != (work_dir() / "runs").resolve() or foreign:
            raise SmokeFailure(f"refusing to delete {rdir}: unexpected location or results of other models {foreign}")
        log.info("smoke: removing previous smoke outputs in %s", rdir)
        shutil.rmtree(rdir)
    if not rdir.exists():
        rdir.mkdir(parents=True)
        (rdir / SMOKE_MARKER).write_text("created by `python -m rtp.cli smoke`; smoke may delete this directory\n")

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

    # ---- M1
    from rtp.data.build import build_data

    ddir = step("build-data", lambda: build_data(cfg))
    step("seal+determinism", lambda: check_data(cfg, ddir))
    del lm  # the M2 stages load the model themselves (one model per stage)

    # ---- M2
    from rtp import io
    from rtp.measure.choice import run_measure
    from rtp.measure.open_ended import run_open_ended
    from rtp.report import run_report

    mdir = step("measure", lambda: run_measure(cfg, model_id, split="dev"))
    odir = step("open-ended", lambda: run_open_ended(cfg, model_id, split="dev"))

    # ---- M3
    from rtp.detect.activations import run_extract
    from rtp.detect.probes import run_probe

    edir = step("extract", lambda: run_extract(cfg, model_id, split="dev"))
    pdir = step("probe", lambda: run_probe(cfg, model_id))
    rpath = step("report", lambda: run_report(cfg))
    for d in (ddir, mdir, odir, edir, pdir):
        if not io.is_done(d) or not (d / "summary.md").exists():
            raise SmokeFailure(f"stage dir {d} lacks _DONE or summary.md")
    info = io.read_json(mdir / io.DONE_FILE)
    cons = info["consistency"]

    total = time.time() - t_start
    log.info("SMOKE PASSED in %.1fs on %s", total, hw)
    for name, sec in timings.items():
        log.info("  %-16s %7.1fs", name, sec)
    log.info("free-answer consistency (greedy answer WITHOUT prefill vs scored letter argmax): %s on %d items%s",
             f"{cons['agreement']:.1%}" if cons["n_checked"] else "n/a", cons["n_checked"],
             "  (WARNING: below threshold)" if info.get("consistency_warning") else "")
    log.info("outputs: %s | %s | %s", mdir, odir, rpath)
    if total > SMOKE_BUDGET_SECONDS:
        log.warning("smoke took %.0fs, above the %ds budget (the budget is defined for a GPU)", total,
                    SMOKE_BUDGET_SECONDS)
    return 0
