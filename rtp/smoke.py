"""``python -m rtp.cli smoke``: small end-to-end run of instrument v2 on the smoke model.

Steps: load + one forward pass -> build data (legacy and ladder items) -> seal and determinism checks ->
one CYCLE (rtp.cycle): the ladder readout on the smoke model and on its pure-scale variant ``@scale=2``,
the open-set stage, the CPU analysis and the scorecard -> the scale identity: the variant's letter-logit
differences are exactly twice the model's, so the scale multiplier must be 2 and every preference level
unchanged. Starts from a clean ``runs/<config name>/`` unless ``--resume``.

The legacy four-option stages and the frozen white-box stages are not part of the smoke run; their own
tests cover them.
"""

from __future__ import annotations

import logging
import platform
import shutil
import time

from rtp import registry
from rtp.config import load_config
from rtp.paths import model_slug, run_dir, work_dir
from rtp.scorecard import IDENTITY_TOL

log = logging.getLogger(__name__)

SMOKE_BUDGET_SECONDS = 600  # smoke end-to-end within 10 minutes (on GPU)
SMOKE_MARKER = "_SMOKE_RUN"
SCALE = 2.0  # the pure-scale variant measured next to the smoke model
# float32 logits times a power of two are exact; the tolerance covers the estimators' arithmetic. The same
# constant decides the pure-scale finding of every scorecard.
SCALE_TOL = IDENTITY_TOL


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

    from rtp.data.build import CHOICE, generate, ladder_file, load_choice_items, load_ladder_items
    from rtp.data.splits import SealedSplitError
    from rtp.io import dumps_record

    readers = [lambda: load_choice_items(cfg, "test")]
    if cfg.ladder.enabled:
        readers.append(lambda: load_ladder_items(cfg, "core", "test"))
    for read in readers:
        try:
            read()
        except SealedSplitError:
            pass
        else:
            raise SmokeFailure("reading the test split without unseal did not raise")
    again = generate(cfg)
    files = [(CHOICE, [r for r in again["choice"] if r["split"] == "dev"])]
    if cfg.ladder.enabled:
        files += [(ladder_file(t), [r for r in again["ladder"] if r["split"] == "dev" and r["tier"] == t])
                  for t in ("core", "ext", "ctx")]
    for name, recs in files:
        blob = "".join(dumps_record(r) + "\n" for r in recs).encode("utf-8")
        if hashlib.sha256(blob).hexdigest() != hashlib.sha256((ddir / "dev" / name).read_bytes()).hexdigest():
            raise SmokeFailure(f"regenerating {name} with the same seed gave different bytes")
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
        own = ("data", "feedback", "analysis", "cycles")  # plus the smoke model and its variants
        foreign = sorted(d.name for d in rdir.iterdir()
                         if d.is_dir() and d.name not in own and not d.name.startswith(model_slug(model_id)))
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

    # ---- one cycle: ladder readout (model and pure-scale variant), open-set, analysis, scorecard
    from rtp import io
    from rtp.cycle import CycleSpec, Job, cycle_dir, run_cycle
    from rtp.paths import stage_dir

    if not cfg.ladder.enabled:
        raise SmokeFailure(f"config '{cfg.name}' has no ladder items (ladder.enabled); the smoke run needs them")
    scaled = f"{model_id}@scale={SCALE:g}"
    spec = CycleSpec(name="smoke", config=config_path, phase="develop", budget_hours=1.0,
                     question="Does the pipeline run end to end, and does a pure scale change leave every "
                              "preference level unchanged?",
                     jobs=[Job("ladder", model_id, "core"), Job("ladder", scaled, "core"), Job("open-set", model_id)],
                     pairs=[(model_id, scaled)])
    # the stages run in their own processes: send their events (errors with tracebacks) to the smoke event
    # file, which the feedback bundle reads and which survives the next smoke run
    events_path = run_dir(cfg.name).parent / f"{cfg.name}.smoke_events.jsonl"
    rc = step("cycle", lambda: run_cycle(spec, sets=overrides, restart=not resume,
                                         env={"RTP_EVENTS_PATH": str(events_path)}))
    if rc != 0:
        ledger = io.read_json(cycle_dir(cfg, spec) / "ledger.json")
        bad = "; ".join(f"{e['job']}: {e['state']}" for e in ledger if e["state"] != "done")
        raise SmokeFailure(f"the smoke cycle ended with exit code {rc} ({bad}): see "
                           f"{cycle_dir(cfg, spec) / 'scorecard.md'} and the feedback bundle")
    dirs = [ddir, stage_dir(cfg.name, model_id, "ladder_core"), stage_dir(cfg.name, scaled, "ladder_core"),
            stage_dir(cfg.name, model_id, "open_set")]
    for d in dirs:
        if not io.is_done(d) or not (d / "summary.md").exists():
            raise SmokeFailure(f"stage dir {d} lacks _DONE or summary.md")
    card = io.read_json(cycle_dir(cfg, spec) / "scorecard.json")
    pair = card["pairs"].get(f"{model_id} -> {scaled}") or {}
    ident = pair.get("identity")
    if "error" in pair or not ident:
        raise SmokeFailure(f"the scorecard has no comparison of the model with its scale variant: {pair}")
    if not ident.get("n_curves") or not ident.get("n_brand_cells") or ident.get("max_abs_theta_diff") is None:
        raise SmokeFailure(f"the pure-scale identity could not be checked: no rising anchor curve or no brand with a "
                           f"preference level in both models ({ident}); the smoke model or config changed?")
    if abs(ident["c_min"] - SCALE) > SCALE_TOL or abs(ident["c_max"] - SCALE) > SCALE_TOL \
            or ident["max_abs_theta_diff"] > SCALE_TOL or ident["n_flag_changes"]:
        raise SmokeFailure(f"pure-scale identity violated: scale multipliers {ident['c_min']}..{ident['c_max']} "
                           f"(expected {SCALE:g}), largest change of a preference level "
                           f"{ident['max_abs_theta_diff']} (expected 0), {ident['n_flag_changes']} brand cell(s) "
                           f"changed their censoring or lost their level (expected 0)")
    info = io.read_json(dirs[1] / io.DONE_FILE)

    total = time.time() - t_start
    log.info("SMOKE PASSED in %.1fs on %s", total, hw)
    for name, sec in timings.items():
        log.info("  %-16s %7.1fs", name, sec)
    log.info("pure-scale identity on %d anchor curves and %d brand cells: multipliers %s..%s (expected %g), largest "
             "change of a preference level %.2e, %d flag changes", ident["n_curves"], ident["n_brand_cells"],
             ident["c_min"], ident["c_max"], SCALE, ident["max_abs_theta_diff"], ident["n_flag_changes"])
    log.info("letter mass at the forced position: %s; scorecard: %s", info.get("c_forced"),
             cycle_dir(cfg, spec) / "scorecard.md")
    if total > SMOKE_BUDGET_SECONDS:
        log.warning("smoke took %.0fs, above the %ds budget (the budget is defined for a GPU)", total,
                    SMOKE_BUDGET_SECONDS)
    return 0
