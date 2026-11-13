"""Unified entry point: ``python -m rtp.cli <stage> [options]``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

# PyTorch only. Colab preinstalls TensorFlow and JAX, which transformers/datasets would otherwise import on first use:
# seconds of start-up per stage, and TensorFlow log lines such as "Could not find cuda drivers on your machine, GPU
# will not be used" that look like a GPU problem but are not PyTorch's. setdefault: an explicit setting still wins.
TORCH_ONLY_ENV = {"USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
for _k, _v in TORCH_ONLY_ENV.items():
    os.environ.setdefault(_k, _v)

from rtp.log import setup_logging

log = logging.getLogger("rtp.cli")

FUTURE_STAGES = {"transplant": "M4", "score": "M4", "plant": "M5", "eval": "M6"}


def _add_model_stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="YAML run config, e.g. configs/pilot.yaml")
    p.add_argument("--model", required=True, help="one model ID (see rtp/registry.py) or local path")
    p.add_argument("--split", default="dev", choices=["dev", "test"], help="data split (default: dev)")
    p.add_argument("--unseal", action="store_true", help="required to read the sealed test split")
    p.add_argument("--force", action="store_true", help="discard existing outputs of this stage")
    p.add_argument("--prompt-mode", default="auto", choices=["auto", "chat", "base"],
                   help="auto: registry decides (unregistered paths: guessed from the tokenizer, with a warning); "
                        "chat: apply_chat_template; base: plain-text continuation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rtp.cli", description=__doc__)
    parser.add_argument("--work-dir", help="override $RTP_WORK_DIR for this invocation")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set measure.max_batch_size=128 (YAML value; repeatable)")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("smoke", help="end-to-end smoke run (M0 forward, M1 data, M2 measurement, M3 extract, "
                                     "probe and localize)")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--resume", action="store_true",
                   help="reuse finished smoke outputs instead of starting from a clean runs/<name>/ dir")

    p = sub.add_parser("build-data", help="generate catalogue, choice items and probe items (M1)")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("measure", help="choice-letter scoring and name effects (M2)")
    _add_model_stage_args(p)

    p = sub.add_parser("format-probe", help="quick check of how a model answers the choice question (dev sample)")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n-items", type=int, default=40, help="items to score and decode (default 40)")
    p.add_argument("--prompt-mode", default="auto", choices=["auto", "chat", "base"])

    p = sub.add_parser("open-ended", help="Name / Choose / Recommend / pick measurement (M2)")
    _add_model_stage_args(p)

    p = sub.add_parser("extract", help="activations at the brand name / answer position, float16 shards (M3)")
    _add_model_stage_args(p)

    p = sub.add_parser("probe", help="linear probes, control task, knowledge subspace K, M4 readiness (M3; dev only)")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--prompt-mode", default="auto", choices=["auto", "chat", "base"],
                   help="must match the prompt mode used by `extract`")

    p = sub.add_parser("localize", help="where the name effect is carried: full transplants by layer and site "
                                        "(M3 readiness; dev only)")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--prompt-mode", default="auto", choices=["auto", "chat", "base"])

    p = sub.add_parser("report", help="cross-model Markdown report from finished stages")
    p.add_argument("--config", required=True)
    p.add_argument("--unseal", action="store_true", help="also include test-split results")

    p = sub.add_parser("make-placebos", help="(seed maintenance) regenerate data/seeds/placebo_names.yaml")
    p.add_argument("--categories-file", default="data/seeds/categories.yaml")
    p.add_argument("--natural-negatives-file", default="data/seeds/natural_negatives.csv")
    p.add_argument("--out", default="data/seeds/placebo_names.yaml")
    p.add_argument("--n-choice", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repair", action="store_true",
                   help="keep the existing file; replace only unusable names (A-D initial, conflicts)")

    p = sub.add_parser("status", help="state of every stage of every model under a config")
    p.add_argument("--config", required=True)
    p.add_argument("--unseal", action="store_true", help="also show results of test-split stages")

    p = sub.add_parser("doctor", help="environment preflight: GPU, bf16, library pins, HF token, disks, Hub access")
    p.add_argument("--config", help="also check the config, its data and Hub access to its models")
    p.add_argument("--no-hub", action="store_true", help="skip the network check of the configured models")

    p = sub.add_parser("feedback", help="write a feedback bundle (status, errors, warnings, results) for an agent")
    p.add_argument("--config", required=True)
    p.add_argument("--tail", type=int, default=60, help="log lines to include per failed/partial stage")
    p.add_argument("--unseal", action="store_true", help="also include results of test-split stages")

    for name, milestone in FUTURE_STAGES.items():
        sub.add_parser(name, help=f"not implemented yet ({milestone})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    if args.work_dir:
        from rtp.paths import set_work_dir

        set_work_dir(args.work_dir)

    if args.stage in FUTURE_STAGES:
        log.error("stage '%s' belongs to %s and is not implemented in this round", args.stage,
                  FUTURE_STAGES[args.stage])
        return 2

    import time
    import traceback

    from rtp import events
    from rtp.config import ConfigError
    from rtp.data.build import DataNotBuiltError
    from rtp.data.splits import SealedSplitError
    from rtp.io import StageConfigMismatch
    from rtp.models import ModelLoadError
    from rtp.paths import WorkDirError
    from rtp.smoke import SmokeFailure

    handler = _start_events(args, argv)
    t0 = time.time()
    try:
        rc = _dispatch(args)
    except SealedSplitError as exc:
        log.error("%s", exc)
        events.emit("error", f"SealedSplitError: {exc}", exit_code=3, argv=_argv(argv))
        rc = 3
    except (StageConfigMismatch, DataNotBuiltError, ModelLoadError, WorkDirError, ConfigError, SmokeFailure) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        events.emit("error", f"{type(exc).__name__}: {exc}", exit_code=4, argv=_argv(argv),
                    traceback=traceback.format_exc())
        rc = 4
    except BaseException as exc:  # unexpected: record the traceback for the feedback bundle, then re-raise
        events.emit("error", f"{type(exc).__name__}: {exc}", exit_code=1, argv=_argv(argv),
                    traceback=traceback.format_exc())
        raise
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
    events.emit("done", f"exit {rc}", exit_code=rc, seconds=round(time.time() - t0, 1), argv=_argv(argv))
    return rc


def _argv(argv: list[str] | None) -> list[str]:
    return list(argv if argv is not None else sys.argv[1:])


def _start_events(args: argparse.Namespace, argv: list[str] | None):
    """Point the event log at runs/<config>/events.jsonl when the command has a config (best effort)."""
    from rtp import events
    from rtp.log import RedactSecrets

    cfg_path = getattr(args, "config", None) or ("configs/smoke.yaml" if args.stage == "smoke" else None)
    if not cfg_path:
        return None
    try:
        from rtp.config import load_config
        from rtp.paths import run_dir

        name = load_config(cfg_path).name
        # smoke deletes and recreates its own run dir, so its events live next to it, never inside it
        path = run_dir(name).parent / f"{name}.smoke_events.jsonl" if args.stage == "smoke" else \
            run_dir(name) / "events.jsonl"
    except Exception:  # noqa: BLE001 - the command itself will report config / work-dir problems
        return None
    events.configure(path, config=name, stage=args.stage, model=getattr(args, "model", None))
    handler = events.EventLogHandler()
    handler.addFilter(RedactSecrets())
    logging.getLogger().addHandler(handler)
    events.emit("start", " ".join(_argv(argv)), argv=_argv(argv))
    return handler


def _dispatch(args: argparse.Namespace) -> int:
    from rtp.config import load_config

    if args.stage == "smoke":
        from rtp.smoke import run_smoke

        return run_smoke(args.config, resume=args.resume, overrides=args.set)
    if args.stage == "doctor":
        from rtp.config import apply_overrides
        from rtp.ops import run_doctor

        cfg = apply_overrides(load_config(args.config), args.set) if args.config else None
        return run_doctor(cfg, check_hub=not args.no_hub)
    if args.stage == "make-placebos":
        from rtp.data.placebo import make_placebo_file

        make_placebo_file(args.categories_file, args.natural_negatives_file, args.out,
                          n_choice=args.n_choice, n_probe=args.n_probe, seed=args.seed, repair=args.repair)
        return 0

    from rtp.config import apply_overrides

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.stage == "status":
        from rtp.ops import run_status

        run_status(cfg, unseal=args.unseal)
        return 0
    if args.stage == "feedback":
        from rtp.ops import build_feedback

        build_feedback(cfg, tail=args.tail, unseal=args.unseal)
        return 0
    if args.stage in ("measure", "open-ended", "extract"):
        from rtp.data.splits import check_split_access

        check_split_access(args.split, args.unseal)  # fail before any model is loaded
    if args.stage == "build-data":
        from rtp.data.build import build_data

        build_data(cfg, force=args.force)
    elif args.stage == "measure":
        from rtp.measure.choice import run_measure

        run_measure(cfg, args.model, split=args.split, unseal=args.unseal, force=args.force,
                    prompt_mode=args.prompt_mode)
    elif args.stage == "format-probe":
        from rtp.measure.format_probe import run_format_probe

        run_format_probe(cfg, args.model, n_items=args.n_items, prompt_mode=args.prompt_mode)
    elif args.stage == "open-ended":
        from rtp.measure.open_ended import run_open_ended

        run_open_ended(cfg, args.model, split=args.split, unseal=args.unseal, force=args.force,
                       prompt_mode=args.prompt_mode)
    elif args.stage == "extract":
        from rtp.detect.activations import run_extract

        run_extract(cfg, args.model, split=args.split, unseal=args.unseal, force=args.force,
                    prompt_mode=args.prompt_mode)
    elif args.stage == "probe":
        from rtp.detect.probes import run_probe

        run_probe(cfg, args.model, force=args.force, prompt_mode=args.prompt_mode)
    elif args.stage == "localize":
        from rtp.detect.localize import run_localize

        run_localize(cfg, args.model, force=args.force, prompt_mode=args.prompt_mode)
    elif args.stage == "report":
        from rtp.report import run_report

        run_report(cfg, unseal=args.unseal)
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown stage {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
