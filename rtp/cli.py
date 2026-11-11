"""Unified entry point: ``python -m rtp.cli <stage> [options]``."""

from __future__ import annotations

import argparse
import logging
import sys

from rtp.log import setup_logging

log = logging.getLogger("rtp.cli")

FUTURE_STAGES = {
    "extract": "M3", "probe": "M3", "transplant": "M4", "score": "M4", "plant": "M5", "eval": "M6",
}


def _add_model_stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="YAML run config, e.g. configs/pilot.yaml")
    p.add_argument("--model", required=True, help="one model ID (see rtp/registry.py) or local path")
    p.add_argument("--split", default="dev", choices=["dev", "test"], help="data split (default: dev)")
    p.add_argument("--unseal", action="store_true", help="required to read the sealed test split")
    p.add_argument("--force", action="store_true", help="discard existing outputs of this stage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rtp.cli", description=__doc__)
    parser.add_argument("--work-dir", help="override $RTP_WORK_DIR for this invocation")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("smoke", help="end-to-end smoke run (M0 forward, M1 data, M2 measurement)")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--resume", action="store_true",
                   help="reuse finished smoke outputs instead of starting from a clean runs/<name>/ dir")

    p = sub.add_parser("build-data", help="generate catalogue, choice items and probe items (M1)")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("measure", help="choice-letter scoring and name effects (M2)")
    _add_model_stage_args(p)

    p = sub.add_parser("open-ended", help="Name / Choose / Recommend / pick measurement (M2)")
    _add_model_stage_args(p)

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

    from rtp.data.splits import SealedSplitError

    try:
        return _dispatch(args)
    except SealedSplitError as exc:
        log.error("%s", exc)
        return 3


def _dispatch(args: argparse.Namespace) -> int:
    from rtp.config import load_config

    if args.stage == "smoke":
        from rtp.smoke import run_smoke

        return run_smoke(args.config, resume=args.resume)
    if args.stage == "make-placebos":
        from rtp.data.placebo import make_placebo_file

        make_placebo_file(args.categories_file, args.natural_negatives_file, args.out,
                          n_choice=args.n_choice, n_probe=args.n_probe, seed=args.seed)
        return 0

    cfg = load_config(args.config)
    if args.stage == "build-data":
        from rtp.data.build import build_data

        build_data(cfg, force=args.force)
    elif args.stage == "measure":
        from rtp.measure.choice import run_measure

        run_measure(cfg, args.model, split=args.split, unseal=args.unseal, force=args.force)
    elif args.stage == "open-ended":
        from rtp.measure.open_ended import run_open_ended

        run_open_ended(cfg, args.model, split=args.split, unseal=args.unseal, force=args.force)
    elif args.stage == "report":
        from rtp.report import run_report

        run_report(cfg, unseal=args.unseal)
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown stage {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
