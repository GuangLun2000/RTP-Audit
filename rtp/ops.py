"""Operations for running on Colab and closing the improvement loop: ``status``, ``doctor``, ``feedback``.

* ``status``: one table of every stage of every model under a config: done / partial (shards) /
  stale (built by other code, data or settings) / not started, with its warnings.
* ``doctor``: environment preflight (GPU, bf16, library pins, HF token presence, disks, Drive,
  config/data, Hub access to the configured models). Writes ``doctor.json``.
* ``feedback``: a self-contained Markdown (+ JSON) bundle for a coding agent: environment, stage
  status, every error with its traceback, deduplicated warnings, key results, rule-based suggested
  next actions, log tails of failed or partial stages, and the last pytest output. See
  docs/AGENT_LOOP.md for how the bundle is used.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from rtp import events, io, registry, stages
from rtp.config import Config
from rtp.paths import data_dir, model_slug, run_dir

log = logging.getLogger(__name__)

# stage directory name -> (stage kind, split, ladder tier)
STAGE_DIRS = {**{f"ladder_{t}": ("ladder", "dev", t) for t in ("core", "ext", "ctx")},
              **{f"ladder_{t}_test": ("ladder", "test", t) for t in ("core", "ext", "ctx")},
              "open_set": ("open_set", "dev", None), "open_set_test": ("open_set", "test", None),
              # legacy four-option family and the frozen white-box stages (still listed when present)
              "measure": ("measure", "dev", None), "measure_test": ("measure", "test", None),
              "open_ended": ("open_ended", "dev", None), "open_ended_test": ("open_ended", "test", None),
              "extract": ("extract", "dev", None), "extract_test": ("extract", "test", None),
              "probe": ("probe", "dev", None), "localize": ("localize", "dev", None)}
PIPELINE = ("ladder_core", "ladder_ext", "ladder_ctx", "open_set")  # instrument v2 (configs with ladder.enabled)
LEGACY_PIPELINE = ("measure", "open_ended")  # extract / probe / localize are frozen: run them by hand if needed


def pipeline(cfg: Config) -> tuple[str, ...]:
    """The stages a model of this config is expected to have (rows 'not started' in `status`)."""
    return PIPELINE if cfg.ladder.enabled else LEGACY_PIPELINE


# --------------------------------------------------------------------------- status
def _warnings_of(kind: str, info: dict[str, Any]) -> list[str]:
    w = []
    if kind == "ladder" and info.get("coverage_warning"):
        return [f"letter mass at the forced position {info.get('c_forced')} < 0.9 (the letter readout is partial)"]
    if kind == "open_set" and info.get("coverage_warning"):
        return [f"open-set coverage {info.get('coverage')} (the model rarely answers with a catalogue brand)"]
    if info.get("consistency_warning"):
        c = info.get("consistency", {})
        w.append(f"free-answer agreement {c.get('agreement', float('nan')):.0%} < 90%")
    if info.get("coverage_warning"):
        w.append(f"answer coverage c_forced {info.get('c_forced')} (model not naming an option after \"Answer:\")")
    if info.get("format_warning"):
        w.append(f"answer format shifts with evidence level ({info.get('format_shifts')} shifts)")
    if info.get("manipulation_warning"):
        w.append(f"manipulation ratio {info.get('manipulation_ratio')} (evidence too weak vs name effects)")
    if info.get("responds_to_ratings") is False:
        w.append("does not respond to rating evidence")
    w.extend(info.get("warnings", []) if kind in ("probe", "localize") else [])
    return w


def collect_status(cfg: Config, unseal: bool = False) -> list[dict[str, Any]]:
    """All stages with their state. Results (warnings) of test-split stages are hidden unless ``unseal``
    (hard constraint 4: method development must not see sealed outcomes)."""
    rdir = run_dir(cfg.name)
    rows: list[dict[str, Any]] = []
    ddir = data_dir(cfg.name)
    if io.is_done(ddir):
        try:
            state = "done" if io.stored_stage_hash(ddir) == cfg.data_hash else "stale"
        except Exception as exc:  # noqa: BLE001
            state = f"unknown ({type(exc).__name__})"
    else:
        state = "partial" if ddir.exists() else "not started"
    rows.append({"model": "(data)", "stage": "build-data", "state": state, "warnings": [], "elapsed_s": None,
                 "path": str(ddir), "sealed": False})
    seen = set()
    for meta_path in sorted(rdir.glob(f"*/*/{io.STAGE_META}")):
        sdir = meta_path.parent
        if sdir.name not in STAGE_DIRS:
            continue
        meta = io.read_json(meta_path)
        kind, split, tier = STAGE_DIRS[sdir.name]
        model_id = meta.get("model_id", sdir.parent.name)
        seen.add((model_id, sdir.name))
        try:
            expected = stages.stage_hash(cfg, kind, model_id, split, meta.get("prompt_mode", "auto"), tier=tier)
        except Exception:  # noqa: BLE001 - e.g. data missing: cannot verify
            expected = None
        sealed = split == "test" and not unseal
        info = io.read_json(sdir / io.DONE_FILE) if io.is_done(sdir) else {}
        n_shards = len(stages.finished_shards(sdir))
        current = meta.get("stage_hash") == expected
        if io.is_done(sdir):
            state = "done" if current else "stale"
        elif current:
            state = f"partial ({n_shards} shards done)"
        else:
            state = f"partial+stale ({n_shards} shards; other code/inputs/settings: re-run with --force)"
        rows.append({"model": model_id, "stage": sdir.name, "state": state,
                     "warnings": ["(sealed: pass --unseal to show)"] if sealed and info else _warnings_of(kind, info),
                     "elapsed_s": info.get("elapsed_s"), "path": str(sdir), "sealed": sealed})
    for m in cfg.models:
        for st in pipeline(cfg):
            if (m, st) not in seen:
                rows.append({"model": m, "stage": st, "state": "not started", "warnings": [], "elapsed_s": None,
                             "path": "", "sealed": False})
    return rows


def status_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | stage | state | warnings | elapsed |", "|---|---|---|---|---|"]
    for r in rows:
        el = "" if r["elapsed_s"] is None else f"{r['elapsed_s']:.0f}s"
        warn = "; ".join(w if len(w) <= 90 else w[:87] + "..." for w in r["warnings"]) or "—"
        lines.append(f"| {r['model']} | {r['stage']} | {r['state']} | {warn} | {el} |")
    return "\n".join(lines) + "\n"


def run_status(cfg: Config, unseal: bool = False) -> list[dict[str, Any]]:
    rows = collect_status(cfg, unseal=unseal)
    text = status_markdown(rows)
    io.atomic_write_text(run_dir(cfg.name) / "status.md", text)
    print(text)
    return rows


# --------------------------------------------------------------------------- doctor
DISK_FAIL_GIB = 2.0  # below this a session's results (about 0.1 GiB per model and tier) are at risk
PEFT_PROBE = "from peft.import_utils import is_torchao_available as f; f()"


def peft_status() -> tuple[str, str]:
    """(level, detail): can peft inject a LoRA adapter here? Colab ships a torchao that newer peft refuses the
    moment an adapter is created (first seen on 2026-11-19: torchao 0.10.0, peft wants 0.16), which breaks every
    ``zoo:`` variant. rtp never uses torchao; ``rtp.notebook.ensure_peft`` removes the incompatible copy."""
    try:
        import peft
        from peft import import_utils
    except ImportError as exc:
        return "WARN", f"peft is not importable ({exc}): LoRA variants (zoo:<run_id>) cannot be loaded"
    probe = getattr(import_utils, "is_torchao_available", None)
    try:
        if probe is not None:
            getattr(probe, "cache_clear", lambda: None)()
            probe()
    except ImportError as exc:
        return "WARN", (f"peft {peft.__version__} cannot create adapters: {exc}. Fix: pip uninstall -y torchao "
                        f"(rtp.notebook.ensure_peft does it)")
    return "PASS", f"{peft.__version__} (adapters can be created)"


def _check(results: list[dict[str, str]], name: str, level: str, detail: str) -> None:
    results.append({"check": name, "level": level, "detail": detail})


def _version_ok(installed: str | None, lo: str, hi: str) -> bool:
    if installed is None:
        return False
    from packaging.version import Version

    return Version(lo) <= Version(installed) < Version(hi)


def run_doctor(cfg: Config | None = None, check_hub: bool = True) -> int:
    import importlib.metadata as md

    from rtp.provenance import environment

    res: list[dict[str, str]] = []
    env = environment()
    _check(res, "python", "PASS" if tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10) else "FAIL",
           platform.python_version())
    try:
        import torch

        _check(res, "torch", "PASS", f"{torch.__version__} (not reinstalled by this package)")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / 2**30
            _check(res, "gpu", "PASS", f"{props.name}, {gb:.0f} GiB")
            _check(res, "bf16", "PASS" if torch.cuda.is_bf16_supported() else "FAIL", "torch.cuda.is_bf16_supported()")
            if gb < 40:
                _check(res, "gpu memory", "WARN", f"{gb:.0f} GiB: 8B models in bf16 need ~17 GiB plus batches")
        else:
            _check(res, "gpu", "WARN", "CUDA not available: CPU float32 (fine for smoke/tests, not for 8B models)")
    except ImportError as exc:
        _check(res, "torch", "FAIL", f"cannot import torch: {exc}")

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    for pkg, lo, hi in (("transformers", "4.57", "5"), ("huggingface_hub", "0.34", "1"), ("numpy", "1.26", "3"),
                        ("pandas", "2.0", "3"), ("scikit-learn", "1.3", "2")):
        v = ver(pkg)
        _check(res, pkg, "PASS" if _version_ok(v, lo, hi) else "FAIL", f"{v} (required >={lo},<{hi})")
    _check(res, "HF_TOKEN", "PASS" if os.environ.get("HF_TOKEN") else "WARN",
           "set" if os.environ.get("HF_TOKEN") else "not set: gated Llama repos will fail (Colab Secrets -> HF_TOKEN)")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    _check(res, "HF_HOME", "WARN" if hf_home.startswith("/content/drive") else "PASS",
           f"{hf_home} ({'on Drive: slow, use local disk' if hf_home.startswith('/content/drive') else 'local'})")
    for label, path in (("HF_HOME disk", hf_home), ("RTP_WORK_DIR disk", os.environ.get("RTP_WORK_DIR", ""))):
        p = Path(path) if path else None
        while p is not None and not p.exists() and p != p.parent:
            p = p.parent
        if p is not None and p.exists():
            free = shutil.disk_usage(p).free / 2**30
            if free < DISK_FAIL_GIB:  # on Drive this is the account's quota: writes stop syncing when it is used up
                _check(res, label, "FAIL", f"{free:.1f} GiB free at {p}: results may not be saved. Free space first "
                                           f"(old runs/*/extract folders take about 6 GiB per model)")
            else:
                _check(res, label, "PASS" if free > 30 else "WARN", f"{free:.0f} GiB free at {p}")
    _check(res, "peft", *peft_status())
    try:
        from rtp.paths import work_dir

        wd = work_dir()
        probe = wd / ".rtp_write_test"
        probe.write_text("ok")
        probe.unlink()
        _check(res, "RTP_WORK_DIR", "PASS", f"{wd} writable")
    except Exception as exc:  # noqa: BLE001
        _check(res, "RTP_WORK_DIR", "FAIL", str(exc))
    if cfg is not None:
        try:
            cfg.seed_file_digests()
            _check(res, "config", "PASS", f"{cfg.name}: seed files readable, data hash {cfg.data_hash}")
        except Exception as exc:  # noqa: BLE001
            _check(res, "config", "FAIL", str(exc))
        ddir = data_dir(cfg.name)
        if io.is_done(ddir):
            from rtp.datafiles import missing_files

            ok = io.stored_stage_hash(ddir) == cfg.data_hash
            lost = missing_files(cfg)
            if lost:  # cycle C1: a full Drive had dropped two item files behind a complete record
                _check(res, "data", "FAIL", f"{len(lost)} item file(s) listed in the manifest are missing: "
                       f"{', '.join(lost)}; build-data rebuilds them (same bytes, finished stages stay valid)")
            else:
                _check(res, "data", "PASS" if ok else "WARN", "built and current" if ok else
                       "built from other seeds/code: run build-data --force")
        else:
            _check(res, "data", "WARN", "not built yet (run build-data)")
        if check_hub:
            for m in dict.fromkeys([*cfg.models, *cfg.confirm_models]):
                try:
                    spec = registry.get_model_spec(m)
                    hub_id = registry.base_model_id(m)  # a variant is checked through its base checkpoint
                    registry.validate_repo_id(hub_id, "model")
                    if spec is not None and spec.gated:  # metadata is public; check that FILES are accessible
                        from huggingface_hub import HfApi

                        HfApi().auth_check(hub_id)
                    _check(res, f"hub:{m}", "PASS", "accessible" + (" (gated: token and licence OK)"
                                                                   if spec and spec.gated else ""))
                except Exception as exc:  # noqa: BLE001
                    known = registry.MODELS.get(m)
                    detail = registry.gated_message(m) if known and known.gated else str(exc).splitlines()[0][:200]
                    _check(res, f"hub:{m}", "FAIL", detail)
    width = max(len(r["check"]) for r in res)
    for r in res:
        print(f"{r['level']:4s}  {r['check']:<{width}}  {r['detail']}")
    n_fail = sum(r["level"] == "FAIL" for r in res)
    print(f"\ndoctor: {n_fail} FAIL, {sum(r['level'] == 'WARN' for r in res)} WARN, "
          f"{sum(r['level'] == 'PASS' for r in res)} PASS")
    if cfg is not None:
        io.write_json(run_dir(cfg.name) / "doctor.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                          "environment": env, "checks": res})
    return 1 if n_fail else 0


# --------------------------------------------------------------------------- feedback
SUGGESTIONS = [
    ("letter mass at the forced position", "After \"Answer:\" the model puts little probability on the option "
                                            "letters. Check the prompt mode and chat_template_kwargs in the registry; "
                                            "compare c_forced with cq_forced (letters + names) in the stage summary: if "
                                            "names carry the mass, the letter readout is not valid for this model. "
                                            "Report it; do not change the readout for one model."),
    ("open-set coverage", "The model rarely answers the open question with a catalogue brand. Look at samples.parquet "
                          "(text column) for preambles or refusals before reading any N_eff."),
    ("answer coverage", "Even after \"Answer:\" the model rarely names an option (letter or name). Check the prompt "
                        "mode (registry is_chat / --prompt-mode), registry chat_template_kwargs, and run `format-probe` "
                        "on the model to see its raw answers."),
    ("agreement", "Greedy free answers disagree with the scored argmax. Check consistency.json per mode: immediate "
                  "answers test the free readout, deferred answers test the O5 imputation. Inspect gen_text for the "
                  "checked items; extend the parsers only if the outputs are genuine answers."),
    ("answer format", "The answer format (coverage or name share at the free position) changes with the evidence "
                      "level, so the ladder compares different readouts. Report it with answer_format.json; compare "
                      "the O3/O4 estimators in estimators_by_entity.parquet."),
    ("manipulation ratio", "One star of handicap moves the score by much less than a typical name effect. Report it "
                           "to the PI (threshold is provisional); do NOT enlarge the handicap or change cards for "
                           "this model alone."),
    ("rating evidence", "This model ignores the rating evidence (manipulation check). Report it; do NOT change the "
                        "card design or thresholds to make the check pass for this model alone."),
    ("selectivity", "Probes barely beat the control task. Check the layer range, the alpha grid and probe item "
                    "counts; consider the choice-frame probes; do not lower SELECTIVITY_WARN."),
    ("binding", "Probes do not prefer the read entity in two-brand texts. K may encode context, not the entity."),
    ("K capture", "K captures little of the real-minus-placebo difference: M4's delta_k would be ~0. This is a "
                   "method-level finding for the PI, not a bug to patch."),
    ("real-brand transfer", "Placebo-trained probes do not rank real brands by tier: K may not describe real-brand "
                            "knowledge. Method-level finding for the PI."),
    ("familiarity", "Familiarity is not represented in K; identity residuals of famous brands would be inflated. "
                    "Method-level finding for the PI (see DECISIONS.md)."),
    ("localize:", "A full transplant at the name's last token does not carry the whole name effect. Look at "
                  "post_name / band in localize/summary.md (where does the effect live?). Method-level finding for "
                  "the PI before M4; do not change the layer rule after seeing results."),
]


def _rules(rows, errors) -> list[str]:
    out = []
    if errors:
        out.append("Fix every error below first. Reproduce it locally with `python -m rtp.cli smoke` or a minimal "
                   "config, add a regression test in tests/, then re-run only the failed command on Colab (finished "
                   "stages whose code and inputs did not change are reused).")
    if any(r["state"] == "stale" or r["state"].startswith("partial+stale") for r in rows):
        out.append("Some stages are stale (other code/data/settings than the current checkout): re-run them with "
                   "--force (a cycle: `cycle run --redo-stale`, develop phase only), or check out the commit they "
                   "were built with.")
    if any(r["state"].startswith("partial (") for r in rows):
        out.append("Some stages are partial: re-run the same command to resume (finished shards are kept).")
    text = " ".join(w for r in rows for w in r["warnings"]).lower()
    for key, advice in SUGGESTIONS:
        if key.lower() in text:
            out.append(advice)
    out.append("Research integrity: never change methods, thresholds or data to make a warning disappear; record "
               "every design change in DECISIONS.md.")
    return out


SUMMARY_LINES, SUMMARY_CHARS = 90, 7000  # per stage summary in the feedback bundle
SCORECARD_LINES, SCORECARD_CHARS = 700, 90000
PROBE_LINES, PROBE_CHARS = 80, 9000  # per model: the token probe's tables and examples  # the newest scorecard goes in whole (6 models on e1: ~400 lines)


def _scorecards(cfg: Config, n: int = 3) -> list[Path]:
    """The newest cycle scorecards of the run (dev results only: a cycle that read the test split is skipped
    unless its scorecard is asked for directly)."""
    out = []
    for path in sorted((run_dir(cfg.name) / "cycles").glob("*/scorecard.md"), key=lambda p: p.stat().st_mtime):
        try:
            card = io.read_json(path.with_suffix(".json"))
        except Exception:  # noqa: BLE001
            continue
        if any("[test]" in k for k in list(card.get("models", {})) + list(card.get("pairs", {}))):
            continue
        out.append(path)
    return out[-n:]


def _newest_card(cfg: Config) -> dict[str, Any] | None:
    cards = _scorecards(cfg, n=1)
    return io.read_json(cards[-1].with_suffix(".json")) if cards else None


def _finding_actions(card: dict[str, Any] | None) -> list[str]:
    """One line that sends the reader to the findings of the newest scorecard (rtp/scorecard.py: fixed rules on
    the analysis numbers; the findings themselves are printed once, in the scorecard above)."""
    items = (card or {}).get("findings") or []
    count = {k: sum(f.get("kind") == k for f in items) for k in ("bug", "instrument", "research", "pending")}
    if not any(count.values()):
        return []
    return [f"Cycle `{card.get('cycle')}`: {count['bug']} bug(s), {count['instrument']} instrument finding(s), "
            f"{count['research']} result(s), {count['pending']} check(s) not evaluated yet (section Findings of the "
            f"newest scorecard above). Order: bugs first (fix with a test); then instrument findings (develop phase: "
            f"propose the change to the PI with the numbers, bump ladder.version, add a DECISIONS.md line, re-measure "
            f"with AFTER_INSTRUMENT_CHANGE; confirm phase: report, change nothing); results are reported with their "
            f"intervals and never tuned away."]


def _head(path: Path, n: int, max_chars: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines[:n])
    if len(text) > max_chars:
        text = text[:max_chars]
    cut = len(lines) > n or len("\n".join(lines[:n])) > max_chars
    return text + ("\n[... truncated; full file: " + str(path) + "]" if cut else "")


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


SESSION_HOURS = 12.0  # one cycle is at most 12 hours: older notebook commands belong to earlier sessions


def _this_session(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The notebook commands of the newest session: those within ``SESSION_HOURS`` of the last one. The history
    file lives on Drive and grows over weeks; a bundle that listed week-old commands of another config hid the
    ones that matter."""
    def stamp(entry: dict[str, Any]) -> float | None:
        try:
            return time.mktime(time.strptime(str(entry.get("time")), "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None

    times = [stamp(h) for h in history]
    known = [t for t in times if t is not None]
    if not known:
        return history
    newest = max(known)
    return [h for h, t in zip(history, times) if t is None or newest - t <= SESSION_HOURS * 3600.0]


def _unresolved_errors(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Errors not followed (in log order) by a successful run of the same stage and model."""
    out = []
    for i, err in enumerate(evs):
        if err.get("kind") != "error":
            continue
        fixed = any(e.get("kind") == "done" and e.get("data", {}).get("exit_code") == 0
                    and e.get("stage") == err.get("stage") and e.get("model") == err.get("model") for e in evs[i + 1:])
        if not fixed:
            out.append(err)
    return out


def _never_finished(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """'start' events with no later done/error from the same command (killed, OOM-killed or disconnected)."""
    out = []
    for i, e in enumerate(evs):
        if e.get("kind") != "start":
            continue
        argv = e.get("data", {}).get("argv")
        if argv and argv[0] in ("feedback", "status"):  # read-only; the one building this bundle is still running
            continue
        if not any(f.get("kind") in ("done", "error") and f.get("data", {}).get("argv") == argv for f in evs[i + 1:]):
            out.append(e)
    return out


DIGEST_BEGIN, DIGEST_END = "===== BEGIN RTP-AUDIT DIGEST", "===== END RTP-AUDIT DIGEST"
RECORD_SKIP = ("finished_with", "ladder_variants", "model_id", "split", "timing_complete", "warnings")


def _compact(value: Any) -> str | None:
    """A scalar or a small dict of scalars as short text; None for anything longer."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, str)):
        return str(value) if len(str(value)) <= 60 else None
    if isinstance(value, dict) and len(value) <= 12:
        parts = [(k, _compact(v)) for k, v in value.items()]
        return "{" + ", ".join(f"{k}: {v}" for k, v in parts) + "}" if all(v is not None for _, v in parts) else None
    return None


def _record_line(info: dict[str, Any]) -> str:
    """The completion record of a stage (its diagnostics and timing) on one line."""
    parts = [(k, _compact(v)) for k, v in info.items() if k not in RECORD_SKIP]
    return ", ".join(f"{k} {v}" for k, v in parts if v is not None)


def _probe_digest(res: dict[str, Any]) -> list[str]:
    def pct(x: Any) -> str:
        return "n/a" if x is None or x != x else f"{100.0 * float(x):.0f}%"

    def ends(d: dict[str, float]) -> str:
        return ", ".join(f"{k} {pct(v)}" for k, v in d.items()) if d else "n/a"

    lines = [f"### {res.get('model_id')} (tier {res.get('tier')}; 'top-1 is a letter' and the endings do not move with "
             f"scale; 'picked option' = the option whose letter has the higher probability)"]
    for cell, b in [("all", res.get("all") or {}), *(res.get("cells") or {}).items()]:
        lines.append(f"  {cell}: n={b.get('n_items')} letter mass {b.get('letter_mass', float('nan')):.3f}, top-1 is a "
                     f"letter {pct(b.get('top1_letter'))}; continuation ends on: {ends(b.get('continuation') or {})}; "
                     f"{b.get('n_top1_not_letter')} items with another top-1 end on: "
                     f"{ends(b.get('continuation_when_top1_not_letter') or {})}")
        lines.append("    other tokens (mean prob, top-1 share): " + "; ".join(
            f"{t['token']} {t['mean_prob']:.3f} ({pct(t['share_top1'])})" for t in (b.get("other_tokens") or [])[:6]))
    for e in (res.get("examples") or [])[:14]:
        first = ", ".join(f"{t} {p:.2f}" for t, p in e.get("first_tokens") or [])
        lines.append(f"  lowest letter mass, {e.get('cell')}: {e.get('letter_mass')} | {first} | {e.get('outcome')} | "
                     f"{e.get('text')}")
    return lines


def build_digest(cfg: Config, ts: str, rows: list[dict[str, Any]], errors: list[dict[str, Any]],
                 unfinished: list[dict[str, Any]], history: list[dict[str, Any]], doctor: dict[str, Any] | None,
                 warn_counts: dict[tuple, int], newest: dict[str, Any] | None) -> str:
    """The cycle on one screen of plain text, made to be copied out of the notebook and pasted to the coding
    agent: environment, what ran and what failed, the findings, every number of the scorecard's tables in
    compact lines, the stage diagnostics and the token probe. The complete record stays in the bundle file."""
    from rtp import scorecard
    from rtp.provenance import environment

    rdir = run_dir(cfg.name)
    env = environment()
    lines = [f"{DIGEST_BEGIN} ({cfg.name}, {ts}): copy everything down to the END line =====",
             f"ENV commit {str(env.get('git_commit'))[:10]}{' (DIRTY)' if env.get('git_dirty') else ''} | python "
             f"{env.get('python')} | torch {env.get('torch')} | transformers {env.get('transformers')} | "
             f"{env.get('device')}"]
    if doctor:
        bad = [c for c in doctor["checks"] if c["level"] != "PASS"]
        lines.append(f"DOCTOR ({doctor.get('time')}): " + ("; ".join(
            f"{c['level']} {c['check']}: {str(c['detail'])[:160]}" for c in bad) or "all checks PASS"))
    if history:
        lines.append("SESSION: " + "; ".join(
            f"{'ok' if h.get('exit_code') == 0 else 'FAILED exit ' + str(h.get('exit_code'))} "
            f"`{str(h.get('command')).replace('rtp.cli ', '')[:90]}` {float(h.get('seconds') or 0):.0f}s"
            for h in history))
    if unfinished:
        lines.append("NEVER FINISHED: " + "; ".join(f"`{str(e.get('message'))[:120]}`" for e in unfinished[-5:]))
    lines.append(f"ERRORS: {len(errors)} unresolved")
    for e in errors[-5:]:
        trace = [x for x in (e.get("data", {}).get("traceback") or "").strip().splitlines() if x.strip()][-4:]
        lines.append(f"- {e.get('stage')} / {e.get('model')}: {str(e.get('message'))[:240]}"
                     + (" :: " + " | ".join(x.strip()[:200] for x in trace) if trace else ""))
    top = sorted(warn_counts.items(), key=lambda kv: -kv[1])[:5]
    lines.append(f"WARNINGS: {len(warn_counts)} distinct" + ("" if not top else ": " + "; ".join(
        f"[{stage} / {model}] x{n} {msg[:140]}" for (stage, model, msg), n in top)))
    pt = [x.strip() for x in _tail(rdir / "pytest_last.txt", 40).splitlines()]
    verdict = [x for x in pt if " passed" in x or " failed" in x or " error" in x][-1:]
    failed = [x for x in pt if x.startswith(("FAILED", "ERROR"))][:6]
    lines.append("PYTEST: " + (verdict[0] if verdict else "(no run saved)") + "".join(f" | {x[:160]}" for x in failed))
    other = [f"{scorecard.short(r['model'])} / {r['stage']}: {r['state']}" for r in rows
             if not r.get("sealed") and r["stage"] != "build-data" and r["state"] not in ("done", "not started")]
    lines.append("STAGES NOT DONE: " + ("; ".join(other) or "none (every started stage is done)"))
    lines.append("")
    if newest:
        lines += scorecard.digest_lines(newest)
    else:
        lines.append("(no cycle scorecard yet: `cycle run` has not finished a job)")
    records = []
    for r in rows:
        p = Path(r["path"]) if r["path"] else None
        if p and not r.get("sealed") and r["stage"] != "build-data" and io.is_done(p):
            records.append(f"- {scorecard.short(r['model'])} / {r['stage']}: {_record_line(io.read_json(p / io.DONE_FILE))}")
    if records:
        lines += ["", "## Stage records (diagnostics and timing of each finished stage)"] + records
    probes = sorted((rdir / "ladder_probe").glob("*/ladder_probe.json"))  # dev only by construction
    if probes:
        lines += ["", "## Token probe (which tokens take the probability the option letters do not)"]
        for fp in probes:
            lines += _probe_digest(io.read_json(fp))
    from rtp.log import redact_text

    body = "\n".join(redact_text(x) for x in lines)
    return body + f"\n{DIGEST_END} ({len(body)} characters) =====\n"


def build_feedback(cfg: Config, tail: int = 60, unseal: bool = False) -> Path:
    from rtp.paths import work_dir
    from rtp.provenance import environment

    rdir = run_dir(cfg.name)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows = collect_status(cfg, unseal=unseal)
    evs = events.read(rdir / "events.jsonl")
    smoke_evs = []
    for f in sorted(rdir.parent.glob("*.smoke_events.jsonl")):
        smoke_evs.extend(events.read(f))
    all_errors = [e for e in evs + smoke_evs if e.get("kind") == "error"]
    errors = _unresolved_errors(evs) + _unresolved_errors(smoke_evs)
    unfinished = _never_finished(evs + smoke_evs)
    hist_path = work_dir() / "notebook_history.jsonl"
    history = _this_session(events.read(hist_path))[-40:]
    warn_counts: dict[tuple, int] = {}
    for e in evs:
        if e.get("kind") in ("warning", "log_error"):
            key = (e.get("stage"), e.get("model"), e.get("message", "")[:300])
            warn_counts[key] = warn_counts.get(key, 0) + 1
    doctor = io.read_json(rdir / "doctor.json") if (rdir / "doctor.json").exists() else None
    lines = [f"# Feedback bundle: {cfg.name} ({ts})", "",
             "> For the coding agent: read CLAUDE.md and docs/AGENT_LOOP.md first, then this file. Fix errors with a "
             "regression test each; treat method-level warnings as findings for the PI, not as bugs.", "",
             "## Environment (now)", "", "```json", json.dumps(environment(), indent=1), "```"]
    if not unseal:
        lines += ["", "_Test-split results are sealed and omitted (pass --unseal to include them)._"]
    if doctor:
        bad = [c for c in doctor["checks"] if c["level"] != "PASS"]
        lines += ["", f"Doctor at {doctor.get('time')} (non-PASS checks):", ""] + \
            [f"- {c['level']} {c['check']}: {c['detail']}" for c in bad] + (["- all checks PASS"] if not bad else [])
    if history:
        lines += ["", "## Notebook session (latest commands)", ""] + \
            [f"- {'✅' if h.get('exit_code') == 0 else '❌'} `{h.get('command')}` exit {h.get('exit_code')}, "
             f"{h.get('seconds')}s, {h.get('warnings')} warnings ({h.get('time')})" for h in history]
    if unfinished:
        lines += ["", "## Commands that never finished (killed, out of host memory, or disconnected?)", ""] + \
            [f"- {e.get('time')} `{e.get('message')}`" for e in unfinished[-20:]]
    cards = _scorecards(cfg)
    newest = _newest_card(cfg)
    if cards:
        lines += ["", "## Cycle scorecards", "",
                  "Each cycle is one budgeted GPU session. A scorecard starts with its FINDINGS (what the numbers "
                  "mean and what to do next, by fixed rules), then the tables: checks, the ruler, the brands on it, "
                  "the stage pairs, the ledger. The newest one is given in full; of earlier ones only the head. "
                  "Read these first.", ""]
        for path in cards[:-1]:
            lines += [f"### {path.parent.name} (earlier cycle; head)", "", _head(path, 45, 6000), ""]
        lines += [f"### {cards[-1].parent.name} (newest)", "", _head(cards[-1], SCORECARD_LINES, SCORECARD_CHARS), ""]
    lines += ["", "## Stage status", "", status_markdown(rows), "## Suggested next actions", ""]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(_rules(rows, errors) + _finding_actions(newest))]
    lines += ["", f"## Errors ({len(errors)} unresolved; {len(all_errors) - len(errors)} resolved by a later "
                  f"successful run and omitted)", ""]
    for e in errors[-20:]:
        lines += [f"### {e.get('time')} `{e.get('message', '')[:200]}`", "",
                  f"- stage: {e.get('stage')}; model: {e.get('model')}; argv: {e.get('data', {}).get('argv')}", "",
                  "```", (e.get("data", {}).get("traceback") or "").strip()[-4000:], "```", ""]
    lines += [f"## Warnings ({len(warn_counts)} distinct)", ""]
    for (stage, model, msg), n in sorted(warn_counts.items(), key=lambda kv: -kv[1])[:60]:
        lines.append(f"- [{stage} / {model}] x{n}: {msg}")
    lines += ["", "## Key results", ""]
    for r in rows:
        p = Path(r["path"]) if r["path"] else None
        if r.get("sealed"):
            continue
        if p and io.is_done(p) and r["stage"] != "build-data":
            info = io.read_json(p / io.DONE_FILE)
            keep = {k: v for k, v in info.items() if k not in ("finished_with", "ladder_variants")}
            lines.append(f"- **{r['model']} / {r['stage']}**: `{json.dumps(keep, default=str)[:900]}`")
    for fp in sorted((rdir / "format_probe").glob("*/format_probe.json")):  # dev only by construction
        res = io.read_json(fp)
        keep = {k: v for k, v in res.items() if not k.startswith("prompt_tail")}
        lines.append(f"- **{res.get('model_id')} / format-probe**: `{json.dumps(keep, default=str)[:900]}`")
    lines += ["", "## Stage summaries (dev; head of each summary.md)", "",
              "The first lines of every finished, unsealed stage's summary (gates, warnings, diagnostics); per-entity "
              "tables are cut. Test-split summaries stay sealed unless --unseal.", ""]
    for r in rows:
        p = Path(r["path"]) if r["path"] else None
        if r.get("sealed") or not p or r["stage"] == "build-data" or not io.is_done(p) or not (p / "summary.md").exists():
            continue
        lines += [f"### {r['model']} / {r['stage']}", "", "```", _head(p / "summary.md", SUMMARY_LINES, SUMMARY_CHARS),
                  "```", ""]
    for fp in sorted((rdir / "format_probe").glob("*/summary.md")):  # dev only by construction
        lines += [f"### {fp.parent.name} / format-probe", "", "```", _head(fp, SUMMARY_LINES, SUMMARY_CHARS), "```", ""]
    probes = sorted((rdir / "ladder_probe").glob("*/summary.md"))  # dev only by construction
    if probes:
        lines += ["## Token probe (which tokens take the mass the option letters do not; dev sample)", ""]
        for fp in probes:
            lines += [_head(fp, PROBE_LINES, PROBE_CHARS), ""]
    adir = rdir / "analysis"
    shown = set()  # models and pairs whose tables are already in the newest scorecard above
    if newest:
        shown = {model_slug(m) for m in newest.get("models", {})}
        shown |= {"__".join(model_slug(x) for x in k.split(" -> ")) for k in newest.get("pairs", {})}
    summaries = [fp for fp in sorted(adir.glob("*/summary.md")) + sorted(adir.glob("_pairs/*/summary.md"))
                 if fp.parent.name not in shown]
    if summaries:
        lines += ["", "## Analysis summaries of models and pairs that are not in the newest scorecard (dev)", ""]
        for fp in summaries:
            lines += [f"### {fp.parent.parent.name + '/' if fp.parent.parent.name == '_pairs' else ''}{fp.parent.name}",
                      "", "```", _head(fp, SUMMARY_LINES, SUMMARY_CHARS), "```", ""]
    lines += ["", "## Log tails of failed or partial stages", ""]
    for r in rows:
        if r["path"] and not r.get("sealed") and (r["state"].startswith("partial") or r["state"] == "stale"):
            lines += [f"### {r['model']} / {r['stage']} ({r['state']})", "", "```",
                      _tail(Path(r["path"]) / "log.txt", tail), "```", ""]
    pt = rdir / "pytest_last.txt"
    lines += ["## Last pytest run", "", "```", _tail(pt, 40) or "(no pytest output saved; run nb.pytest())", "```"]
    from rtp.log import redact_text

    lines = [redact_text(x) for x in lines]  # second layer: log tails and pytest output
    fdir = rdir / "feedback"
    md_path = fdir / f"feedback_{ts}.md"
    io.atomic_write_text(md_path, "\n".join(lines) + "\n")
    io.write_json(fdir / f"feedback_{ts}.json", {"config": cfg.name, "time": ts, "status": rows, "errors": errors,
                                                  "warnings": [{"stage": k[0], "model": k[1], "message": k[2],
                                                                "count": n} for k, n in warn_counts.items()],
                                                  "doctor": doctor})
    digest_path = fdir / f"digest_{ts}.md"  # what the notebook prints for copy and paste
    io.atomic_write_text(digest_path, build_digest(cfg, ts, rows, errors, unfinished, history, doctor, warn_counts,
                                                   newest))
    print(f"feedback bundle: {md_path}")
    print(f"digest: {digest_path}")
    return md_path
