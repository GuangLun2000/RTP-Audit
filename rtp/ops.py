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
from rtp.paths import data_dir, run_dir

log = logging.getLogger(__name__)

STAGE_DIRS = {"measure": ("measure", "dev"), "measure_test": ("measure", "test"),
              "open_ended": ("open_ended", "dev"), "open_ended_test": ("open_ended", "test"),
              "extract": ("extract", "dev"), "extract_test": ("extract", "test"), "probe": ("probe", "dev"),
              "localize": ("localize", "dev")}
PIPELINE = ("measure", "open_ended", "extract", "probe", "localize")


# --------------------------------------------------------------------------- status
def _warnings_of(kind: str, info: dict[str, Any]) -> list[str]:
    w = []
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
        kind, split = STAGE_DIRS[sdir.name]
        model_id = meta.get("model_id", sdir.parent.name)
        seen.add((model_id, sdir.name))
        try:
            expected = stages.stage_hash(cfg, kind, model_id, split, meta.get("prompt_mode", "auto"))
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
        for st in PIPELINE:
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
            _check(res, label, "PASS" if free > 30 else "WARN", f"{free:.0f} GiB free at {p}")
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
            ok = io.stored_stage_hash(ddir) == cfg.data_hash
            _check(res, "data", "PASS" if ok else "WARN", "built and current" if ok else
                   "built from other seeds/code: run build-data --force")
        else:
            _check(res, "data", "WARN", "not built yet (run build-data)")
        if check_hub:
            for m in cfg.models:
                spec = registry.get_model_spec(m)
                try:
                    registry.validate_repo_id(m, "model")
                    if spec is not None and spec.gated:  # metadata is public; check that FILES are accessible
                        from huggingface_hub import HfApi

                        HfApi().auth_check(m)
                    _check(res, f"hub:{m}", "PASS", "accessible" + (" (gated: token and licence OK)"
                                                                   if spec and spec.gated else ""))
                except Exception as exc:  # noqa: BLE001
                    detail = registry.gated_message(m) if spec and spec.gated else str(exc).splitlines()[0][:200]
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
                   "--force, or check out the commit they were built with.")
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
        if not any(f.get("kind") in ("done", "error") and f.get("data", {}).get("argv") == argv for f in evs[i + 1:]):
            out.append(e)
    return out


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
    history = events.read(hist_path)[-40:]
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
    lines += ["", "## Stage status", "", status_markdown(rows), "## Suggested next actions", ""]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(_rules(rows, errors))]
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
    print(f"feedback bundle: {md_path}")
    return md_path
