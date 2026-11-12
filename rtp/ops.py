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
              "extract": ("extract", "dev"), "extract_test": ("extract", "test"), "probe": ("probe", "dev")}
PIPELINE = ("measure", "open_ended", "extract", "probe")


# --------------------------------------------------------------------------- status
def _warnings_of(kind: str, info: dict[str, Any]) -> list[str]:
    w = []
    if info.get("consistency_warning"):
        c = info.get("consistency", {})
        w.append(f"letter/generation agreement {c.get('agreement', float('nan')):.0%} < 90%")
    if info.get("letter_mass_warning"):
        w.append(f"letter mass {info.get('mean_letter_mass')} (model not answering with a letter)")
    if info.get("name_mass_warning"):
        w.append(f"name-answer mass gap {info.get('name_mass_max_gap')}")
    if info.get("responds_to_ratings") is False:
        w.append("does not respond to rating evidence")
    w.extend(info.get("warnings", []) if kind == "probe" else [])
    return w


def collect_status(cfg: Config) -> list[dict[str, Any]]:
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
                 "path": str(ddir)})
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
        info = io.read_json(sdir / io.DONE_FILE) if io.is_done(sdir) else {}
        n_shards = len(list(sdir.glob("shards*/*.done"))) + len(list(sdir.glob("*/shard_*.done")))
        if io.is_done(sdir):
            state = "done" if meta.get("stage_hash") == expected else "stale"
        else:
            state = f"partial ({n_shards} shards done)"
        rows.append({"model": model_id, "stage": sdir.name, "state": state, "warnings": _warnings_of(kind, info),
                     "elapsed_s": info.get("elapsed_s"), "path": str(sdir)})
    for m in cfg.models:
        for st in PIPELINE:
            if (m, st) not in seen:
                rows.append({"model": m, "stage": st, "state": "not started", "warnings": [], "elapsed_s": None,
                             "path": ""})
    return rows


def status_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | stage | state | warnings | elapsed |", "|---|---|---|---|---|"]
    for r in rows:
        el = "" if r["elapsed_s"] is None else f"{r['elapsed_s']:.0f}s"
        warn = "; ".join(w if len(w) <= 90 else w[:87] + "..." for w in r["warnings"]) or "—"
        lines.append(f"| {r['model']} | {r['stage']} | {r['state']} | {warn} | {el} |")
    return "\n".join(lines) + "\n"


def run_status(cfg: Config) -> list[dict[str, Any]]:
    rows = collect_status(cfg)
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
                    _check(res, f"hub:{m}", "PASS", "accessible" + (" (gated: licence accepted)" if spec and spec.gated
                                                                   else ""))
                except Exception as exc:  # noqa: BLE001
                    _check(res, f"hub:{m}", "FAIL", str(exc).splitlines()[0][:200])
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
    ("letter mass", "The model does not answer with a letter at the scored position. Check the prompt mode "
                    "(registry is_chat / --prompt-mode), registry chat_template_kwargs, and ANSWER_PREFILL in "
                    "rtp/measure/choice.py on one rendered prompt."),
    ("agreement", "Greedy generations disagree with the letter argmax. Inspect gen_text in scores.parquet for the "
                  "checked items; extend parse_letter only if the outputs are genuine answers."),
    ("name-answer", "The target's name-first-token mass differs between real and placebo versions. Compare "
                    "entity_evidence_credit.parquet with entity_evidence.parquet before interpreting name effects."),
    ("rating evidence", "This model ignores the rating evidence (manipulation check). Report it; do NOT change the "
                        "card design or thresholds to make the check pass for this model alone."),
    ("selectivity", "Probes barely beat the control task. Check the layer range, the alpha grid and probe item "
                    "counts; consider the choice-frame probes; do not lower SELECTIVITY_WARN."),
    ("binding", "Probes do not prefer the read entity in two-brand texts. K may encode context, not the entity."),
    ("K captures", "K captures little of the real-minus-placebo difference: M4's delta_k would be ~0. This is a "
                   "method-level finding for the PI, not a bug to patch."),
    ("real-brand transfer", "Placebo-trained probes do not rank real brands by tier: K may not describe real-brand "
                            "knowledge. Method-level finding for the PI."),
    ("familiarity", "Familiarity is not represented in K; identity residuals of famous brands would be inflated. "
                    "Method-level finding for the PI (see DECISIONS.md)."),
]


def _rules(rows, errors) -> list[str]:
    out = []
    if errors:
        out.append("Fix every error below first. Reproduce it locally with `python -m rtp.cli smoke` or a minimal "
                   "config, add a regression test in tests/, then re-run only the failed command on Colab (finished "
                   "stages whose code and inputs did not change are reused).")
    if any(r["state"] == "stale" for r in rows):
        out.append("Some stages are stale (other code/data/settings than the current checkout): re-run them with "
                   "--force, or check out the commit they were built with.")
    if any(r["state"].startswith("partial") for r in rows):
        out.append("Some stages are partial: re-run the same command to resume (finished shards are kept).")
    text = " ".join(w for r in rows for w in r["warnings"]).lower()
    for key, advice in SUGGESTIONS:
        if key.lower() in text:
            out.append(advice)
    out.append("Research integrity: never change methods, thresholds or data to make a warning disappear; record "
               "every design change in DECISIONS.md.")
    return out


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def build_feedback(cfg: Config, tail: int = 60) -> Path:
    from rtp.provenance import environment

    rdir = run_dir(cfg.name)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows = collect_status(cfg)
    evs = events.read(rdir / "events.jsonl")
    errors = [e for e in evs if e.get("kind") == "error"]
    warn_counts: dict[tuple, int] = {}
    for e in evs:
        if e.get("kind") in ("warning", "log_error"):
            key = (e.get("stage"), e.get("model"), e.get("message", "")[:300])
            warn_counts[key] = warn_counts.get(key, 0) + 1
    doctor = io.read_json(rdir / "doctor.json") if (rdir / "doctor.json").exists() else None
    lines = [f"# Feedback bundle: {cfg.name} ({ts})", "",
             "> For the coding agent: read CLAUDE.md and docs/AGENT_LOOP.md first, then this file. Fix errors with a "
             "regression test each; treat method-level warnings as findings for the PI, not as bugs.", "",
             "## Environment", "", "```json", json.dumps(doctor["environment"] if doctor else environment(), indent=1),
             "```"]
    if doctor:
        bad = [c for c in doctor["checks"] if c["level"] != "PASS"]
        lines += ["", "Doctor (non-PASS checks):", ""] + [f"- {c['level']} {c['check']}: {c['detail']}" for c in bad] \
            + (["- all checks PASS"] if not bad else [])
    lines += ["", "## Stage status", "", status_markdown(rows), "## Suggested next actions", ""]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(_rules(rows, errors))]
    lines += ["", f"## Errors ({len(errors)})", ""]
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
        if p and io.is_done(p) and r["stage"] != "build-data":
            info = io.read_json(p / io.DONE_FILE)
            keep = {k: v for k, v in info.items() if k not in ("finished_with", "ladder_variants")}
            lines.append(f"- **{r['model']} / {r['stage']}**: `{json.dumps(keep, default=str)[:900]}`")
    lines += ["", "## Log tails of failed or partial stages", ""]
    for r in rows:
        if r["path"] and (r["state"].startswith("partial") or r["state"] == "stale"):
            lines += [f"### {r['model']} / {r['stage']} ({r['state']})", "", "```",
                      _tail(Path(r["path"]) / "log.txt", tail), "```", ""]
    pt = rdir / "pytest_last.txt"
    lines += ["## Last pytest run", "", "```", _tail(pt, 40) or "(no pytest output saved; run nb.pytest())", "```"]
    fdir = rdir / "feedback"
    md_path = fdir / f"feedback_{ts}.md"
    io.atomic_write_text(md_path, "\n".join(lines) + "\n")
    io.write_json(fdir / f"feedback_{ts}.json", {"config": cfg.name, "time": ts, "status": rows, "errors": errors,
                                                  "warnings": [{"stage": k[0], "model": k[1], "message": k[2],
                                                                "count": n} for k, n in warn_counts.items()],
                                                  "doctor": doctor})
    print(f"feedback bundle: {md_path}")
    return md_path
