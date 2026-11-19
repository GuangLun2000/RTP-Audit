"""Experiment cycles: one short, budgeted GPU session with a question, a job list and a scorecard.

A cycle file (``configs/cycles/<name>.yaml``) names the run config, the question the session answers, its
phase (``develop`` or ``confirm``), a list of jobs (stage x tier x models) in priority order, and the model
pairs to compare. ``cycle plan`` prices every job with the cost model (rtp.cost) and refuses a cycle whose
remaining work exceeds its budget; no cycle may budget more than ``MAX_BUDGET_HOURS``. ``cycle run`` executes
the jobs in order, each stage in its own process, and after every job runs the CPU analysis and rewrites
the scorecard, so an interrupted session still leaves usable results. A job that no longer fits in the
remaining budget is skipped ("deferred"); the next run of the same cycle continues with it.

A cycle never mixes results: a job whose finished or half-finished stage was built by other code, items or
settings ("stale"), or half-finished in another environment, blocks the cycle before any GPU time is spent.
``--redo-stale`` (develop phase only) re-measures such jobs from scratch.

The scorecard (``runs/<config>/cycles/<name>/scorecard.{md,json}``) holds the jobs with predicted and actual
seconds, the instrument's validity checks per model, the stage-pair comparisons, and what changed against
the previous cycle. It is what the feedback bundle shows first.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rtp import cost, io
from rtp.config import REPO_ROOT, Config, ConfigError, load_config
from rtp.paths import run_dir, stage_dir

log = logging.getLogger(__name__)

MAX_BUDGET_HOURS = 12.0  # hard cap of one cycle (PI rule: one session on one G4 GPU, short enough to iterate)
PHASES = ("develop", "confirm")
JOB_STAGES = ("ladder", "open-set")
SPLITS = ("dev", "test")


class CycleError(RuntimeError):
    pass


class BudgetExceeded(CycleError):
    pass


@dataclass
class Job:
    stage: str  # "ladder" | "open-set"
    model: str
    tier: str | None = None
    split: str = "dev"

    @property
    def label(self) -> str:
        t = f"-{self.tier}" if self.tier else ""
        s = "" if self.split == "dev" else " [test]"
        return f"{self.stage}{t} {self.model}{s}"

    @property
    def stage_name(self) -> str:
        if self.stage == "ladder":
            from rtp.data.build import ladder_file  # validates the tier

            ladder_file(self.tier or "")
            return f"ladder_{self.tier}" if self.split == "dev" else f"ladder_{self.tier}_test"
        return "open_set" if self.split == "dev" else "open_set_test"

    @property
    def kind(self) -> str:
        return "ladder" if self.stage == "ladder" else "open_set"


@dataclass
class CycleSpec:
    name: str
    config: str
    phase: str
    question: str
    budget_hours: float
    jobs: list[Job]
    pairs: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    cut_order: list[str] = field(default_factory=list)
    source_path: str | None = None


def load_cycle(path: str | Path) -> CycleSpec:
    p = Path(path)
    if not p.exists() and (REPO_ROOT / p).exists():
        p = REPO_ROOT / p
    if not p.exists():
        raise ConfigError(f"cycle file not found: {path}")
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a cycle file must be a mapping")
    allowed = {"name", "config", "phase", "question", "budget_hours", "jobs", "pairs", "decisions", "cut_order"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    for key in ("name", "config", "phase", "question", "jobs"):
        if not raw.get(key):
            raise ConfigError(f"{path}: '{key}' is required")
    if raw["phase"] not in PHASES:
        raise ConfigError(f"{path}: phase must be one of {PHASES}")
    budget = float(raw.get("budget_hours", MAX_BUDGET_HOURS))
    if not 0 < budget <= MAX_BUDGET_HOURS:
        raise ConfigError(f"{path}: budget_hours must be in (0, {MAX_BUDGET_HOURS:g}]: no cycle runs longer than "
                          f"{MAX_BUDGET_HOURS:g} h; split the work into two cycles")
    jobs: list[Job] = []
    for i, j in enumerate(raw["jobs"]):
        if not isinstance(j, dict) or set(j) - {"stage", "tier", "models", "split"}:
            raise ConfigError(f"{path}: job {i} must be a mapping with keys stage, tier, models, split")
        if j.get("stage") not in JOB_STAGES:
            raise ConfigError(f"{path}: job {i}: stage must be one of {JOB_STAGES}")
        if (j["stage"] == "ladder") != bool(j.get("tier")):
            raise ConfigError(f"{path}: job {i}: 'tier' is required for ladder jobs and only for them")
        if j.get("split", "dev") not in SPLITS:
            raise ConfigError(f"{path}: job {i}: split must be one of {SPLITS}")
        models = j.get("models") or []
        if not models or not all(isinstance(m, str) and m for m in models):
            raise ConfigError(f"{path}: job {i}: 'models' must be a non-empty list of model ids")
        jobs.extend(Job(stage=j["stage"], model=m, tier=j.get("tier"), split=j.get("split", "dev")) for m in models)
    for j in jobs:
        j.stage_name  # noqa: B018 - validates the tier
    pairs = []
    for pr in raw.get("pairs") or []:
        if not (isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, str) for x in pr)):
            raise ConfigError(f"{path}: each pair must be a list of two model ids")
        pairs.append((pr[0], pr[1]))
    return CycleSpec(name=str(raw["name"]), config=str(raw["config"]), phase=raw["phase"],
                     question=" ".join(str(raw["question"]).split()), budget_hours=budget, jobs=jobs, pairs=pairs,
                     decisions=[str(x) for x in raw.get("decisions") or []],
                     cut_order=[str(x) for x in raw.get("cut_order") or []], source_path=str(p))


def cycle_dir(cfg: Config, spec: CycleSpec) -> Path:
    return run_dir(cfg.name) / "cycles" / spec.name


# --------------------------------------------------------------------------- state and plan
NEEDS_FORCE = ("stale", "partial-env")  # states that only a re-measurement from scratch resolves


def job_state(cfg: Config, job: Job) -> str:
    """done | stale | partial | partial-env | todo, judged against the CURRENT code, data and settings.
    ``partial-env``: half-finished under another torch / transformers / tokenizers version or device; the
    stage refuses to resume (its shards would mix numerics) and must be restarted."""
    from rtp import stages
    from rtp.provenance import RESUME_CRITICAL, environment

    sdir = stage_dir(cfg.name, job.model, job.stage_name)
    if not (sdir / io.STAGE_META).exists():
        return "todo"
    meta = io.read_json(sdir / io.STAGE_META)
    try:
        expected = stages.stage_hash(cfg, job.kind, job.model, job.split, meta.get("prompt_mode", "auto"),
                                     tier=job.tier)
    except Exception:  # noqa: BLE001 - e.g. data not built
        expected = None
    current = io.stored_stage_hash(sdir) == expected
    if io.is_done(sdir):
        return "done" if current else "stale"
    if not current:
        return "stale"
    if stages.finished_shards(sdir):
        created, now = meta.get("created_with", {}), environment()
        if any(created.get(k) != now.get(k) for k in RESUME_CRITICAL):
            return "partial-env"
    return "partial"


def check_phase(cfg: Config, spec: CycleSpec) -> None:
    from rtp import phase

    confirm = sorted({j.model for j in spec.jobs if phase.is_confirm_model(cfg, j.model)})
    if spec.phase == "develop" and confirm:
        raise CycleError(f"cycle '{spec.name}' is a develop cycle but lists confirmation models {confirm}")
    if spec.phase == "develop" and any(j.split == "test" for j in spec.jobs):
        raise CycleError(f"cycle '{spec.name}' is a develop cycle but reads the sealed test split")
    if spec.phase == "confirm" and not phase.is_frozen(cfg):
        raise CycleError(f"cycle '{spec.name}' is a confirm cycle but the instrument of '{cfg.name}' is not frozen")
    for split in {j.split for j in spec.jobs}:
        phase.check_test_split_allowed(cfg, split)


def plan(cfg: Config, spec: CycleSpec) -> dict[str, Any]:
    """Every job with its state and predicted seconds; the remaining total against the budget."""
    calib = cost.calibration(cfg)
    seen: set[str] = set()
    rows = []
    for job in spec.jobs:
        state = job_state(cfg, job)
        est = cost.estimate(cfg, job.kind, job.model, job.tier, job.split, calib, seen)
        from rtp import registry

        seen.add(registry.base_model_id(job.model))
        rows.append({"job": job.label, "stage": job.stage, "tier": job.tier, "model": job.model, "split": job.split,
                     "state": state, "predicted_s": 0.0 if state == "done" else round(est["seconds"], 1),
                     "n_items": est.get("n_items"), "n_tokens": est.get("n_tokens"),
                     "tokens_measured": est.get("tokens_measured")})
    n_analysis = len({j.model for j in spec.jobs}) + len(spec.pairs)
    total = sum(r["predicted_s"] for r in rows) + n_analysis * cost.ANALYZE_S
    return {"cycle": spec.name, "phase": spec.phase, "budget_s": spec.budget_hours * 3600.0, "total_s": round(total, 1),
            "calibration": {"k": round(calib["k"], 3), "n_stages": calib["n_stages"],
                            "load_again_s": round(calib["load_again_s"], 1)}, "jobs": rows,
            "fits": total <= spec.budget_hours * 3600.0}


def fmt_h(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def plan_markdown(p: dict[str, Any]) -> str:
    lines = [f"Cycle `{p['cycle']}` ({p['phase']}): predicted {fmt_h(p['total_s'])} of GPU time for the remaining "
             f"jobs, budget {fmt_h(p['budget_s'])}: {'fits' if p['fits'] else 'DOES NOT FIT'}.",
             f"Cost model: prior x {p['calibration']['k']} (calibrated on {p['calibration']['n_stages']} finished "
             f"stage(s) of this run; 0 = the pilot prior, which errs on the long side).", "",
             "| # | job | state | predicted | items | prompt tokens |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(p["jobs"], 1):
        tok = "" if r["n_tokens"] is None else f"{r['n_tokens']:,}" + ("" if r["tokens_measured"] else " (est.)")
        lines.append(f"| {i} | {r['job']} | {r['state']} | {fmt_h(r['predicted_s']) if r['state'] != 'done' else '—'} | "
                     f"{'' if r['n_items'] is None else format(r['n_items'], ',')} | {tok} |")
    return "\n".join(lines) + "\n"


def run_plan(cfg: Config, spec: CycleSpec, redo_stale: bool = False) -> dict[str, Any]:
    """Print the priced plan; raise when the cycle does not fit its budget, or when a job could only run by
    discarding an existing stage and ``redo_stale`` was not given."""
    check_phase(cfg, spec)
    p = plan(cfg, spec)
    cdir = cycle_dir(cfg, spec)
    io.write_json(cdir / "plan.json", p)
    print(plan_markdown(p))
    if not p["fits"]:
        raise BudgetExceeded(
            f"cycle '{spec.name}' needs about {fmt_h(p['total_s'])}, more than its budget of "
            f"{fmt_h(p['budget_s'])}. Remove jobs (cut_order: {spec.cut_order or 'not given'}) or split the cycle.")
    blocked = [r["job"] for r in p["jobs"] if r["state"] in NEEDS_FORCE]
    if blocked and spec.phase == "confirm":
        raise CycleError(f"confirm cycle '{spec.name}' has stale or interrupted stages {blocked}: after the freeze "
                         f"the instrument must not change. Check out the frozen commit and environment; a confirm "
                         f"cycle never re-measures.")
    if blocked and not redo_stale:
        raise CycleError(
            f"cycle '{spec.name}': {len(blocked)} job(s) have a stage built by other code, items or settings "
            f"(stale), or interrupted in another environment: {blocked}. Their results cannot be mixed with new "
            f"ones. Re-run with --redo-stale to measure them again from scratch (the predicted time above "
            f"already includes them); nothing was run.")
    return p


# --------------------------------------------------------------------------- run
def code_state() -> dict[str, str]:
    from rtp.provenance import environment, stage_code_digest
    from rtp.stages import STAGE_MODULES

    env = environment()
    return {"git_commit": env.get("git_commit") or "", "ladder": stage_code_digest(STAGE_MODULES["ladder"]),
            "open_set": stage_code_digest(STAGE_MODULES["open_set"])}


def _stage_argv(cfg_path: str, job: Job, overrides: list[str], unseal: bool) -> list[str]:
    argv = [sys.executable, "-m", "rtp.cli", *overrides, job.stage, "--config", cfg_path, "--model", job.model]
    if job.tier:
        argv += ["--tier", job.tier]
    if job.split != "dev":
        argv += ["--split", job.split]
        if unseal:
            argv.append("--unseal")
    return argv


def stored_sets(cfg_name: str, spec: CycleSpec) -> list[str]:
    """The config overrides the cycle was started with (``report`` and a resumed ``run`` reuse them)."""
    path = run_dir(cfg_name) / "cycles" / spec.name / "run_meta.json"
    return list(io.read_json(path).get("sets") or []) if path.exists() else []


def cycle_config(spec: CycleSpec, sets: list[str] | None) -> Config:
    """The run config of a cycle with its overrides. Without ``sets`` the overrides recorded when the cycle
    started are reused; different overrides for a started cycle are refused (its stages would look stale)."""
    from rtp.config import apply_overrides

    base = load_config(spec.config)
    recorded = stored_sets(base.name, spec)
    if sets and recorded and list(sets) != recorded:
        raise CycleError(f"cycle '{spec.name}' was started with --set {recorded}; got {list(sets)}. Use the same "
                         f"overrides, or start over with `cycle run --restart`.")
    return apply_overrides(base, list(sets) if sets else recorded)


def run_cycle(spec: CycleSpec, restart: bool = False, unseal: bool = False, sets: list[str] | None = None,
              budget_hours: float | None = None, redo_stale: bool = False, env: dict[str, str] | None = None) -> int:
    """Run the cycle's jobs within its budget. Returns 0 when every job is done, 7 when some were deferred
    or failed (the scorecard says which). ``sets``: ``section.key=value`` overrides of the run config, applied
    here and passed on to every stage. ``env``: extra environment of the stage processes."""
    from rtp import analyze
    from rtp.config import apply_overrides
    from rtp.data.build import build_data
    from rtp.notebook import gpu_overrides, stream

    cfg = apply_overrides(load_config(spec.config), sets) if restart else cycle_config(spec, sets)
    if any(j.split == "test" for j in spec.jobs) and not unseal:
        from rtp.data.splits import SealedSplitError

        raise SealedSplitError(f"cycle '{spec.name}' reads the sealed test split: pass --unseal deliberately")
    build_data(cfg)
    cdir = cycle_dir(cfg, spec)
    meta_path = cdir / "run_meta.json"
    now_code = code_state()
    first = restart or not meta_path.exists()
    if not first:  # a resumed cycle: the measurement code must be the one it started with
        old = io.read_json(meta_path).get("code", {})
        changed = sorted(k for k in ("ladder", "open_set") if old.get(k) not in (None, now_code[k]))
        touched = [j.label for j in spec.jobs if j.kind in changed and job_state(cfg, j) != "todo"]
        if touched:
            raise CycleError(
                f"the measurement code ({', '.join(changed)}) changed since cycle '{spec.name}' started at commit "
                f"{str(old.get('git_commit', '?'))[:10]}: {len(touched)} of its jobs already have results from the "
                f"old code. Check out that commit to finish the cycle, or start over with --restart --redo-stale "
                f"(re-measures them with the new code).")
    p = run_plan(cfg, spec, redo_stale=redo_stale)
    budget_s = min(spec.budget_hours, budget_hours or spec.budget_hours) * 3600.0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session = {"time": stamp, "git_commit": now_code["git_commit"], "budget_hours": budget_s / 3600.0}
    if first:  # the code a cycle started with is recorded once, never overwritten by a later session
        meta = {"code": now_code, "started": stamp, "cycle_file": spec.source_path, "sets": list(sets or []),
                "sessions": [session]}
    else:
        meta = io.read_json(meta_path)
        meta.setdefault("sessions", []).append(session)
    io.write_json(meta_path, meta)
    earlier = {}
    if not first and (cdir / "ledger.json").exists():  # actual times of jobs finished in earlier sessions
        earlier = {e["job"]: e for e in io.read_json(cdir / "ledger.json")}
    t_start = time.time()
    deadline = t_start + budget_s
    ovr = gpu_overrides() + [x for s in sets or meta.get("sets") or [] for x in ("--set", s)]  # the user's last
    child_env = {**(env or {}), "RTP_DEADLINE": repr(deadline)}
    ledger: list[dict[str, Any]] = []
    for row, job in zip(p["jobs"], spec.jobs):
        entry = {"job": job.label, "predicted_s": row["predicted_s"], "actual_s": None, "state": row["state"]}
        ledger.append(entry)
        if row["state"] == "done":
            old_entry = earlier.get(job.label)
            if old_entry and old_entry.get("state") == "done":
                entry.update(predicted_s=old_entry.get("predicted_s", 0.0), actual_s=old_entry.get("actual_s"))
            continue
        left = deadline - time.time()
        if row["predicted_s"] > left:
            entry["state"] = "deferred (does not fit in the remaining budget)"
            log.warning("%s: predicted %s but only %s left in the budget; deferred", job.label,
                        fmt_h(row["predicted_s"]), fmt_h(left))
            continue
        argv = _stage_argv(spec.config, job, ovr, unseal)
        if row["state"] in NEEDS_FORCE:  # only reachable with --redo-stale (run_plan refuses otherwise)
            argv.append("--force")
            entry["redone"] = row["state"]
        t0 = time.time()
        rc = stream(argv, env=child_env, label="rtp.cli " + job.label)
        entry["actual_s"] = round(time.time() - t0, 1)
        entry["state"] = "done" if rc == 0 else ("deferred (budget reached mid-stage; resumable)" if rc == 6
                                                 else f"failed (exit {rc})")
        _refresh(cfg, spec, ledger, analyze, t_start, final=False, unseal=unseal)  # the scorecard grows as jobs end
    card = _refresh(cfg, spec, ledger, analyze, t_start, final=True, unseal=unseal)
    print(f"\nscorecard: {cdir / 'scorecard.md'}")
    return 0 if all(e["state"] == "done" for e in card["jobs"]) else 7


def _refresh(cfg: Config, spec: CycleSpec, ledger: list[dict[str, Any]], analyze: Any, t_start: float,
             final: bool, unseal: bool = False, force: bool = False) -> dict[str, Any]:
    """Run the CPU analysis for every model with finished ladder stages and rewrite the scorecard. The
    analysis is cached by its inputs and code, so calling this after every job is cheap. The test split is
    analysed only with ``unseal``."""
    cdir = cycle_dir(cfg, spec)
    io.write_json(cdir / "ledger.json", ledger)
    models: dict[str, Any] = {}
    pairs: dict[str, Any] = {}
    splits = sorted({j.split for j in spec.jobs})
    for split in splits:
        if split == "test" and not unseal:
            continue
        for m in dict.fromkeys(j.model for j in spec.jobs if j.stage == "ladder" and j.split == split):
            try:
                res = analyze.analyze_model(cfg, m, split=split, unseal=unseal, force=force)
            except Exception as exc:  # noqa: BLE001 - analysis problems must not stop a GPU session
                log.error("analysis of %s (%s) failed: %s: %s", m, split, type(exc).__name__, exc)
                res = {"error": f"{type(exc).__name__}: {exc}"}
            if res is not None:
                models[m if split == "dev" else f"{m} [test]"] = res
        for a, b in spec.pairs:
            try:
                res = analyze.compare(cfg, a, b, split=split, unseal=unseal, force=force)
            except Exception as exc:  # noqa: BLE001
                log.error("comparison %s -> %s (%s) failed: %s: %s", a, b, split, type(exc).__name__, exc)
                res = {"error": f"{type(exc).__name__}: {exc}"}
            if res is not None:
                pairs[f"{a} -> {b}" + ("" if split == "dev" else " [test]")] = res
    card = {"cycle": spec.name, "config": cfg.name, "phase": spec.phase, "question": spec.question,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "final": final,
            "instrument": {"ladder_version": cfg.ladder.version, "data_hash": cfg.data_hash, **code_state()},
            "elapsed_s": round(time.time() - t_start, 1), "budget_s": spec.budget_hours * 3600.0,
            "jobs": ledger, "models": models, "pairs": pairs, "decisions": spec.decisions,
            "cut_order": spec.cut_order}
    prev = previous_scorecard(cfg, spec)
    card["changes"] = diff_cards(prev, card) if prev else []
    card["compared_with"] = prev.get("cycle") if prev else None
    io.write_json(cdir / "scorecard.json", card)
    io.atomic_write_text(cdir / "scorecard.md", scorecard_markdown(card))
    return card


# --------------------------------------------------------------------------- scorecard
def previous_scorecard(cfg: Config, spec: CycleSpec) -> dict[str, Any] | None:
    """The most recently written scorecard of ANOTHER cycle of this run."""
    best: tuple[float, dict[str, Any]] | None = None
    for path in (run_dir(cfg.name) / "cycles").glob("*/scorecard.json"):
        if path.parent.name == spec.name:
            continue
        try:
            card = io.read_json(path)
        except Exception:  # noqa: BLE001
            continue
        if best is None or path.stat().st_mtime > best[0]:
            best = (path.stat().st_mtime, card)
    return best[1] if best else None


def _flatten(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}/"))
    elif isinstance(obj, bool):
        out[prefix.rstrip("/")] = float(obj)
    elif isinstance(obj, (int, float)) and obj == obj:
        out[prefix.rstrip("/")] = float(obj)
    return out


def diff_cards(old: dict[str, Any], new: dict[str, Any], rel: float = 0.05, top: int = 40) -> list[dict[str, Any]]:
    """Headline numbers (model checks and pair summaries) that moved by more than ``rel`` since ``old``."""
    a = _flatten({"models": old.get("models", {}), "pairs": old.get("pairs", {})})
    b = _flatten({"models": new.get("models", {}), "pairs": new.get("pairs", {})})
    rows = []
    for k in sorted(set(a) & set(b)):
        if abs(b[k] - a[k]) > rel * max(abs(a[k]), abs(b[k]), 1e-9):
            rows.append({"key": k, "before": a[k], "now": b[k]})
    rows.sort(key=lambda r: -abs(r["now"] - r["before"]) / max(abs(r["before"]), abs(r["now"]), 1e-9))
    return rows[:top]


def _num(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def _bullets(obj: Any, indent: int = 0, max_items: int = 60) -> list[str]:
    """A nested dict as an indented bullet list (the scorecard stays readable without knowing its schema)."""
    lines: list[str] = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                lines.append(f"{pad}- ... ({len(obj) - max_items} more)")
                break
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}- {k}:")
                lines.extend(_bullets(v, indent + 1, max_items))
            else:
                lines.append(f"{pad}- {k}: {_num(v)}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if i >= max_items:
                lines.append(f"{pad}- ... ({len(obj) - max_items} more)")
                break
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- " + json.dumps(v, default=str)[:300])
            else:
                lines.append(f"{pad}- {_num(v)}")
    return lines


def scorecard_markdown(card: dict[str, Any]) -> str:
    ins = card["instrument"]
    lines = [f"# Scorecard: cycle {card['cycle']} ({card['phase']}, config {card['config']})", "",
             f"**Question.** {card['question']}", "",
             f"- written {card['time']}{'' if card['final'] else ' (session still running)'}; elapsed "
             f"{fmt_h(card['elapsed_s'])} of a {fmt_h(card['budget_s'])} budget",
             f"- instrument: ladder version {ins['ladder_version']}, data hash `{ins['data_hash']}`, readout code "
             f"`{ins['ladder']}`, commit `{ins['git_commit'][:10]}`"]
    if card["phase"] == "develop":
        lines.append("- phase **develop**: every number below is exploratory; the instrument may still change")
    lines += ["", "## Jobs (budget ledger)", "", "| job | state | predicted | actual |", "|---|---|---|---|"]
    for e in card["jobs"]:
        act = "—" if e["actual_s"] is None else fmt_h(e["actual_s"])
        lines.append(f"| {e['job']} | {e['state']} | {fmt_h(e['predicted_s'])} | {act} |")
    lines += ["", "## Instrument checks per model", ""]
    if not card["models"]:
        lines.append("(no model has a finished ladder stage yet)")
    for m, res in card["models"].items():
        lines += [f"### {m}", ""] + _bullets(res) + [""]
    lines += ["## Stage pairs", ""]
    if not card["pairs"]:
        lines.append("(no pair has both models analysed yet)")
    for name, res in card["pairs"].items():
        lines += [f"### {name}", ""] + _bullets(res) + [""]
    if card.get("compared_with"):
        lines += [f"## What changed since cycle {card['compared_with']}", ""]
        lines += [f"- {c['key']}: {_num(c['before'])} -> {_num(c['now'])}" for c in card["changes"]] or \
            ["(no headline number moved by more than 5%)"]
        lines.append("")
    if card["decisions"]:
        lines += ["## Decisions this cycle informs", ""] + [f"- {d}" for d in card["decisions"]] + [""]
    if card["cut_order"]:
        lines += ["## Cut first if the session runs long", ""] + [f"{i}. {c}" for i, c in
                                                                  enumerate(card["cut_order"], 1)] + [""]
    return "\n".join(lines) + "\n"


def run_report(spec: CycleSpec, sets: list[str] | None = None, unseal: bool = False) -> Path:
    """Recompute the CPU analysis from the stored scores and rewrite the scorecard; no GPU stage is touched.
    Uses the overrides the cycle was run with. A cycle that reads the test split needs ``unseal`` here too."""
    from rtp import analyze

    cfg = cycle_config(spec, sets)
    check_phase(cfg, spec)
    if any(j.split == "test" for j in spec.jobs) and not unseal:
        from rtp.data.splits import SealedSplitError

        raise SealedSplitError(f"cycle '{spec.name}' reads the sealed test split: pass --unseal deliberately")
    cdir = cycle_dir(cfg, spec)
    by_job = {e["job"]: e for e in io.read_json(cdir / "ledger.json")} if (cdir / "ledger.json").exists() else {}
    ledger = []
    for j in spec.jobs:
        state = job_state(cfg, j)
        old = by_job.get(j.label, {})
        ledger.append({"job": j.label, "predicted_s": old.get("predicted_s", 0.0), "actual_s": old.get("actual_s"),
                       "state": "done" if state == "done" else (old.get("state") if old.get("state", "done") != "done"
                                                                else state)})
    _refresh(cfg, spec, ledger, analyze, time.time(), final=True, unseal=unseal, force=True)
    print(f"scorecard: {cdir / 'scorecard.md'}")
    return cdir / "scorecard.md"


def latest_scorecards(cfg: Config, n: int = 3) -> list[Path]:
    paths = sorted((run_dir(cfg.name) / "cycles").glob("*/scorecard.md"), key=lambda p: p.stat().st_mtime)
    return paths[-n:]


__all__ = ["BudgetExceeded", "CycleError", "CycleSpec", "Job", "MAX_BUDGET_HOURS", "load_cycle", "plan", "run_cycle",
           "run_plan", "run_report", "latest_scorecards"]
