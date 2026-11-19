"""Cycles: the budget cap, the cost model, the develop/confirm guard, the deadline, the scorecard."""

from __future__ import annotations

import time

import pandas as pd
import pytest
import yaml

from rtp import cost, cycle, io, phase, registry
from rtp.cli import main as cli_main
from rtp.config import ConfigError, config_from_dict
from rtp.data.build import build_data
from rtp.paths import run_dir, stage_dir

SMOKE = registry.SMOKE_MODEL
TULU_SFT = "allenai/Llama-3.1-Tulu-3-8B-SFT"
TULU_FINAL = "allenai/Llama-3.1-Tulu-3-8B"


def _write_cfg(tmp_path, name="cyc", **extra):
    raw = {"name": name, "seed": 2, "data": {
        "categories": ["laptop", "beer"], "test_categories": ["beer"], "n_real_per_category": 3,
        "n_placebo_per_category": 3, "cards_per_cell": 1, "evidence_levels": [0], "handicaps": [0],
        "probe_entities_per_category": 2, "probe_templates": 2},
        "ladder": {"enabled": True, "rating_deltas": [0.5], "price_deltas": [0.3], "review_ratios": [10.0],
                   "n_base": 1, "direct_entities": 1, "direct_controls": 1, "favourable_levels": 1},
        "models": [SMOKE, TULU_SFT], "confirm_models": [TULU_FINAL], **extra}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, config_from_dict(raw)


def _write_cycle(tmp_path, cfg_path, name="c", **over):
    raw = {"name": name, "config": str(cfg_path), "phase": "develop", "question": "Does it run?",
           "budget_hours": 1, "jobs": [{"stage": "ladder", "tier": "core", "models": [SMOKE, f"{SMOKE}@scale=2"]},
                                       {"stage": "open-set", "models": [SMOKE]}],
           "pairs": [[SMOKE, f"{SMOKE}@scale=2"]], **over}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_cycle_files_in_the_repo_are_valid_and_capped():
    for name in ("c0", "c1", "smoke"):
        spec = cycle.load_cycle(f"configs/cycles/{name}.yaml")
        assert spec.name == name and 0 < spec.budget_hours <= cycle.MAX_BUDGET_HOURS and spec.jobs
        assert spec.question and spec.phase == "develop"
    c0 = cycle.load_cycle("configs/cycles/c0.yaml")
    assert [j.label for j in c0.jobs][:2] == [f"ladder-core {TULU_SFT}", "ladder-core allenai/Llama-3.1-Tulu-3-8B-DPO"]
    assert not any("Olmo" in j.model or j.model == TULU_FINAL for n in ("c0", "c1")
                   for j in cycle.load_cycle(f"configs/cycles/{n}.yaml").jobs)


def test_cycle_file_validation(tmp_path):
    cfg_path, _ = _write_cfg(tmp_path)
    with pytest.raises(ConfigError, match="no cycle runs longer"):
        cycle.load_cycle(_write_cycle(tmp_path, cfg_path, budget_hours=13))
    with pytest.raises(ConfigError, match="unknown key"):
        cycle.load_cycle(_write_cycle(tmp_path, cfg_path, bugdet_hours=2))
    with pytest.raises(ConfigError, match="'tier' is required"):
        cycle.load_cycle(_write_cycle(tmp_path, cfg_path, jobs=[{"stage": "ladder", "models": [SMOKE]}]))
    with pytest.raises(ConfigError, match="stage must be one of"):
        cycle.load_cycle(_write_cycle(tmp_path, cfg_path, jobs=[{"stage": "extract", "models": [SMOKE]}]))
    with pytest.raises(ValueError, match="unknown ladder tier"):
        cycle.load_cycle(_write_cycle(tmp_path, cfg_path, jobs=[{"stage": "ladder", "tier": "big", "models": [SMOKE]}]))


def test_plan_prices_jobs_and_refuses_a_cycle_over_budget(tmp_path, work_dir, monkeypatch, capsys):
    cfg_path, cfg = _write_cfg(tmp_path)
    build_data(cfg)
    spec = cycle.load_cycle(_write_cycle(tmp_path, cfg_path))
    monkeypatch.setattr(cost, "is_cached", lambda model_id: False)  # a fresh Colab VM: nothing on local disk
    p = cycle.plan(cfg, spec)
    assert [r["state"] for r in p["jobs"]] == ["todo", "todo", "todo"] and p["fits"]
    first, second, third = p["jobs"]
    assert first["n_items"] == len(io.read_jsonl(run_dir(cfg.name) / "data" / "dev" / "ladder_core.jsonl"))
    # the second job loads the same checkpoint again: no download this time
    assert first["predicted_s"] - second["predicted_s"] == pytest.approx(cost.LOAD_FIRST_S - cost.LOAD_AGAIN_S)
    assert third["predicted_s"] == pytest.approx(cost.OPEN_SET_S + cost.LOAD_AGAIN_S)
    assert cli_main(["cycle", "plan", "--cycle", spec.source_path]) == 0
    assert "fits" in capsys.readouterr().out
    monkeypatch.setattr(cost, "S_PER_ITEM", 60.0)  # every item a minute: the cycle cannot fit one hour
    assert not cycle.plan(cfg, spec)["fits"]
    assert cli_main(["cycle", "plan", "--cycle", spec.source_path]) == 5
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_cost_model_recalibrates_on_finished_stages(tmp_path, work_dir, monkeypatch):
    cfg_path, cfg = _write_cfg(tmp_path)
    build_data(cfg)
    monkeypatch.setattr(cost, "is_cached", lambda model_id: True)
    assert cost.calibration(cfg)["k"] == 1.0
    n_items, n_tokens, measured = cost.tier_size(cfg, "core", "dev")
    assert not measured and n_tokens > n_items * cost.FRAME_TOKENS
    sdir = stage_dir(cfg.name, SMOKE, "ladder_core")
    io.prepare_stage_dir(sdir, "x: 1\n", "h", extra_meta={"model_id": SMOKE})
    prior = cost.prior_seconds(5000, n_items)
    io.mark_done(sdir, {"split": "dev", "tier": "core", "n_items": n_items, "n_tokens": 5000, "timing_complete": True,
                        "load_cached": True,
                        "timing": {"load_s": 21.0, "primary_s": 0.2 * prior, "diag_s": 0.05 * prior, "gen_s": 0.0}})
    cal = cost.calibration(cfg)
    assert cal["k"] == pytest.approx(0.25) and cal["load_again_s"] == 21.0 and cal["tokens"] == {"dev/core": 5000}
    assert cal["load_first_s"] == cost.LOAD_FIRST_S  # no download was timed yet: a warm load never prices a cold one
    monkeypatch.setattr(cost, "is_cached", lambda model_id: False)
    assert cost.estimate(cfg, "ladder", SMOKE, "core")["load_s"] == cost.LOAD_FIRST_S
    monkeypatch.setattr(cost, "is_cached", lambda model_id: True)
    est = cost.estimate(cfg, "ladder", f"{SMOKE}@scale=2", "core")  # same checkpoint, measured token count
    assert est["tokens_measured"] and est["n_tokens"] == 5000
    assert est["seconds"] == pytest.approx(0.25 * prior + 21.0)


def test_confirmation_models_are_refused_until_the_freeze(tmp_path, work_dir):
    cfg_path, cfg = _write_cfg(tmp_path)
    build_data(cfg)
    phase.check_model_allowed(cfg, TULU_SFT)
    for m in (TULU_FINAL, f"{TULU_FINAL}@scale=2"):
        with pytest.raises(phase.PhaseError, match="confirmation model"):
            phase.check_model_allowed(cfg, m)
    assert cli_main(["ladder", "--config", str(cfg_path), "--model", TULU_FINAL]) == 4  # refused before any load
    # EVERY command that loads a model refuses it, the legacy and frozen ones included
    for stage in ("open-set", "measure", "open-ended", "extract", "probe", "localize", "format-probe"):
        assert cli_main([stage, "--config", str(cfg_path), "--model", TULU_FINAL]) == 4, stage
    # the model lists cannot be emptied from the command line, and the project's own set holds under any config
    assert cli_main(["--set", "confirm_models=[]", "ladder", "--config", str(cfg_path), "--model", TULU_FINAL]) == 4
    bare_path, bare = _write_cfg(tmp_path, name="bare", confirm_models=[])
    for m in ("allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Instruct-SFT@scale=2", TULU_FINAL):
        assert phase.is_confirm_model(bare, m), m
    assert not phase.is_confirm_model(bare, TULU_SFT) and not phase.is_confirm_model(bare, SMOKE)
    with pytest.raises(ConfigError, match="confirmation model"):
        _write_cfg(tmp_path, name="wrong", models=[TULU_FINAL], confirm_models=[])
    # the sealed test categories belong to the confirmation phase: --unseal alone does not open them
    assert cli_main(["ladder", "--config", str(cfg_path), "--model", SMOKE, "--split", "test", "--unseal"]) == 4
    with pytest.raises(phase.PhaseError, match="opened only after"):
        phase.check_test_split_allowed(cfg, "test")
    bad = cycle.load_cycle(_write_cycle(tmp_path, cfg_path, name="bad",
                                        jobs=[{"stage": "ladder", "tier": "core", "models": [TULU_FINAL]}], pairs=[]))
    with pytest.raises(cycle.CycleError, match="develop cycle but lists confirmation models"):
        cycle.check_phase(cfg, bad)
    conf = cycle.load_cycle(_write_cycle(tmp_path, cfg_path, name="conf", phase="confirm",
                                         jobs=[{"stage": "ladder", "tier": "core", "models": [TULU_FINAL]}], pairs=[]))
    with pytest.raises(cycle.CycleError, match="not frozen"):
        cycle.check_phase(cfg, conf)
    test_read = cycle.load_cycle(_write_cycle(tmp_path, cfg_path, name="t", pairs=[], jobs=[
        {"stage": "ladder", "tier": "core", "models": [SMOKE], "split": "test"}]))
    with pytest.raises(cycle.CycleError, match="sealed test split"):
        cycle.check_phase(cfg, test_read)
    (run_dir(cfg.name) / phase.FREEZE_FILE).write_text("{}")
    phase.check_model_allowed(cfg, TULU_FINAL)
    phase.check_test_split_allowed(cfg, "test")
    cycle.check_phase(cfg, conf)


def _fake_stage(cfg, model, name, stage_hash, done=True, shards=0, created_with=None):
    sdir = stage_dir(cfg.name, model, name)
    io.prepare_stage_dir(sdir, "x: 1\n", stage_hash, extra_meta={"model_id": model, "created_with": created_with or {}})
    for k in range(shards):
        (sdir / "shards").mkdir(exist_ok=True)
        (sdir / "shards" / f"shard_{k:05d}.done").write_text("1")
    if done:
        io.mark_done(sdir, {})
    return sdir


def test_a_cycle_refuses_stale_or_interrupted_stages_unless_told_to_redo_them(tmp_path, work_dir, monkeypatch):
    """After an instrument change the stages of earlier cycles are stale. A cycle that skipped them would spend
    its budget and end with an empty scorecard; it must stop before any GPU time instead."""
    from rtp import stages
    from rtp.provenance import environment

    cfg_path, cfg = _write_cfg(tmp_path)
    build_data(cfg)
    spec = cycle.load_cycle(_write_cycle(tmp_path, cfg_path))
    _fake_stage(cfg, SMOKE, "ladder_core", "hash-of-an-older-instrument")
    assert cycle.job_state(cfg, spec.jobs[0]) == "stale"
    with pytest.raises(cycle.CycleError, match="--redo-stale"):
        cycle.run_plan(cfg, spec)
    assert cli_main(["cycle", "plan", "--cycle", spec.source_path]) == 4
    assert cycle.run_plan(cfg, spec, redo_stale=True)["jobs"][0]["state"] == "stale"  # priced, and allowed
    # half-finished under another library version: the stage would refuse to resume
    current = stages.stage_hash(cfg, "ladder", f"{SMOKE}@scale=2", "dev", tier="core")
    other = {**environment(), "transformers": "0.0.0"}
    _fake_stage(cfg, f"{SMOKE}@scale=2", "ladder_core", current, done=False, shards=1, created_with=other)
    assert cycle.job_state(cfg, spec.jobs[1]) == "partial-env"
    _fake_stage(cfg, SMOKE, "open_set", stages.stage_hash(cfg, "open_set", SMOKE, "dev"), done=False,
                created_with=environment())
    assert cycle.job_state(cfg, spec.jobs[2]) == "partial"  # nothing computed yet: resumable
    # a confirm cycle never re-measures: stale stages after the freeze mean the instrument changed
    (run_dir(cfg.name) / phase.FREEZE_FILE).write_text("{}")
    conf = cycle.load_cycle(_write_cycle(tmp_path, cfg_path, name="conf2", phase="confirm"))
    with pytest.raises(cycle.CycleError, match="never re-measures"):
        cycle.run_plan(cfg, conf, redo_stale=True)


def test_cycle_run_orchestration(tmp_path, work_dir, monkeypatch):
    """Without a model: the stage processes are replaced by a recorder. Checks the argv of every stage (the
    user's overrides after the GPU defaults, --force only for redone jobs), the environment passed down, the
    code-state guard, the ledger carried over between sessions, and the overrides remembered by the cycle."""
    import rtp.notebook as nb

    cfg_path, cfg = _write_cfg(tmp_path)
    build_data(cfg)
    spec = cycle.load_cycle(_write_cycle(tmp_path, cfg_path))
    calls = []

    def fake_stream(cmd, env=None, label=None):
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "label": label})
        return 6 if "open-set" in cmd else 0  # the open-set job "reaches the deadline"

    monkeypatch.setattr(nb, "stream", fake_stream)
    monkeypatch.setattr(nb, "gpu_overrides", lambda: ["--set", "ladder_measure.max_batch_size=128"])
    rc = cycle.run_cycle(spec, sets=["ladder_measure.max_batch_size=8"], env={"RTP_EVENTS_PATH": "/tmp/x.jsonl"})
    assert rc == 7 and len(calls) == 3
    first = calls[0]["cmd"]
    sets = [first[i + 1] for i, x in enumerate(first) if x == "--set"]
    assert sets == ["ladder_measure.max_batch_size=128", "ladder_measure.max_batch_size=8"]  # the user's wins
    assert "--force" not in first and first[first.index("--tier") + 1] == "core"
    assert calls[0]["env"]["RTP_EVENTS_PATH"] == "/tmp/x.jsonl" and float(calls[0]["env"]["RTP_DEADLINE"]) > time.time()
    cdir = cycle.cycle_dir(cfg, spec)
    card = io.read_json(cdir / "scorecard.json")
    assert [e["state"] for e in card["jobs"]] == ["done", "done", "deferred (budget reached mid-stage; resumable)"]
    assert card["final"] and (cdir / "scorecard.md").exists()
    meta = io.read_json(cdir / "run_meta.json")
    assert meta["sets"] == ["ladder_measure.max_batch_size=8"] and len(meta["sessions"]) == 1
    # the overrides belong to the cycle: report and a resumed run reuse them, other ones are refused
    assert cycle.cycle_config(spec, None).ladder_measure.max_batch_size == 8
    with pytest.raises(cycle.CycleError, match="was started with --set"):
        cycle.cycle_config(spec, ["ladder_measure.max_batch_size=16"])

    # second session: the measurement code changed and a job already has results from the old code -> refused
    from rtp import stages

    done_hash = stages.stage_hash(cycle.cycle_config(spec, None), "ladder", SMOKE, "dev", tier="core")
    _fake_stage(cfg, SMOKE, "ladder_core", done_hash)
    io.write_json(cdir / "ledger.json", [{"job": spec.jobs[0].label, "predicted_s": 111.0, "actual_s": 99.0,
                                          "state": "done"}])
    real_state = cycle.code_state()
    monkeypatch.setattr(cycle, "code_state", lambda: {**real_state, "ladder": "another-digest"})
    with pytest.raises(cycle.CycleError, match="measurement code .* changed"):
        cycle.run_cycle(spec)
    assert io.read_json(cdir / "run_meta.json")["code"]["ladder"] == real_state["ladder"]  # never overwritten
    monkeypatch.setattr(cycle, "code_state", lambda: real_state)
    calls.clear()
    assert cycle.run_cycle(spec) == 7  # same code: resumes; the finished job keeps its earlier times
    assert [c["label"].split()[1] for c in calls] == ["ladder-core", "open-set"]
    ledger = io.read_json(cdir / "ledger.json")
    assert (ledger[0]["predicted_s"], ledger[0]["actual_s"], ledger[0]["state"]) == (111.0, 99.0, "done")
    assert len(io.read_json(cdir / "run_meta.json")["sessions"]) == 2
    # --redo-stale re-measures a stale job from scratch, and says so in the ledger
    _fake_stage(cfg, f"{SMOKE}@scale=2", "ladder_core", "old-instrument")
    calls.clear()
    cycle.run_cycle(spec, redo_stale=True)
    redone = [c["cmd"] for c in calls if f"{SMOKE}@scale=2" in c["cmd"]][0]
    assert redone[-1] == "--force"
    assert io.read_json(cdir / "ledger.json")[1]["redone"] == "stale"


def test_run_sharded_stops_at_the_deadline_and_resumes(tmp_path, monkeypatch):
    done: list[int] = []

    def process(chunk, idx):
        done.append(idx)
        if idx == 1:  # the budget runs out while shard 2 is being computed
            monkeypatch.setenv("RTP_DEADLINE", repr(time.time() - 1))
        return pd.DataFrame({"x": list(chunk)})

    with pytest.raises(io.DeadlineReached, match="re-run the same command"):
        io.run_sharded(list(range(10)), 2, tmp_path / "s", process, desc="t")
    assert done == [0, 1] and len(io.load_shards(tmp_path / "s")) == 4  # finished shards are kept
    monkeypatch.delenv("RTP_DEADLINE")
    io.run_sharded(list(range(10)), 2, tmp_path / "s", process, desc="t")
    assert done == [0, 1, 2, 3, 4] and list(io.load_shards(tmp_path / "s")["x"]) == list(range(10))


def test_scorecard_renders_and_reports_changes():
    old = {"cycle": "c0", "models": {"m": {"checks": {"readout": {"pass": True, "c_forced": 0.97}}}},
           "pairs": {"a -> b": {"attr/rating": {"c": 1.70, "T1": 0.02}}}}
    card = {"cycle": "c1", "config": "e1", "phase": "develop", "question": "Q?", "time": "t", "final": True,
            "instrument": {"ladder_version": 2, "data_hash": "abc", "ladder": "d1", "open_set": "d2", "git_commit": "f" * 40},
            "elapsed_s": 4000.0, "budget_s": 6 * 3600.0,
            "jobs": [{"job": "ladder-core m", "predicted_s": 600.0, "actual_s": 420.0, "state": "done"},
                     {"job": "ladder-ext m", "predicted_s": 1800.0, "actual_s": None,
                      "state": "deferred (does not fit in the remaining budget)"}],
            "models": {"m": {"checks": {"readout": {"pass": False, "c_forced": 0.71, "mean_c_forced": 0.71},
                                        "thresholds": {"COVERAGE_MIN": 0.9}}}},
            "pairs": {"a -> b": {"attr/rating": {"c": 1.71, "T1": 0.20}}}, "decisions": ["Keep the full card?"],
            "cut_order": ["ctx tier"]}
    changes = cycle.diff_cards(old, card)
    keys = {c["key"] for c in changes}
    assert "pairs/a -> b/attr/rating/T1" in keys and "models/m/checks/readout/c_forced" in keys
    assert "pairs/a -> b/attr/rating/c" not in keys  # moved by less than 5%
    card["changes"], card["compared_with"] = changes, "c0"
    md = cycle.scorecard_markdown(card)
    for needle in ("# Scorecard: cycle c1", "**Question.** Q?", "phase **develop**", "| ladder-core m | done | 10 min |",
                   "deferred", "## Findings", "the letter readout is not the model's answer", "FAIL",
                   "1 of 2 jobs are not done", "## What changed since cycle c0", "Keep the full card?", "1. ctx tier"):
        assert needle in md, needle
    assert md.index("## Findings") < md.index("## Instrument checks per model") < md.index("## Jobs (budget ledger)")
