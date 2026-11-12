"""Colab tooling and the improvement loop: --set overrides, events, status, doctor, feedback, notebook helpers."""

from __future__ import annotations

import json
import logging
import sys

import pytest
import yaml

from rtp import events, io
from rtp.cli import main as cli_main
from rtp.config import REPO_ROOT, ConfigError, apply_overrides, config_from_dict, load_config


def _cfg_file(tmp_path, name="ops"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, "data": {
        "categories": ["running shoe"], "n_real_per_category": 3, "n_placebo_per_category": 3, "cards_per_cell": 1,
        "evidence_levels": [0], "probe_entities_per_category": 1, "probe_templates": 1},
        "models": ["dummy/model"]}))
    return path


def test_apply_overrides():
    cfg = load_config("configs/smoke.yaml")
    c2 = apply_overrides(cfg, ["measure.max_batch_size=8", "probe.alphas=[1, 2]", "open_ended.temperature=0.5"])
    assert c2.measure.max_batch_size == 8 and c2.probe.alphas == [1, 2] and c2.open_ended.temperature == 0.5
    for bad in (["measure.nope=1"], ["nosection.x=1"], ["measure.max_batch_size"], ["measure.max_batch_size=0"]):
        with pytest.raises(ConfigError):
            apply_overrides(cfg, bad)
    # batch limits are not part of a stage's identity
    assert c2.stage_hash("measure", "m", "dev") == cfg.stage_hash("measure", "m", "dev")


def test_cli_events_record_start_done_and_errors(work_dir, tmp_path):
    path = _cfg_file(tmp_path)
    assert cli_main(["--set", "measure.max_batch_size=4", "status", "--config", str(path)]) == 0
    assert cli_main(["measure", "--config", str(path), "--model", "dummy/model"]) == 4  # data not built
    evs = events.read(work_dir / "runs" / "ops" / "events.jsonl")
    kinds = [e["kind"] for e in evs]
    assert kinds.count("start") == 2 and kinds.count("done") == 2 and "error" in kinds
    err = next(e for e in evs if e["kind"] == "error")
    assert err["stage"] == "measure" and err["model"] == "dummy/model" and "DataNotBuiltError" in err["message"]
    assert "Traceback" in err["data"]["traceback"]


def test_status_and_feedback_bundle(work_dir, tmp_path, monkeypatch):
    from rtp.data.build import build_data
    from rtp.ops import build_feedback, collect_status
    from rtp.paths import stage_dir

    secret = "hf_" + "S" * 30
    monkeypatch.setenv("HF_TOKEN", secret)
    path = _cfg_file(tmp_path)
    cfg = load_config(path)
    build_data(cfg)
    sdir = stage_dir(cfg.name, "dummy/model", "measure")  # a finished stage from "other code"
    io.prepare_stage_dir(sdir, "x: 1\n", "old-hash", extra_meta={"model_id": "dummy/model", "stage": "measure",
                                                                  "split": "dev"})
    io.mark_done(sdir, {"responds_to_ratings": False, "elapsed_s": 3.0})
    (sdir / "log.txt").write_text("line 1\nline 2\n")
    assert cli_main(["measure", "--config", str(path), "--model", "dummy/model"]) == 4  # stale -> refused
    logging.getLogger("rtp.test").warning("a warning mentioning %s", secret)
    rows = {(r["model"], r["stage"]): r for r in collect_status(cfg)}
    assert rows[("(data)", "build-data")]["state"] == "done"
    assert rows[("dummy/model", "measure")]["state"] == "stale"
    assert rows[("dummy/model", "open_ended")]["state"] == "not started"
    assert "does not respond to rating evidence" in rows[("dummy/model", "measure")]["warnings"]
    bundle = build_feedback(cfg).read_text()
    for section in ("## Environment", "## Stage status", "## Suggested next actions", "## Errors (1)",
                    "## Warnings", "## Key results", "## Log tails", "## Last pytest run"):
        assert section in bundle, section
    assert "StageConfigMismatch" in bundle and "re-run them with --force" in bundle
    assert "do NOT change the card design or thresholds" in bundle
    assert secret not in bundle and "line 2" in bundle
    js = json.loads(next((work_dir / "runs" / "ops" / "feedback").glob("*.json")).read_text())
    assert js["errors"] and any(r["state"] == "stale" for r in js["status"])


def test_doctor_runs_without_hub(work_dir, tmp_path, capsys):
    from rtp.ops import run_doctor

    cfg = load_config(_cfg_file(tmp_path))
    rc = run_doctor(cfg, check_hub=False)
    out = capsys.readouterr().out
    assert "transformers" in out and "RTP_WORK_DIR" in out and "doctor:" in out
    assert rc in (0, 1)
    assert (work_dir / "runs" / "ops" / "doctor.json").exists()


def test_notebook_stream_and_history(capsys):
    from rtp import notebook as nb

    nb.HISTORY.clear()
    rc = nb.stream([sys.executable, "-c", "import sys; print('hello'); print('WARNING x', file=sys.stderr); sys.exit(3)"])
    out = capsys.readouterr().out
    assert rc == 3 and "hello" in out and "⚠️ WARNING x" in out and "FAILED (exit 3)" in out
    assert nb.HISTORY[-1]["exit_code"] == 3 and nb.HISTORY[-1]["warnings"] == 1
    with pytest.raises(RuntimeError):
        nb.run("not-a-stage", strict=True)
    assert nb.gpu_overrides() == [] or nb.gpu_overrides()[0] == "--set"


def test_progress_log_mode(monkeypatch, caplog):
    from rtp import progress

    monkeypatch.setenv("RTP_PROGRESS", "log")
    times = iter(range(0, 1000, 7))
    monkeypatch.setattr(progress.time, "time", lambda: next(times))
    with caplog.at_level(logging.INFO, logger="rtp.progress"):
        assert list(progress.track(range(20), desc="work")) == list(range(20))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ETA" in m for m in msgs) and any("20/20 done" in m for m in msgs)


def test_notebook_covers_spec_cells():
    nb = json.loads((REPO_ROOT / "colab/run_pilot.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ('drive.mount("/content/drive")', 'os.environ["RTP_WORK_DIR"] = "/content/drive/MyDrive/rtp"',
                   'os.environ["HF_HOME"] = "/content/hf"', 'userdata.get("HF_TOKEN")', "nvidia-smi",
                   'pip -q install -e ".[dev]"', 'nb.run("smoke")', "nb.pytest(", '"build-data"', '"measure"',
                   '"open-ended"', '"extract"', '"probe"', '"report"', "nb.feedback(CONFIG)", "doctor"):
        assert needle in src, needle


def test_event_log_redacts_secrets(tmp_path, monkeypatch):
    from rtp.log import RedactSecrets

    secret = "hf_" + "Q" * 30
    monkeypatch.setenv("HF_TOKEN", secret)
    events.configure(tmp_path / "events.jsonl", config="c", stage="s")
    handler = events.EventLogHandler()
    handler.addFilter(RedactSecrets())
    logger = logging.getLogger("rtp.redaction-test")
    logger.addHandler(handler)
    try:
        logger.warning("token is %s", secret)
    finally:
        logger.removeHandler(handler)
        events.configure(None)
    text = (tmp_path / "events.jsonl").read_text()
    assert secret not in text and "***" in text
