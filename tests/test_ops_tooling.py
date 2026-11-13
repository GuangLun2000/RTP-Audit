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
    for section in ("## Environment", "## Stage status", "## Suggested next actions", "## Errors (1 unresolved",
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
                   '"pip", "install", "-q", "-e", ".[dev]"', 'nb.run("smoke")', "nb.pytest(", '"build-data"', '"measure"',
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



def test_smoke_starts_on_a_fresh_work_dir_with_event_log(work_dir, monkeypatch):
    """Regression: the event log used to create runs/smoke/ first, and smoke then refused to use it."""
    from rtp.paths import run_dir

    class Stop(RuntimeError):
        pass

    def stub(path):
        raise Stop("model load reached")

    monkeypatch.setattr("rtp.models.load_model", stub)
    with pytest.raises(Stop):
        cli_main(["smoke"])
    assert (run_dir("smoke") / "_SMOKE_RUN").exists()
    assert (run_dir("smoke").parent / "smoke.smoke_events.jsonl").exists()


def test_status_and_feedback_seal_test_results(work_dir, tmp_path):
    from rtp.data.build import build_data
    from rtp.ops import build_feedback, collect_status
    from rtp.paths import stage_dir

    cfg = load_config(_cfg_file(tmp_path, "sealy"))
    build_data(cfg)
    sdir = stage_dir(cfg.name, "dummy/model", "measure_test")
    io.prepare_stage_dir(sdir, "x: 1\n", "h", extra_meta={"model_id": "dummy/model", "stage": "measure",
                                                           "split": "test"})
    io.mark_done(sdir, {"ladder_slope": 0.4242, "responds_to_ratings": False})
    (sdir / "summary.md").write_text("# sealed summary\nTEST-ONLY-NUMBER 0.7777\n")
    ddir = stage_dir(cfg.name, "dummy/model", "measure")  # a dev stage: its summary head is in the bundle
    io.prepare_stage_dir(ddir, "x: 1\n", "h", extra_meta={"model_id": "dummy/model", "stage": "measure", "split": "dev"})
    io.mark_done(ddir, {"n_items": 1})
    (ddir / "summary.md").write_text("# dev summary\nDEV-NUMBER 0.1234\n" + "row\n" * 200)
    row = next(r for r in collect_status(cfg) if r["stage"] == "measure_test")
    assert row["warnings"] == ["(sealed: pass --unseal to show)"]
    text = build_feedback(cfg).read_text()
    assert "0.4242" not in text and "0.7777" not in text
    assert "DEV-NUMBER 0.1234" in text and "truncated" in text
    unsealed = build_feedback(cfg, unseal=True).read_text()
    assert "0.4242" in unsealed and "0.7777" in unsealed


def test_error_events_are_redacted_and_resolved_errors_omitted(work_dir, tmp_path, monkeypatch):
    from rtp.ops import build_feedback

    secret = "hf_" + "Z" * 30
    monkeypatch.setenv("HF_TOKEN", secret)
    path = _cfg_file(tmp_path, "redact")

    def boom(args):
        raise RuntimeError(f"boom with {secret}")

    import rtp.cli

    original = rtp.cli._dispatch
    monkeypatch.setattr("rtp.cli._dispatch", boom)
    with pytest.raises(RuntimeError):
        cli_main(["status", "--config", str(path)])
    text = (work_dir / "runs" / "redact" / "events.jsonl").read_text()
    assert secret not in text and "boom with" in text
    monkeypatch.setattr("rtp.cli._dispatch", original)
    assert cli_main(["status", "--config", str(path)]) == 0  # same stage succeeds later -> resolved
    bundle = build_feedback(load_config(path)).read_text()
    assert "## Errors (0 unresolved; 1 resolved" in bundle and secret not in bundle


def test_notebook_bootstraps_from_github_and_is_versioned():
    """The notebook is opened from GitHub, clones the code from GitHub, and carries a revision of its code cells
    so that an outdated copy is noticed (bump NOTEBOOK_REV with rtp.notebook.notebook_rev after editing)."""
    import re

    from rtp.notebook import notebook_rev

    path = REPO_ROOT / "colab" / "run_pilot.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    rev = re.findall(r'(?m)^NOTEBOOK_REV = "([^"]+)"', src)
    assert rev == [notebook_rev(path)], f"code cells changed: set NOTEBOOK_REV = \"{notebook_rev(path)}\" in cell 1"
    url = re.search(r'REPO_URL = "([^"]+)"', src).group(1)
    pin = re.search(r'PIN = "([^"]+)"', src).group(1)
    assert url.startswith("https://github.com/") and "<" not in url
    owner_repo = url.removeprefix("https://github.com/").removesuffix(".git")
    link = f"https://colab.research.google.com/github/{owner_repo}/blob/{pin}/colab/run_pilot.ipynb"
    assert src.count(link) >= 2  # the "Open in Colab" link (cell 0) and cell 2's outdated-copy message
    assert "❌" in "".join(nb["cells"][2]["source"]) and "✅ code ready" in "".join(nb["cells"][2]["source"])


def test_stage_processes_do_not_load_tensorflow():
    """Colab preinstalls TensorFlow; its import printed "Could not find cuda drivers ... GPU will not be used" in
    every stage (TensorFlow's message, not PyTorch's) and cost start-up time."""
    import subprocess
    import sys

    from rtp.notebook import _env

    assert _env()["USE_TF"] == "0" and _env({"USE_TF": "1"})["USE_TF"] == "1"
    env = {k: v for k, v in _env().items() if k not in ("USE_TF", "USE_FLAX", "TF_CPP_MIN_LOG_LEVEL")}
    out = subprocess.run([sys.executable, "-c", "import os, rtp.cli; print(os.environ['USE_TF'], os.environ['USE_FLAX'])"],
                         capture_output=True, text=True, env=env, check=True).stdout.split()
    assert out == ["0", "0"]
