"""Helpers for colab/run_cycle.ipynb: run stages with live, readable output and keep a session history.

Keeping this logic in the package (not in notebook cells) means it is versioned, tested and improved
together with the code: the notebook stays a thin, stable list of calls.

    from rtp import notebook as nb
    nb.run("doctor", "--config", CONFIG)
    nb.run("measure", "--config", CONFIG, "--model", m, *nb.gpu_overrides())
    nb.pytest(CONFIG)
    nb.feedback(CONFIG)        # writes the bundle and shows where it is
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HISTORY: list[dict[str, Any]] = []
_MARK = {"WARNING": "⚠️ ", "ERROR": "❌ ", "Traceback": "❌ "}


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("RTP_PROGRESS", "log")  # one line per ~10% with ETA, no carriage-return bars
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # download bars would print one line per refresh
    env.setdefault("TQDM_DISABLE", "1")
    from rtp.cli import TORCH_ONLY_ENV  # no TensorFlow/JAX in stages or pytest (start-up time, misleading GPU logs)

    for k, v in TORCH_ONLY_ENV.items():
        env.setdefault(k, v)
    env.update(extra or {})
    return env


def stream(cmd: list[str], env: dict[str, str] | None = None, label: str | None = None) -> int:
    """Run ``cmd``, print its output line by line as it arrives (warnings/errors marked), return the exit code."""
    label = label or " ".join(cmd)
    t0 = time.time()
    print(f"\n▶ {time.strftime('%H:%M:%S')}  {label}", flush=True)
    n_warn = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          env=_env(env)) as proc:  # the context manager closes the pipe
        assert proc.stdout is not None
        for line in proc.stdout:
            mark = next((m for k, m in _MARK.items() if k in line), "")
            n_warn += mark == _MARK["WARNING"]
            print(mark + line, end="", flush=True)
        rc = proc.wait()
    secs = time.time() - t0
    status = "✅ ok" if rc == 0 else f"❌ FAILED (exit {rc})"
    print(f"{status}  {label}  [{secs:.0f}s, {n_warn} warnings]", flush=True)
    _record({"command": label, "exit_code": rc, "seconds": round(secs, 1), "warnings": n_warn})
    return rc


def _record(entry: dict[str, Any]) -> None:
    """Keep the session history in memory and on Drive ($RTP_WORK_DIR/notebook_history.jsonl) for `feedback`."""
    import json

    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    HISTORY.append(entry)
    try:
        from rtp.events import redact
        from rtp.paths import work_dir

        with open(work_dir() / "notebook_history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(redact(entry), ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - history must never break a run
        pass


def run(*args: str, strict: bool = False, env: dict[str, str] | None = None) -> int:
    """``python -m rtp.cli <args>`` with live output. ``strict=True`` raises on failure (default: continue,
    so one failing model does not stop the others; the failure is recorded for the feedback bundle)."""
    rc = stream([sys.executable, "-m", "rtp.cli", *args], env=env, label="rtp.cli " + " ".join(args))
    if strict and rc != 0:
        raise RuntimeError(f"rtp.cli {' '.join(args)} failed with exit code {rc}")
    return rc


def _pytest_env(require_integration: bool = True) -> dict[str, str]:
    """Environment for the test run: a throw-away RTP_WORK_DIR, so tests never write into the real results or
    the session history (tests of the notebook helpers record commands; they used to land in the PI's history)."""
    import tempfile

    env = {"RTP_WORK_DIR": tempfile.mkdtemp(prefix="rtp-pytest-")}
    if require_integration:
        env["RTP_REQUIRE_INTEGRATION"] = "1"
    return env


def pytest(config: str | None = None, require_integration: bool = True) -> int:
    """Run the test suite (in a throw-away work dir) and save its output where ``feedback`` picks it up."""
    env = _pytest_env(require_integration)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"]
    out_path = _run_dir(config) / "pytest_last.txt" if config else None
    t0 = time.time()
    print(f"\n▶ {time.strftime('%H:%M:%S')}  pytest", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env(env))
    text = proc.stdout + proc.stderr
    tail = "\n".join(text.splitlines()[-25:])
    print(tail, flush=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    _record({"command": "pytest", "exit_code": proc.returncode, "seconds": round(time.time() - t0, 1),
             "warnings": 0})
    print(("✅ tests passed" if proc.returncode == 0 else f"❌ tests FAILED (exit {proc.returncode})"), flush=True)
    return proc.returncode


def _run_dir(config: str) -> Path:
    from rtp.config import load_config
    from rtp.paths import run_dir

    return run_dir(load_config(config).name)


def gpu_overrides() -> list[str]:
    """Batch limits sized to the GPU (batch limits are not part of any stage identity, so resuming is safe)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:  # noqa: BLE001
        return []
    if gib >= 120:  # H200 (141 GB)
        sizes = {"ladder_measure.max_batch_size": 192, "ladder_measure.max_batch_tokens": 65536,
                 "open_set.max_batch_size": 128, "open_set.max_batch_tokens": 65536,
                 "measure.max_batch_size": 128, "measure.max_batch_tokens": 65536,
                 "extract.max_batch_size": 64, "extract.max_batch_tokens": 32768,
                 "localize.max_batch_size": 96, "localize.max_batch_tokens": 49152}
    elif gib >= 76:  # 96 GB class (e.g. G4 / RTX PRO 6000) and 80 GB A100/H100 (~79 GiB usable)
        sizes = {"ladder_measure.max_batch_size": 128, "ladder_measure.max_batch_tokens": 49152,
                 "open_set.max_batch_size": 96, "open_set.max_batch_tokens": 49152,
                 "measure.max_batch_size": 96, "measure.max_batch_tokens": 49152,
                 "extract.max_batch_size": 48, "extract.max_batch_tokens": 24576,
                 "localize.max_batch_size": 64, "localize.max_batch_tokens": 32768}
    else:
        return []
    out: list[str] = []
    for k, v in sizes.items():
        out += ["--set", f"{k}={v}"]
    return out


def config_name(config: str) -> str:
    """The run name of a config file (its directory under ``runs/``)."""
    from rtp.config import load_config

    return load_config(config).name


def cycle_name(cycle: str) -> str:
    from rtp.cycle import load_cycle

    return load_cycle(cycle).name


def show(path: str | Path, max_chars: int = 20000) -> None:
    """Render a Markdown file in the notebook (plain print outside IPython)."""
    path = Path(path)
    if not path.exists():
        print(f"(missing: {path})")
        return
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:  # never cut silently: the reader must know that something follows
        text = text[:max_chars] + f"\n\n[... cut after {max_chars} characters; the full file is {path}]\n"
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)


def summary() -> None:
    """Commands run in this session, with exit codes (the last thing to look at before `feedback`)."""
    if not HISTORY:
        print("(no commands run yet)")
        return
    width = max(len(h["command"]) for h in HISTORY)
    for h in HISTORY:
        mark = "✅" if h["exit_code"] == 0 else "❌"
        print(f"{mark} {h['command']:<{width}}  exit {h['exit_code']}  {h['seconds']:>6.0f}s  {h['warnings']} warnings")


def feedback(config: str) -> Path | None:
    """Write the feedback bundle and PRINT THE DIGEST: the cycle's findings and every key number as plain
    text, between a BEGIN and an END line, made to be copied into the chat with the coding agent. Returns
    the path of the complete bundle (the fallback when the agent asks for more)."""
    summary()
    rc = run("feedback", "--config", config)
    if rc != 0:
        return None
    fdir = _run_dir(config) / "feedback"
    latest = max(fdir.glob("feedback_*.md"), key=lambda p: p.stat().st_mtime, default=None)
    digest = max(fdir.glob("digest_*.md"), key=lambda p: p.stat().st_mtime, default=None)
    if digest:
        text = digest.read_text(encoding="utf-8")
        print("\n" + text)
        copy_box(text)
        print(f"📋 Copy the digest above, from the BEGIN line to the END line ({len(text) / 1024:.0f} KB), and paste "
              f"it to the coding agent: it holds the findings and every key number of this cycle.")
    if latest:
        print(f"📦 Complete bundle: {latest} ({latest.stat().st_size / 1024:.0f} KB); only needed when the agent "
              f"asks for more than the digest.")
    return latest


def copy_box(text: str) -> bool:
    """On Colab: a text box holding ``text`` and a button that copies it (a click inside the box, then
    Ctrl/Cmd+A and Ctrl/Cmd+C, copies it as well). False elsewhere: the printed text is the same."""
    try:
        from google.colab import output as _colab  # only on Colab: elsewhere the printed text is enough
        from IPython.display import HTML, display

        del _colab
    except ImportError:
        return False
    import html

    display(HTML(
        '<div style="margin:6px 0"><button onclick="(function(b){var t=b.parentNode.querySelector(\'textarea\');'
        't.focus();t.select();var ok=false;try{ok=document.execCommand(\'copy\');}catch(e){}'
        'var m=b.parentNode.querySelector(\'span\');function say(x){m.textContent=x;}'
        'if(navigator.clipboard){navigator.clipboard.writeText(t.value).then(function(){say(\' copied\');},'
        'function(){say(ok?\' copied\':\' the text is selected: press Ctrl/Cmd+C\');});}'
        'else{say(ok?\' copied\':\' the text is selected: press Ctrl/Cmd+C\');}})(this)">Copy the digest</button>'
        '<span></span><br><textarea readonly rows="5" style="width:100%;font-family:monospace;font-size:11px">'
        + html.escape(text) + "</textarea></div>"))
    return True


def download(path: str | Path) -> bool:
    """Send a results file to the browser (Colab). False outside Colab; the file stays where it is."""
    try:
        from google.colab import files
    except ImportError:
        print(f"not on Colab: the file is at {path}")
        return False
    files.download(str(path))
    print(f"⬇️ download started: {Path(path).name}. It is also on Drive under runs/<config>/feedback/.")
    return True


def ensure_peft() -> bool:
    """Make sure peft can create LoRA adapters (needed by every ``zoo:`` variant). Colab ships a torchao that
    newer peft refuses when an adapter is injected; rtp never uses torchao, so an incompatible copy is removed
    from this session's environment. Returns True when adapters can be created."""
    from rtp.ops import PEFT_PROBE

    def probe() -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", PEFT_PROBE], capture_output=True, text=True)

    first = probe()
    if first.returncode == 0:
        return True
    if "torchao" not in first.stdout + first.stderr:
        print("⚠️ peft cannot be used:", (first.stdout + first.stderr).strip()[-400:])
        return False
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], capture_output=True, text=True)
    ok = probe().returncode == 0
    print("ℹ️ removed the preinstalled torchao (this peft refuses its version; rtp does not use torchao): "
          + ("peft can create adapters now" if ok else "peft still fails, see doctor"))
    return ok


def release_gpu(wait_seconds: float = 10.0, force: bool = False) -> bool:
    """End the Colab session so the GPU is released: wait ``wait_seconds`` (the browser finishes downloading
    the feedback bundle), let Drive finish writing, then unassign the runtime. Everything on the VM's local
    disk is lost, so the session is kept when the results are not on Drive, unless ``force``. Returns False
    when nothing was released (not on Colab, or results only on the local disk)."""
    import time

    from rtp.paths import work_dir

    try:
        from google.colab import drive, runtime
    except ImportError:
        print("not on Colab: no runtime to release")
        return False
    on_drive = str(work_dir()).startswith("/content/drive")
    if not on_drive and not force:
        print(f"⚠️ GPU NOT released: the results are in {work_dir()} on the VM's local disk and would be lost. "
              f"Copy them first, or call nb.release_gpu(force=True).")
        return False
    print(f"all cells finished; releasing the GPU in {wait_seconds:g} s (interrupt this cell to keep the session)")
    time.sleep(wait_seconds)
    if on_drive:
        drive.flush_and_unmount()  # returns when every pending write has reached Drive
        print("Drive is synced")
    print("runtime released: results are on Drive; reconnect and run the notebook again for the next cycle")
    runtime.unassign()
    return True


def notebook_rev(path: str | Path) -> str:
    """Revision of a notebook's CODE cells (sha256, 10 hex digits), with the ``NOTEBOOK_REV = "..."`` line itself
    blanked. Cell 1 stores it; cell 2 compares the running notebook's value with the repo's copy, so an outdated
    copy (e.g. saved to Drive) is noticed. tests/test_ops_tooling.py fails when a cell changes without it."""
    import hashlib
    import json
    import re

    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    code = "\n\0\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code = re.sub(r'(?m)^NOTEBOOK_REV = "[^"]*"', 'NOTEBOOK_REV = ""', code)
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:10]
