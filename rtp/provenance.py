"""Provenance: which code and which environment produced a result.

* ``code_digest(module)`` hashes the source of ``module`` and every ``rtp`` module it imports
  (transitively, found by parsing import statements, not by importing). Changing any of that
  code changes the digest, so a finished or half-finished stage produced by older code is
  never silently reused or resumed. Pure plumbing modules whose effect is already captured by
  the serialised settings (config, io, paths, log, provenance) are excluded.
* ``environment()`` records git commit, library versions and device for stage metadata.
"""

from __future__ import annotations

import ast
import hashlib
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
PLUMBING = frozenset({"rtp.config", "rtp.io", "rtp.paths", "rtp.log", "rtp.provenance", "rtp"})


def _module_file(name: str) -> Path | None:
    rel = Path(*name.split("."))
    for cand in (REPO_ROOT / rel.with_suffix(".py"), REPO_ROOT / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _imports(path: Path, own: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    pkg = own if path.name == "__init__.py" else own.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "rtp")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative import
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
            if base.split(".")[0] != "rtp":
                continue
            found.add(base)
            for a in node.names:  # "from rtp.data import schema" imports a submodule
                if _module_file(f"{base}.{a.name}") is not None:
                    found.add(f"{base}.{a.name}")
    return found


def import_closure(module: str) -> list[str]:
    seen: set[str] = set()
    todo = [module]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_file(name)
        if path is not None:
            todo.extend(_imports(path, name) - seen)
    return sorted(m for m in seen if _module_file(m) is not None)


@lru_cache(maxsize=None)
def code_digest(module: str) -> str:
    h = hashlib.sha256()
    for name in import_closure(module):
        if name in PLUMBING:
            continue
        path = _module_file(name)
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def environment() -> dict[str, Any]:
    """Informational record: git commit (+ dirty flag), Python/library versions, device."""
    import importlib.metadata as md

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    dirty = _git("status", "--porcelain", "--", "rtp", "configs", "data")
    env: dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        **{pkg: ver(pkg) for pkg in ("torch", "transformers", "tokenizers", "numpy", "pandas")},
    }
    try:
        import torch

        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu-{platform.machine()}"
    except Exception:  # noqa: BLE001 - informational only
        env["device"] = None
    return env


# Environment fields that must not change while a stage is half-finished (results would mix numerics)
RESUME_CRITICAL = ("torch", "transformers", "tokenizers", "device")
