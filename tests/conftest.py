from __future__ import annotations

import pytest

_SMOKE_LM = None
_SMOKE_ERR: str | None = None


@pytest.fixture()
def work_dir(tmp_path, monkeypatch):
    """Point $RTP_WORK_DIR at a fresh temp dir."""
    wd = tmp_path / "work"
    monkeypatch.setenv("RTP_WORK_DIR", str(wd))
    return wd


@pytest.fixture(scope="session")
def smoke_lm():
    """The real smoke model, or skip when it cannot be loaded (no network / no cache)."""
    global _SMOKE_LM, _SMOKE_ERR
    if _SMOKE_LM is None and _SMOKE_ERR is None:
        try:
            from rtp import registry
            from rtp.models import load_model

            _SMOKE_LM = load_model(registry.SMOKE_MODEL)
        except Exception as exc:  # noqa: BLE001
            _SMOKE_ERR = f"{type(exc).__name__}: {exc}"
    if _SMOKE_LM is None:
        pytest.skip(f"smoke model unavailable: {_SMOKE_ERR}")
    return _SMOKE_LM
