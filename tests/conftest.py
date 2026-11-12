from __future__ import annotations

import os

import pytest

_SMOKE_LM = None
_SMOKE_ERR: str | None = None
_SMOKE_LM_FP32 = None


@pytest.fixture()
def work_dir(tmp_path, monkeypatch):
    """Point $RTP_WORK_DIR at a fresh temp dir."""
    wd = tmp_path / "work"
    monkeypatch.setenv("RTP_WORK_DIR", str(wd))
    return wd


@pytest.fixture(scope="session")
def smoke_lm():
    """The real smoke model. Skips ONLY when the model cannot be fetched/loaded (ModelLoadError: no network,
    no cache); any other exception is a bug and fails. Set RTP_REQUIRE_INTEGRATION=1 to fail instead of skip."""
    global _SMOKE_LM, _SMOKE_ERR
    from rtp import registry
    from rtp.models import ModelLoadError, load_model

    if _SMOKE_LM is None and _SMOKE_ERR is None:
        try:
            _SMOKE_LM = load_model(registry.SMOKE_MODEL)
        except ModelLoadError as exc:
            _SMOKE_ERR = str(exc)
    if _SMOKE_LM is None:
        if os.environ.get("RTP_REQUIRE_INTEGRATION") == "1":
            pytest.fail(f"smoke model unavailable but RTP_REQUIRE_INTEGRATION=1: {_SMOKE_ERR}")
        pytest.skip(f"smoke model unavailable: {_SMOKE_ERR}")
    return _SMOKE_LM


@pytest.fixture(scope="session")
def smoke_lm_fp32(smoke_lm):
    """The smoke model in float32 on the same device. Tests that compare two exact computation paths (KV-cache
    extension vs full forwards, self-transplant vs clean run) need it: on a GPU the stages load bfloat16
    (spec §3), and bf16 differences between batch shapes (|dq| up to ~0.1) would fail them for numerical, not
    logical, reasons."""
    global _SMOKE_LM_FP32
    import torch

    if next(smoke_lm.model.parameters()).dtype == torch.float32:
        return smoke_lm
    if _SMOKE_LM_FP32 is None:  # a float32 copy (the checkpoint is bf16, so upcasting loses nothing);
        import copy             # load_model itself keeps exactly one argument (hard constraint 1)
        from dataclasses import replace

        _SMOKE_LM_FP32 = replace(smoke_lm, model=copy.deepcopy(smoke_lm.model).float().eval())
    return _SMOKE_LM_FP32
