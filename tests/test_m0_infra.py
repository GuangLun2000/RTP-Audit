"""M0 infrastructure: config, paths, io/resume, logging, registry discipline, batching, layer access."""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

from rtp import io, paths, registry
from rtp.config import REPO_ROOT, ConfigError, config_from_dict, load_config
from rtp.infer import make_batches, pad_left, run_batched
from rtp.log import RedactSecrets
from rtp.models import get_blocks, load_model


# --------------------------------------------------------------------------- config
def test_shipped_configs_load():
    smoke = load_config("configs/smoke.yaml")
    pilot = load_config("configs/pilot.yaml")
    assert smoke.name == "smoke" and pilot.name == "pilot"
    assert pilot.data.test_categories == ["watch", "beer"]
    assert pilot.data.n_real_per_category == 8 and pilot.data.cards_per_cell == 4
    assert smoke.data.n_real_per_category == 3 and smoke.data.n_placebo_per_category == 3
    assert len(smoke.data.categories) == 2


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"name": "x", "data": {"n_reall_per_category": 3}})
    with pytest.raises(ConfigError, match="unknown top-level"):
        config_from_dict({"name": "x", "dta": {}})


@pytest.mark.parametrize("data", [
    {"evidence_levels": [0, 5]},
    {"handicaps": [1, 2]},
    {"n_real_per_category": 2},
    {"categories": ["a"], "test_categories": ["b"]},
    {"handicap_rating_deltas": [0.1, 0.2, 0.3, 0.4, 0.5]},
])
def test_invalid_data_config(data):
    with pytest.raises(ConfigError):
        config_from_dict({"name": "x", "data": data})


def test_data_hash_ignores_measure_settings():
    a = config_from_dict({"name": "x", "measure": {"shard_size": 10}})
    b = config_from_dict({"name": "x", "measure": {"shard_size": 20}})
    c = config_from_dict({"name": "x", "seed": 1})
    assert a.data_hash == b.data_hash != c.data_hash
    assert a.config_hash != b.config_hash


# --------------------------------------------------------------------------- paths
def test_paths_layout(work_dir):
    assert paths.model_slug("allenai/Llama-3.1-Tulu-3-8B-SFT") == "allenai__Llama-3.1-Tulu-3-8B-SFT"
    sd = paths.stage_dir("pilot", "org/m", "measure")
    assert sd == work_dir / "runs" / "pilot" / "org__m" / "measure"
    assert paths.data_dir("pilot") == work_dir / "runs" / "pilot" / "data"


# --------------------------------------------------------------------------- io
def test_jsonl_roundtrip_is_byte_stable(tmp_path):
    recs = [{"b": 1, "a": "Hermès", "x": [1.5, None]}, {"b": 2, "a": "Peet's", "x": []}]
    io.write_jsonl(tmp_path / "a.jsonl", recs)
    io.write_jsonl(tmp_path / "b.jsonl", recs)
    assert io.file_sha256(tmp_path / "a.jsonl") == io.file_sha256(tmp_path / "b.jsonl")
    assert io.read_jsonl(tmp_path / "a.jsonl") == recs


def test_run_sharded_resumes(tmp_path):
    calls = []

    def process(chunk, idx):
        calls.append(idx)
        if idx == 1 and len(calls) == 2:
            raise RuntimeError("simulated crash in shard 1")
        return pd.DataFrame({"v": list(chunk)})

    records = list(range(25))
    with pytest.raises(RuntimeError):
        io.run_sharded(records, 10, tmp_path, process)
    assert calls == [0, 1]
    io.run_sharded(records, 10, tmp_path, process)
    assert calls == [0, 1, 1, 2]  # shard 0 skipped on resume
    df = io.load_shards(tmp_path)
    assert df["v"].tolist() == records


def test_stage_dir_refuses_config_mismatch(tmp_path):
    d = tmp_path / "stage"
    io.prepare_stage_dir(d, "a: 1\n", "hash-a")
    io.prepare_stage_dir(d, "a: 1\n", "hash-a")  # same hash: fine (resume)
    with pytest.raises(io.StageConfigMismatch):
        io.prepare_stage_dir(d, "a: 2\n", "hash-b")
    io.prepare_stage_dir(d, "a: 2\n", "hash-b", force=True)
    assert (d / "config.yaml").read_text() == "a: 2\n"
    assert not io.is_done(d)
    io.mark_done(d, {"n": 1})
    assert io.is_done(d)


# --------------------------------------------------------------------------- logging
def test_log_redacts_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "supersecretvalue123")
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "token=%s other=hf_%s", ("supersecretvalue123", "a" * 30), None)
    RedactSecrets().filter(rec)
    assert "supersecretvalue123" not in rec.getMessage()
    assert "hf_aaaa" not in rec.getMessage()


# --------------------------------------------------------------------------- registry discipline
_ID_RE = re.compile(r"""["'](?:Qwen|allenai|meta-llama)/[A-Za-z0-9._-]+["']""")


def test_model_and_dataset_ids_only_in_registry():
    offenders = []
    for path in (REPO_ROOT / "rtp").rglob("*.py"):
        if path.name == "registry.py":
            continue
        if _ID_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"hard-coded model/dataset IDs outside rtp/registry.py: {offenders}"


def test_registry_flags():
    assert registry.MODELS["meta-llama/Llama-3.1-8B"].is_chat is False
    assert registry.MODELS["meta-llama/Llama-3.1-8B"].gated is True
    assert registry.MODELS["allenai/Olmo-3-1025-7B"].verified is False
    assert registry.DATASETS["allenai/dolci-3-instruct-dpo-with-metadata"].verified is False
    assert registry.SMOKE_MODEL in registry.MODELS


# --------------------------------------------------------------------------- hard constraint 1
def test_load_model_takes_exactly_one_model_path():
    params = list(inspect.signature(load_model).parameters)
    assert params == ["model_path"]


def test_detect_package_never_mentions_base_loading():
    for path in (REPO_ROOT / "rtp" / "detect").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", src)  # ignore docstrings that state the rule
        assert not re.search(r"base_model|base_path|load_base|weight_diff", code), path


# --------------------------------------------------------------------------- layers
class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_get_blocks_variants():
    assert get_blocks(_NS(model=_NS(layers=[1, 2, 3]))) == [1, 2, 3]
    assert get_blocks(_NS(transformer=_NS(h=[4, 5]))) == [4, 5]
    with pytest.raises(AttributeError):
        get_blocks(_NS(foo=1))


# --------------------------------------------------------------------------- batching
def test_make_batches_limits_and_coverage():
    lengths = [5, 50, 7, 49, 8, 100, 3, 3, 3]
    batches = make_batches(lengths, max_batch_size=3, max_batch_tokens=120)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 120


def test_pad_left_positions():
    ids, mask, pos = pad_left([[7, 8, 9], [5]], pad_id=0)
    assert ids.tolist() == [[7, 8, 9], [0, 0, 5]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert pos.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_run_batched_halves_once_on_oom():
    seen = []

    def fn(batch):
        seen.append(len(batch))
        if len(batch) == 4:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [i * 10 for i in batch]

    out = run_batched([[1]] * 4, fn, max_batch_size=4, max_batch_tokens=100)
    assert out == [0, 10, 20, 30] and seen == [4, 2, 2]

    def always_oom(batch):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_batched([[1]] * 4, always_oom, max_batch_size=4, max_batch_tokens=100)


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_smoke_model_forward(smoke_lm):
    from rtp.smoke import check_forward

    check_forward(smoke_lm)
    assert smoke_lm.is_chat


@pytest.mark.integration
def test_padding_invariance(smoke_lm):
    from rtp.infer import last_token_reduce
    from rtp.models import encode, format_chat

    tok = smoke_lm.tokenizer
    texts = ["Pick a letter.", "Here is a much longer question about running shoes and laptops. Pick one."]
    seqs = [encode(tok, format_chat(tok, t), True) for t in texts]
    top = lambda logits: torch.log_softmax(logits, -1)[:, :50]  # noqa: E731
    batched = last_token_reduce(smoke_lm, seqs, top, max_batch_size=8, max_batch_tokens=10_000)
    single = [last_token_reduce(smoke_lm, [s], top, 1, 10_000)[0] for s in seqs]
    for b, s in zip(batched, single):
        assert abs(b - s).max() < 1e-3


def test_repo_root_is_project():
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert Path(io.__file__).parent.name == "rtp"
