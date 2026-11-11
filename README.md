# rtp-audit

Research code for auditing brand preferences in open-weight LLMs. Given a single model, with no training history and
no clean base model, it asks whether the model's preference for a brand includes *undisclosed promotion*:
preference attached to the brand's identity rather than to what the model knows about the brand.

Status: **M0 (infrastructure)** implemented. See `DECISIONS.md` for design choices not fixed by the spec.

## Install

Requires Python ≥ 3.10 and a preinstalled `torch` (Colab ships one; it is intentionally not a dependency, so it is
never reinstalled).

```bash
pip install -e ".[dev]"
```

Environment variables:

| variable | meaning | default |
|---|---|---|
| `RTP_WORK_DIR` | persistent results directory | `/content/drive/MyDrive/rtp` |
| `HF_HOME` | Hugging Face cache; keep it on local disk, not Drive | HF default |
| `HF_TOKEN` | read token, needed for gated Llama repos (accept the licence on the Hub first) | unset |

Never put tokens in code, configs or logs. On Colab, store `HF_TOKEN` in Secrets.

## Commands

All stages run through `python -m rtp.cli <stage>`. The global options `--work-dir DIR`, which overrides
`$RTP_WORK_DIR`, and `--log-level` go before the stage name.

### `smoke`

```bash
python -m rtp.cli smoke
```

Loads `Qwen/Qwen2.5-0.5B-Instruct` and runs one forward pass through its chat template. It starts from a clean
`runs/smoke/` directory; `--resume` reuses finished outputs.

## Tests

```bash
pytest -q
```

Unit tests use synthetic data and run on CPU. Tests marked `integration` load the real smoke model and are skipped
automatically when it cannot be loaded (no network and no cache).
