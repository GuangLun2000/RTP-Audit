# rtp-audit

Research code for auditing brand preferences in open-weight LLMs. Given a single model, with no training history and
no clean base model, it asks whether the model's preference for a brand includes *undisclosed promotion*:
preference attached to the brand's identity rather than to what the model knows about the brand.

Status: **M0 (infrastructure)** and **M1 (data generation)** implemented. See `DECISIONS.md` for design choices not fixed by the spec.

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

Loads `Qwen/Qwen2.5-0.5B-Instruct`, runs one forward pass through its chat template, builds the smoke data
(`configs/smoke.yaml`: 2 categories, 3 real + 3 placebo brands each), checks that the test split refuses to open, and
checks that regeneration is byte-identical. It starts from a clean `runs/smoke/` directory; `--resume` reuses
finished outputs.

### `build-data`

```bash
python -m rtp.cli build-data --config configs/pilot.yaml
```

Generates the model-independent data once per config in `$RTP_WORK_DIR/runs/<name>/data/`:

- `catalogue.jsonl`: real and placebo brands, with split and natural-negative role.
- `dev/` and `test/`: `choice_items.jsonl` and `probe_items.jsonl`. `test/` is sealed and is read only with
  `--unseal`.
- `cell_counts.csv` and `data_report.md`: sample counts per split × category × evidence × handicap × name.
- `data_manifest.json`: file hashes.

The same seed and the same seed files always give byte-identical files. A finished data dir is reused. If the
data config changed, the command refuses to run; pass `--force` to rebuild or use a new config `name`.

### Seed files (need human review)

- `data/seeds/natural_negatives.csv`: Appendix A, verbatim. Do not edit.
- `data/seeds/categories.yaml`: drafted brands, tiers, aliases, card ranges and texts. Every category is marked
  `needs_review: true`.
- `data/seeds/placebo_names.yaml`: generated fictional names, every one marked `needs_web_check: true`. Regenerate
  with:

  ```bash
  python -m rtp.cli make-placebos
  ```

  Each name passes an automatic conflict check (case-insensitive, including substrings) against every known real
  brand, and that check runs again on every build.

## Tests

```bash
pytest -q
```

Unit tests use synthetic data and run on CPU. Tests marked `integration` load the real smoke model and are skipped
automatically when it cannot be loaded (no network and no cache).
