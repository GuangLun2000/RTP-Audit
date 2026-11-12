# rtp-audit

Research code for auditing brand preferences in open-weight LLMs. Given a single model, with no training history and
no clean base model, it asks whether the model's preference for a brand includes *undisclosed promotion*:
preference attached to the brand's identity rather than to what the model knows about the brand.

Status: **M0 (infrastructure)**, **M1 (data generation)** and **M2 (behavioural measurement)** are implemented. M3+
(activations, probes, transplant, residuals, planting, evaluation) are not implemented yet. See `DECISIONS.md` for design choices not fixed by the spec.

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

Runs the full pipeline end to end on `Qwen/Qwen2.5-0.5B-Instruct` with `configs/smoke.yaml` (2 categories, 3 real
and 3 placebo brands each):

1. One forward pass through the chat template.
2. `build-data`.
3. Checks that the test split refuses to open and that regeneration is byte-identical.
4. `measure`, `open-ended` and `report`.

It prints per-step timings and the letter-argmax vs generation agreement. It starts from a clean `runs/smoke/`
directory; `--resume` reuses finished outputs. Measured time: about 2.5 min on an Apple M4 CPU; the 10-minute
budget in the spec is for a GPU.

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
  `needs_review: true`. It also holds:
  - `open_ended_only`: draft brands for the 16 Appendix A categories without choice items.
  - `extra_known_brands`: used only by the placebo conflict check.
- `data/seeds/placebo_names.yaml`: generated fictional names, every one marked `needs_web_check: true`. Regenerate
  with:

  ```bash
  python -m rtp.cli make-placebos
  ```

  Each name passes an automatic conflict check (case-insensitive, including substrings) against every known real
  brand, and that check runs again on every build. Names never start with A–D, which could collide with the
  option letters. After editing seeds, run the command below; it replaces only unusable names and keeps the others
  together with their review flags:

  ```bash
  python -m rtp.cli make-placebos --repair
  ```

### `measure`

```bash
python -m rtp.cli measure --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Scores every dev choice item by option letter.

- The chat template is applied (base checkpoints get plain text plus `\nAnswer:` instead). Every token whose
  decoded text equals A, B, C or D after stripping whitespace is merged with logsumexp, and the result is
  renormalised over the four letters.
- `s_target = log(p/(1-p))`, with `p` clipped to [1e-4, 1-1e-4].
- 4 cyclic rotations per item are averaged. Name effect = real-name minus placebo-name score within each `pair_id`.
- A fixed 2% sample is decoded greedily (≤ 4 tokens) to check agreement with the letter argmax. Agreement below
  90% triggers a warning in the summary.

Outputs in `runs/<config>/<model_slug>/measure/`:

- `scores.parquet`: `item_id, model_id, p_A..p_D, s_target, gen_letter`, plus `argmax_letter, letter_mass,
  name_mass_target, name_mass_other, target_name_collision, s_target_credit, gen_text`. Chat prompts end with a
  prefilled assistant turn "Answer:", so the scored token is the answer letter rather than a brand name.
- `pair_scores.parquet`, `name_effects.parquet`.
- `entity_evidence.parquet`: entity × evidence-level name effects, plus the residual of the robust regression
  through the origin of evidence 4 on evidence 0.
- `handicap_effects.parquet`, `ladder_regression.json` (spec variant plus handicap-0 and format-baseline variants
  with bootstrap CIs), `consistency.json`.
- `evidence_sensitivity.json`: does the model respond to the rating, price and review gaps at all?
- `name_mass_by_entity.parquet` and `entity_evidence_credit.parquet`: name-answer diagnostics.
- `summary.md`, `_DONE`.

`--split test --unseal` measures the sealed test split, writing to `measure_test/`.

Registered base checkpoints such as `meta-llama/Llama-3.1-8B` are measured automatically with plain-text
continuation. For an unregistered or local path the prompt mode is only guessed, with a warning; set it with
`--prompt-mode base` or `--prompt-mode chat`. Qwen3's thinking mode is switched off through its template (registry
`chat_template_kwargs`).

### `open-ended`

```bash
python -m rtp.cli open-ended --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Runs the Name, Choose and Recommend prompts ("Reply with the name only.") with `n_samples` answers at temperature
1, then matches each answer to a brand. It also runs the fixed-list pick: three brands, "Which one would you
pick?", with the reply prefilled as "I'd pick". Each brand's continuation log-probability (summed over tokens,
aliases merged) is normalised over the list and averaged over both list orders. Base checkpoints use text
continuation ("If you want a good {category}, I recommend").

`shares.parquet` and `summary.md` give the Named and Picked shares of the Appendix A brands for each category and
prompt type. With `open_ended.all_natural_negatives: true` (as in the pilot config), all 26 Appendix A categories
are measured. Categories without choice items take their brands from the `open_ended_only` section of
`categories.yaml` and are always dev. Test categories of the run still need `--split test --unseal`. Answers are
matched against every brand of the category in the seed snapshot stored with the data.

### `report`

```bash
python -m rtp.cli report --config configs/pilot.yaml
```

Collects every finished `measure` / `open-ended` stage of the config into `runs/<config>/report.md` and also
prints it. The report covers consistency rates, per-entity name effects at evidence 0 and 4 for each model, and
the Named and Picked shares across checkpoints. Pass `--unseal` to include test-split stages.

### Resuming

Long stages write `shard_XXXXX.parquet` plus a `.done` marker every `shard_size` prompts and end with `_DONE` and
`summary.md`. Re-running the same command skips finished shards and finished stages.

A stage dir is keyed by stage settings, model, split, prompt mode, the content of the files it reads, and the
source code it imports. Any code change in a stage's import closure invalidates it. The stage metadata records the
git commit, library versions, device and model revision. A half-finished stage refuses to resume under a different
torch, transformers or tokenizers version, GPU or model revision. A stage dir from a
different configuration is refused, whether finished or not, with exit code 4; pass `--force` to redo it. `report`
leaves such stale stages out and lists them.

Editing a seed file's content (brands, aliases, ranges, placebo names) therefore requires `build-data --force`
followed by `--force` on the model stages. Ticking `needs_review` or `needs_web_check` does not.

Batch limits (`max_batch_size`, `max_batch_tokens`) can change between resumes. On CUDA OOM a batch is halved once.

Exit codes: 0 ok, 2 stage not implemented yet, 3 sealed split without `--unseal`, 4 configuration, data or model
error (the message says which).

## Colab

`colab/run_pilot.ipynb` mirrors spec §13:

1. Mount Drive and set `RTP_WORK_DIR`, `HF_HOME` and `HF_TOKEN` (from Secrets).
2. Clone and install.
3. Run smoke and the tests.
4. `build-data`.
5. `measure` and `open-ended` for each model (re-run the cell to resume).
6. `report`.

## Tests

```bash
pytest -q
```

Unit tests use synthetic data and run on CPU. They cover:

- letter merging and normalisation, and log-odds clipping;
- rotation averaging cancelling a synthetic position bias;
- pairing strictly within `pair_id`;
- robust regression with an outlier;
- brand matching and pick aggregation;
- determinism (file hashes), schema and design invariants, the sealed split, and placebo conflicts.

Tests marked `integration` load the real smoke model and are skipped only when it cannot be loaded (no network
and no cache). Any other error fails. Set `RTP_REQUIRE_INTEGRATION=1` to turn those skips into failures. To run
only the unit tests:

```bash
pytest -q -m "not integration"
```
