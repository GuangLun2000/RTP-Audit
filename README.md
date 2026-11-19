# rtp-audit

Research code for separating the CONTENT of a model's brand preference from its SCALE.

When two models, or two training stages of one model, differ on a propensity metric (choice share, flip rate,
concentration of recommendations), two things are mixed: what the model prefers, and how decisively it says so.
Post-training raises decisiveness, so a rise in a raw metric is not by itself a change of preference. This
repository measures, inside each open-weight model:

- the **name effect** Δ of a brand: how much its name alone moves a choice against a fictional control (logits);
- the **anchor curve** A(d): how much a graded piece of evidence (a lower rating, a higher price, fewer reviews)
  moves a choice between two FICTIONAL options (logits per step);
- the **preference level** θ = A⁻¹(Δ): how much adverse evidence it takes to cancel the name, in evidence units
  ("worth 0.6 stars"). Multiplying every logit by a constant changes Δ and A and leaves θ unchanged.

Between two stages (SFT → DPO → RLVR of Tulu 3 and Olmo 3) it then reports a scale multiplier per evidence type,
the content residual θ_B − θ_A per brand, and a test of the null model "post-training = uniform sharpening".

Status: **batch 1** of instrument v2 is implemented (paired ladder items, the letter-logit readout, model
variants, the estimators, experiment cycles with a budget). Batch 2 (content-neutral and dose-planting LoRA
runs: noise band, dose-response) and batch 3 (instrument freeze, the single unseal, prediction) are not
implemented yet. The legacy four-option measurement and the white-box stages (activations, probes,
localisation) of the earlier framing are kept, frozen: see the last section.
See `DECISIONS.md` for the design record, `docs/CYCLES.md` for the experiment plan and log, and
`docs/AGENT_LOOP.md` for the Colab → feedback bundle → coding agent loop.

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

## How an experiment runs

Experiments are **cycles**: one question, a job list, at most 12 hours on one GPU (G4 / RTX PRO 6000, 96 GB),
one scorecard. A cycle file (`configs/cycles/c0.yaml`) names the run config, the jobs in priority order and the
model pairs to compare.

```bash
python -m rtp.cli build-data --config configs/e1.yaml
python -m rtp.cli cycle plan --cycle configs/cycles/c0.yaml   # prices every job; refuses a cycle over budget
python -m rtp.cli cycle run  --cycle configs/cycles/c0.yaml   # runs the jobs, analysis and scorecard
python -m rtp.cli feedback --config configs/e1.yaml           # bundle for the coding agent
```

The instrument is developed on the models under `models` in the config (Tulu 3 SFT and DPO). The models under
`confirm_models` (Tulu 3 final = the RLVR stage, and the Olmo 3 stages) and the sealed test categories are
measured only after the instrument is frozen.

## Commands

All stages run through `python -m rtp.cli <stage>`. The global options `--work-dir DIR`, which overrides
`$RTP_WORK_DIR`, and `--log-level` go before the stage name.

### `smoke`

```bash
python -m rtp.cli smoke
```

Runs instrument v2 end to end on `Qwen/Qwen2.5-0.5B-Instruct` with `configs/smoke.yaml` (2 categories, 3 real
brands and 3 fictional names each, two-step ladders):

1. One forward pass through the chat template.
2. `build-data` (legacy and ladder items); checks that the test split refuses to open and that regeneration is
   byte-identical.
3. One cycle: the `ladder` stage on the model and on its pure-scale variant `@scale=2`, the `open-set` stage, the
   analysis and the scorecard.
4. The scale identity: the variant's scale multipliers must all be 2 and no preference level may move.

It starts from a clean `runs/smoke/` directory; `--resume` reuses finished outputs.

### `build-data`

```bash
python -m rtp.cli build-data --config configs/e1.yaml
```

Generates the model-independent data once per config in `$RTP_WORK_DIR/runs/<name>/data/`:

- `catalogue.jsonl`: real brands and fictional names, with split and natural-negative role.
- `dev/` and `test/`: `ladder_core.jsonl`, `ladder_ext.jsonl`, `ladder_ctx.jsonl` (with `ladder.enabled`), and the
  legacy `choice_items.jsonl` and `probe_items.jsonl`. `test/` is sealed and is read only with `--unseal`.
- `panel_stats.json`: items, units and prompt characters per tier (what `cycle plan` prices).
- `cell_counts.csv`, `data_report.md`, `data_manifest.json` (file hashes).

A ladder item shows two options, a target and a fictional control, and asks for one letter. The target is a
real brand (family `real`) or a fictional name (family `anchor`). Three things are crossed:

- **stratum** (information sufficiency): `name` (names only), `attr` (only the manipulated attribute), `full`
  (a full card), `comp` (a full card plus a requirement stated by the user);
- **evidence type**: `rating`, `price`, `reviews`, each with its own ladder in natural units, and `composite`;
- **level `d`**: 0 = equal information; d > 0 = the target is worse by step d; d < 0 = the control is.

Tiers: `core` (names and attribute-only; every brand at d = 0; a regular subset of the anchor pairs on the
ladders), `ext` (the other anchor pairs, full cards, the composite ladder, d ≠ 0 items for a few brands per
category), `ctx` (the core comparison with two real competitors added: a four-option robustness arm).
`configs/e1.yaml`: 18 categories × 18 brands, 8 fictional names per category; dev has 40,216 core, 65,824 ext
and 14,080 ctx items.

The same seed and the same seed files always give byte-identical files. A finished data dir is reused. If the
data config changed, the command refuses to run; pass `--force` to rebuild or use a new config `name`. With
`ladder.enabled` the build refuses test categories that share a real brand with a dev category.

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
  option letters. After editing seeds, run the command below; it replaces only unusable names, keeps the others
  together with their review flags, and creates pools for categories added to categories.yaml:

  ```bash
  python -m rtp.cli make-placebos --repair
  ```

### `ladder`

```bash
python -m rtp.cli ladder --config configs/e1.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT --tier core
```

One model id in, one table out (`runs/<name>/<model>/ladder_<tier>/scores.parquet`). For every item it stores
the logit of each option's letter at the forced position (after the `Answer:` prefill) in two token forms, and
the letter probability mass at the forced and the free position. Logits are recomputed in float32 from the
final hidden state (bf16 rounds a logit near 20 to a grid of 0.125).

The preference level uses only the letter-logit difference target − control. Diagnostics on deterministic
samples check that this readout is valid: answer-path masses (letters and names) at both positions, the greedy
free answer and a few sampled free answers. The summary warns when the letter mass at the forced position is
below 0.9.

`--model` also takes a **variant**: `<checkpoint>@scale=2` (every output logit multiplied by 2: a pure scale
change) or `zoo:<run_id>` (the run's base checkpoint with its LoRA adapter). A confirmation model is refused
until the instrument is frozen, by this and by every other command that loads a model.

### `open-set`

```bash
python -m rtp.cli open-set --config configs/e1.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

"Name a laptop brand. Reply with the name only." For every category and prompt type: the exact probability
that the reply starts with each catalogue brand (answer-path trie; `coverage` is the total mass on catalogue
brands), its effective number of options N_eff = exp(entropy), and sampled answers on a temperature grid.

### `analyze`

```bash
python -m rtp.cli analyze --config configs/e1.yaml --pair allenai/Llama-3.1-Tulu-3-8B-SFT allenai/Llama-3.1-Tulu-3-8B-DPO
```

CPU only; a rebuildable view under `runs/<name>/analysis/`, never a hashed stage. Per model: unit values, anchor
curves, the pulls of the fictional names, name effects and preference levels per brand (converted and direct),
raw scale-dependent metrics, and the instrument checks. Per pair: scale multipliers per evidence type, the
pooled null statistics with intervals, content residuals. `cycle run` calls it after every job.

### `cycle`

```bash
python -m rtp.cli cycle plan   --cycle configs/cycles/c0.yaml
python -m rtp.cli cycle run    --cycle configs/cycles/c0.yaml
python -m rtp.cli cycle report --cycle configs/cycles/c0.yaml   # re-run the analysis, rewrite the scorecard
```

- `plan` prices every job with the cost model (seconds per prompt token and per item plus model load,
  recalibrated on every finished stage) and exits with code 5 when the remaining work exceeds the budget. No
  cycle file may budget more than 12 hours.
- `run` executes the jobs in order, each stage in its own process, and rewrites
  `runs/<name>/cycles/<cycle>/scorecard.md` after every job: the budget ledger (predicted vs actual), the
  instrument checks per model, the stage pairs, and what changed since the previous cycle. Jobs that no longer
  fit are deferred; re-running continues the cycle.
- A cycle never mixes results. If a job's existing stage was built by other code, items or settings, `plan` and
  `run` stop before any GPU time; `--redo-stale` (develop phase only) measures those jobs again from scratch.

### `status`, `doctor`, `feedback` (operations)

```bash
python -m rtp.cli status --config configs/e1.yaml
python -m rtp.cli doctor --config configs/e1.yaml
python -m rtp.cli feedback --config configs/e1.yaml
```

- `status`: every stage of every model, shown as done / partial / partial+stale / stale / not started, with its
  warnings. Test-split results stay hidden unless `--unseal` is passed.
- `doctor`: GPU, bf16, library pins, whether the HF token is set, disks, Drive, data, and Hub access to the
  configured models.
- `feedback`: writes `runs/<config>/feedback/feedback_<UTC>.md` for a coding agent (see `docs/AGENT_LOOP.md`),
  starting with the newest cycle scorecards and including the head of every finished dev summary.

Every command also appends `start`/`done`/`error` events (errors with tracebacks) and all warnings to
`runs/<config>/events.jsonl`.

`--set SECTION.KEY=VALUE` (repeatable, before the stage name) overrides a config value, for example
`python -m rtp.cli --set measure.max_batch_size=128 measure ...`.

Progress is reported as one log line per ~10% with ETA; `RTP_PROGRESS=tqdm` gives bars in a terminal.

### Resuming

Long stages write `shard_XXXXX.parquet` plus a `.done` marker every `shard_size` prompts and end with `_DONE` and
`summary.md`. Re-running the same command skips finished shards and finished stages.

A stage dir is keyed by stage settings, model, split, prompt mode, the content of the files it reads, and the
source code it imports. Any code change in a stage's import closure invalidates it. The stage metadata records the
git commit, library versions, device and model revision. A half-finished stage refuses to resume under a different
torch, transformers or tokenizers version, GPU or model revision. A stage dir from a
different configuration is refused, whether finished or not, with exit code 4; pass `--force` to redo it. The analysis
leaves such stale stages out and says so.

Editing a seed file's content (brands, aliases, ranges, placebo names) therefore requires `build-data --force`
followed by `--force` on the model stages. Ticking `needs_review` or `needs_web_check` does not.

Batch limits (`max_batch_size`, `max_batch_tokens`) can change between resumes. On CUDA OOM a batch is halved once.

Exit codes: 0 ok, 2 command not implemented yet, 3 sealed split without `--unseal`, 4 configuration, data, phase or
stage-identity error (the message says which), 5 a cycle does not fit its budget, 6 the session's deadline was
reached between two shards (resumable), 7 a cycle ended with deferred or failed jobs.

## Colab

Open the notebook straight from GitHub: [Open in Colab](https://colab.research.google.com/github/GuangLun2000/RTP-Audit/blob/main/colab/run_cycle.ipynb). Cell 1 holds the settings (`CYCLE`, `CONFIG`,
`REPO_URL`, `PIN`, `USE_DRIVE`); Drive is used only for results.

1. Mount Drive and set the environment.
2. Clone the code from GitHub and install it. `PIN` is a branch (updated to its newest push) or a commit hash
   (use a commit for confirmation cycles); the cell stops with ❌ on any failing step, refuses a changed torch
   and warns when the notebook is outdated.
3. `doctor` preflight.
4. Smoke and the tests.
5. `build-data`.
6. `cycle plan`: the priced job list.
7. `cycle run`, then `status`. Re-run this cell to resume.
8. The scorecard.
9. **Feedback bundle** for the coding agent.

Stages run through `rtp.notebook`, which streams live output, marks warnings and errors, keeps going when one job
fails, and keeps a session history.

## Legacy four-option family and frozen white-box stages

The commands below belong to the earlier framing (a single-model audit of undisclosed promotion). They are
kept, with their tests, and are not part of any cycle, the smoke run or the notebook. In the legacy items
`evidence_level` 0–4 is information sufficiency and `handicap` is the only adverse evidence; the O5 estimator
described here is not invariant to a scale change and is not used for preference levels. `configs/pilot.yaml`
is the config of the first pilot.

### `measure` (legacy)

```bash
python -m rtp.cli measure --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Scores every dev choice item with the answer-path estimator (`rtp/measure/answers.py`, DECISIONS.md fourth round).

- The chat template is applied (base checkpoints get plain text plus `\nAnswer:`). Each item is read at two
  positions from one forward pass: the free assistant start (spec position) and after a prefilled "Answer:".
- At each position, every way of starting an answer that names an option is summed into that option: letter forms
  ("B", " B", "**B", "(B", ...) and the option's displayed brand name. The target slot's absent identity is a
  separate `identity_leak` class.
- `p_A..p_D` and `s_target = log(p/(1-p))` (p clipped to [1e-4, 1-1e-4]) are the O5 estimator: immediate free
  answers, with the deferred share imputed by the committed answer after "Answer:". O3 (free only), O4 (forced
  only), coverage, name share, leak and imputation-free bounds are stored per item.
- 4 cyclic rotations per item are averaged. Name effect = real-name minus placebo-name score within each `pair_id`.
  Cards are matched across evidence levels and handicaps (`card_set`).
- A fixed 2% sample is decoded greedily from the free position. Immediate answers (option named within 4 tokens)
  are compared with the free argmax; the others are continued to 48 tokens and their concluding answer is compared
  with the forced argmax. Agreement with the O5 argmax below 90% triggers a warning; chance levels are reported.

Outputs in `runs/<config>/<model_slug>/measure/`:

- `scores.parquet`: `item_id, model_id, p_A..p_D, s_target, gen_letter`, plus `argmax_letter, argmax_free,
  argmax_forced, s_target_free, s_target_forced, s_target_lo, s_target_hi, c_free, c_forced, name_share_*,
  leak_*, article_free, pruned_*, gen_text, gen_mode, s_target_solo`. `s_target_solo` (2% sample) is the same
  item scored alone: the difference from `s_target` is the numerical noise floor of bfloat16 batching, reported in
  the summary and `numeric_noise.json` next to the size of the name effects. `gen_refusal` flags checked free
  answers that open with a refusal ("I'm sorry, but I can't ..."); their rate per evidence level and version is in
  `refusals_by_level.parquet` and the summary.
- `pair_scores.parquet`, `name_effects.parquet`.
- `entity_evidence.parquet`: entity × evidence-level name effects (SEs over placebo substitutes), plus the residual of the
  robust regression through the origin of evidence 4 on evidence 0.
- `estimators_by_entity.parquet` (O5 / O3 / O4), `coverage_by_level.parquet`, `answer_channel_by_entity.parquet`,
  `answer_format.json` (format shifts, estimator agreement, evidence-0 bounds).
- `handicap_effects.parquet`, `ladder_regression.json` (spec variant plus handicap-0 and format-baseline variants
  with bootstrap CIs), `consistency.json`.
- `evidence_sensitivity.json`: within-card-set handicap response per star, the manipulation ratio (placebo per-star
  drop / median |ne_e0|; warning below the provisional 0.25), and the across-pair OLS on card gaps.
- `summary.md`, `_DONE`.

`--split test --unseal` measures the sealed test split, writing to `measure_test/`.

Registered base checkpoints such as `meta-llama/Llama-3.1-8B` are measured automatically with plain-text
continuation. For an unregistered or local path the prompt mode is only guessed, with a warning; set it with
`--prompt-mode base` or `--prompt-mode chat`. Qwen3's thinking mode is switched off through its template (registry
`chat_template_kwargs`).

### `format-probe` (legacy)

```bash
python -m rtp.cli format-probe --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Scores a balanced 40-item dev sample (`--n-items`) like `measure` and decodes every sampled item greedily. It
prints the tails of the rendered prompts, coverage at both positions, name share, the free and forced argmax and
the parsed free answer per item, and writes `runs/<config>/format_probe/<model>/summary.md`. A few minutes on a
GPU; not a stage (nothing is cached).

### `open-ended` (legacy)

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

### `report` (legacy)

```bash
python -m rtp.cli report --config configs/pilot.yaml
```

Collects every finished `measure` / `open-ended` stage of the config into `runs/<config>/report.md` and also
prints it. The report covers consistency rates, per-entity name effects at evidence 0 and 4 for each model, and
the Named and Picked shares across checkpoints. Pass `--unseal` to include test-split stages.

### `extract` (frozen)

```bash
python -m rtp.cli extract --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Extracts float16 activations `[n_items, n_layers, d_model]` with `output_hidden_states=True` at hidden-state
layers 0, `extract.layer_stride`, 2 × stride, … below the number of blocks (the post-final-norm state is excluded). Prompts are rendered exactly as
they are measured. The extracted positions:

- probe items: the last token of the read brand-name mention;
- evidence-0 choice items (rotation 0 by default): the last token of the target name, and the last input token.

The expected size is logged before writing. For the pilot on an 8B model (stride 2: 16 layers) it is about 6.0 GB: 7,920
probe items (1.0 GB) and 19,008 evidence-0 choice items at 2 positions (5.0 GB).

### `probe` (frozen, dev split only)

```bash
python -m rtp.cli probe --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Trains ridge probes for each frame (`background` = spec text; `choice` = the background followed by the choice
prompt, read at the option line), each layer and each attribute, and evaluates them in four ways:

- grouped CV by entity;
- the control task (labels permuted across entities);
- held-out dev categories (`probe.heldout_categories`), within the same frame and across frames;
- binding controls: two-brand texts and undescribed names.

It exports `K_{frame}_L{layer}.npy` for every layer, at rank ≤ `k_rank`. K is an orthonormal basis of the probes'
Haufe encoding patterns, then the PCs of entity-averaged level means, then the probe weights.

`summary.md` starts with **Readiness for M4**. Each check is compared with `probe.n_null_k` (pilot 99) shuffled-label
nulls built the same way, with per-layer permutation p-values and a max-over-layers family-wise p; the embedding
layer is excluded. It warns when:

- probes do not beat the control task;
- binding is not shown in both description orders;
- K captures no more of the real-minus-placebo activation difference than a shuffled-label K;
- the price-tier read-out does not rank real brands by tier within categories;
- the popularity read-out separates real brands from placebos no better than a shuffled probe (familiarity).

### `localize` (frozen, dev split only)

```bash
python -m rtp.cli localize --config configs/pilot.yaml --model allenai/Llama-3.1-Tulu-3-8B-SFT
```

Where is the name effect carried? Full activation transplants on evidence-0 pairs (`localize.pairs_per_entity`
per entity, `localize.rotations`), knock-in (placebo run ← real residual, spec §7.6) and knock-out, at the target
name's last token, every post-name token, the final token, and a band of `localize.band` layers, at every
`localize.layer_stride`-th layer (default: `extract.layer_stride`; it must be a multiple of it, so that the chosen
layer has a K). `recovery.parquet` and `summary.md` give the pooled recovery with an entity
bootstrap CI, the between-entity R², recovery per token-count class, the spec layer (highest knock-in recovery at
the name's last token; warning below 0.8) and the pre-registered alternative layer. Cost: about
2 × (4 × layers) + 2 scored runs per unit; the pilot settings give 1,584 units (198 dev entities × 4 pairs ×
2 rotations).

## Tests

```bash
pytest -q
```

Unit tests use synthetic data and run on CPU. They cover:

- instrument v2: ladder items (determinism, the crossing of names, both directions on the ladders, only the
  manipulated attribute differing, the sealed tier files), model variants and the adapter zoo, the answer-path
  estimator for N options, the cycle files, the budget cap, the cost model and its recalibration, the
  develop/confirm guard, the deadline, the scorecard;
- the estimators (`rtp/scale`) on synthetic scores: exact invariance of the preference level to a scale change,
  cancellation of position bias, recovery of the scale multiplier, the null statistics and their error rate;
- the legacy four-option family: the answer-path estimator, deferred-answer parsing, matched card sets, the
  handicap response, robust regression, brand matching, and a golden hash that keeps its items byte-identical;
- the frozen white-box code: ridge probes, the control task, the subspace K, the localisation estimators;
- the operations tooling: status, feedback bundles (including secret redaction), events, overrides, the notebook.

Tests marked `integration` load the real smoke model (the float32 letter readout against the model's logits, a
LoRA adapter round trip through the zoo, the ladder and open-set stages end to end) and are skipped only when
it cannot be loaded (no network and no cache). Any other error fails. Set `RTP_REQUIRE_INTEGRATION=1` to turn
those skips into failures. To run only the unit tests:

```bash
pytest -q -m "not integration"
```
