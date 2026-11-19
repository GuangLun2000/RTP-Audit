# The improvement loop: cycle on Colab → feedback bundle → coding agent → next cycle

The code base and the instrument are meant to get better with every GPU session. A session is one **cycle**:
one question, a job list, a budget of at most 12 hours on one G4 GPU (RTX PRO 6000, 96 GB), one scorecard.
Short cycles are the point: each one ends with a decision about what to adjust, and the next one tests it.

```
 ┌──────────────────┐  live output, events.jsonl, scorecard  ┌──────────────────────┐
 │ cycle on Colab   │ ─────────────────────────────────────▶ │ feedback bundle (.md) │
 └──────▲───────────┘                                        └──────────┬───────────┘
        │ git pull; re-run: finished stages are reused when             │ given to the agent
        │ their code, inputs and settings are unchanged                 ▼
        │                                                    ┌──────────────────────┐
        └─────────────────────────────────────────────────── │ coding agent: fixes, │
            next cycle file, or the same one resumed         │ tests, decisions,    │
                                                             │ docs/CYCLES.md entry │
                                                             └──────────────────────┘
```

## 1. Run (PI, on Colab)

Open `colab/run_cycle.ipynb` straight from GitHub ([Open in Colab](https://colab.research.google.com/github/GuangLun2000/RTP-Audit/blob/main/colab/run_cycle.ipynb)), so the newest pushed notebook is the one
that runs. Settings are in cell 1: `CYCLE` (the cycle file), `CONFIG` (its run config), `PIN` (branch or commit),
`USE_DRIVE`. Drive holds only results (`RTP_WORK_DIR`), so finished shards survive a disconnect.

1. Cells 1–5: environment, code, `doctor`, smoke and tests, `build-data`.
2. Cell 6, `cycle plan`: every job with its state and predicted time. The estimate comes from a cost model
   (`rtp/cost.py`: seconds per prompt token and per item, plus model load) that recalibrates itself on every
   finished stage. A cycle whose remaining work exceeds its budget is refused here, before any GPU time is spent.
3. Cell 7, `cycle run`: the jobs run in the order of the cycle file (most informative first), each stage in its
   own process. After every job the CPU analysis runs and the scorecard is rewritten, so an interrupted
   session still leaves usable results. A job that no longer fits the remaining budget is deferred; a stage that
   reaches the deadline stops between two shards and resumes later. Re-running cell 7 continues the cycle.
   A cycle never mixes results: when a job's existing stage was built by other code, items or settings (stale),
   or was interrupted under another GPU or library version, cell 6 stops before any GPU time and names the jobs.
   For the first session after an instrument change set `AFTER_INSTRUMENT_CHANGE = True` in cell 1: it rebuilds
   the items (`build-data --force`) and re-measures those jobs from scratch (`cycle run --redo-stale`).
4. Cell 8 shows the scorecard; cell 9 writes the feedback bundle.
5. Cell 10 releases the GPU: it waits 10 seconds, lets Drive finish writing and ends the session
   (`RELEASE_GPU_WHEN_DONE = False` in cell 1 keeps it open; it is also kept when the results are not on Drive).

- Every stage logs one progress line per ~10% with an ETA, and marks warnings ⚠️ and errors ❌.
- If one job fails, the next one still runs.
- `python -m rtp.cli status --config configs/e1.yaml` shows what is done, partial, stale or not started.

## 2. Collect (PI)

`nb.feedback(CONFIG)` writes `runs/<config>/feedback/feedback_<UTC>.md` and a `.json` twin. The bundle contains:

- the newest cycle scorecard in full (earlier ones: their head). A scorecard has three layers:
  1. **Findings**: statements derived from the numbers by fixed rules (`rtp/scorecard.py`), each with its kind
     and the next step. Kinds: `bug`, `instrument`, `research`, `pending` (not evaluated yet, and which job
     would), `budget`.
  2. **Tables**: the four checks per model; the ruler (response of the anchor curve at every ladder step, and
     the share of brands that falls in each step); the brands on it (name effect, preference level, shares
     beyond the top, inside the first step, without a level); the scale-dependent metrics of prior work on the
     same items; the fictional names ranked by pull; two options vs four. Per stage pair: c, m and T1 with
     intervals, the change of level, the shape of the ruler step by step, the raw metrics of the two stages
     side by side. Then the open question, what changed since the previous cycle, and the budget ledger.
  3. The full numbers stay on disk: `scorecard.json`, `analysis/*/summary.json`, `checks.json`, `.parquet`.

  "What changed" compares a cycle that was run again after an instrument change with its own earlier scorecard
  (kept under `cycles/<name>/history/`): this is how a change of the instrument is judged. A cycle with no
  earlier version of its own is compared with the newest scorecard of another cycle.
- the environment and the non-PASS doctor checks; a status table of all stages;
- suggested next actions from fixed rules on the stage states (errors, stale or partial stages), and one line
  that counts the scorecard's findings by kind;
- every error with its traceback; deduplicated warnings with counts;
- the key results of each finished stage and the head of every dev stage summary; analysis summaries only of
  models and pairs that are not in the newest scorecard; test-split results stay sealed unless `--unseal`;
- log tails of failed and partial stages; the last pytest output.

Give the bundle file to the agent. There is no need to paste the notebook output.

## 3. Triage and improve (coding agent)

1. Read `CLAUDE.md`, `DECISIONS.md` (section "Instrument v2"), `docs/CYCLES.md`, then the bundle.
2. Triage every item; the scorecard's Findings section has already sorted the analysis numbers. Each item
   is one of four kinds:
   - **Bug** (error, crash, wrong file, broken resume; a `bug` finding: the pure-scale identity fails, an
     analysis crashed): reproduce it locally with `python -m rtp.cli smoke` or a minimal config. Fix it, and
     add a regression test in `tests/` that fails before the fix.
   - **Environment** (doctor FAIL, library versions, GPU): fix it in `pyproject.toml` or the notebook. Never
     reinstall torch.
   - **Instrument finding**: in the **develop** phase propose a change of the instrument, with the numbers;
     if the PI agrees, change the config or code, bump `ladder.version`, add a `DECISIONS.md` line, and plan
     the cycle that tests it. Collect the findings of one bundle into ONE change. In the **confirm** phase
     nothing changes: the finding is reported as it is.
   - **Research finding** (a scale multiplier, a content residual, a failed hypothesis): this is a result.
     Report it to the PI with its interval and noise band. Never change methods, thresholds or data to make a
     finding appear or disappear.

   What an instrument finding changes in the program:

   | Finding | What it means | What changes (develop phase) |
   |---|---|---|
   | The letter readout is not the model's answer (letter mass below 0.9) | No level of this model can be read | Prompt or prefill (`rtp/data/ladder_items.py`, `rtp/measure/parse.py`): a change of the GPU closure, every stage is re-measured |
   | A curve does not rise | The evidence does not move the model in that stratum | How the evidence is rendered, or the type is reported as unusable; a longer ladder does not help |
   | Brands beyond the top step (more than 10%) | The ladder is too short: those levels are bounds | Larger steps in `ladder.rating_deltas` / `price_deltas` / `review_ratios`; the finding gives the 95th percentile of the name effects against the top response |
   | Most brands inside the first step | The ladder is too coarse near 0 | A finer first step, about the median name effect |
   | Pooled curve not monotone | Neighbouring steps are closer than the noise | Merge or widen those steps (ruler table) |
   | Fictional anchors fail (SD ratio at least 0.5) | Made-up names are not neutral | Fix the replacement rule in DECISIONS.md, then replace names in `data/seeds/placebo_names.yaml` |
   | Fictional anchors not evaluated (real brands censored) | The ratio is only a bound | Extend the ladder first; do not replace names on this evidence |
   | Open question: low mass on catalogue brands | N_eff describes a minority of answers | Read the sampled texts; prompt wording of `rtp/measure/open_set.py` |
   | Estimator or report gap (a number the decision needs is missing) | The view is incomplete | `rtp/scale`, `rtp/analyze.py`, `rtp/scorecard.py`: outside every GPU closure, `cycle report` recomputes from stored scores, no GPU time |
3. Add an entry to `docs/CYCLES.md`: the cycle's question, what the scorecard showed, the decision, the change.
4. Run `pytest -q` and `python -m rtp.cli smoke` locally, commit with a message that names the bundle, push.
5. Tell the PI which cycle to run next (a new cycle file, or the same one to resume).

## 4. Develop and confirm

Iterating on an instrument after seeing results is legitimate only if the results that count were not seen.

- **Develop phase.** Models listed under `models` in the run config (Tulu 3 SFT and DPO). Everything measured
  here is exploratory, and the instrument may change between cycles.
- **Confirm phase.** The project's confirmation models (`rtp/registry.py` `CONFIRMATION_MODELS`: Tulu 3 final =
  the RLVR stage, and the three Olmo 3 stages; a config may add more under `confirm_models`) and the sealed test
  categories. Until `runs/<config>/FREEZE.json` exists, every command that loads a model refuses them and their
  variants, under every config: the ladder and open-set stages, and the legacy and frozen ones too (`measure`,
  `format-probe`, `open-ended`, `extract`, `probe`, `localize`). The model lists of a config cannot be overridden
  with `--set`. The ladder family also refuses the test split until then, with or without `--unseal`. A confirm
  cycle refuses to start without the freeze and never re-measures a stale stage. After the freeze the instrument
  does not change, the notebook is pinned to a commit, and the test split is opened once (`--unseal`).
  The `freeze` command itself (the record of what is frozen) belongs to batch 3.

## 5. What invalidates a result

Stage identity covers the stage settings, the model (its registry entry, including a variant's logit scale and
adapter sha256), the split, the prompt mode, the sha256 of the item file, and the digest of the source of the
stage's import closure. So:

- Unchanged finished stages are skipped.
- Stages whose code or inputs changed refuse to resume (exit 4) until they are re-run with `--force`: results
  from two code versions are never mixed. `cycle plan` and `cycle run` stop when a job has such a stage, unless
  `--redo-stale` is given (develop phase), and `cycle run` refuses to continue a cycle whose measurement code
  changed since it started (`--restart --redo-stale` starts it over). The code a cycle started with is recorded
  once in its `run_meta.json`.
- A half-finished stage also refuses to resume under a different torch, transformers or tokenizers version,
  GPU, or model revision.
- The estimators (`rtp/scale`, `rtp/analyze.py`) are outside every GPU closure: changing them never forces a
  re-measurement. The analysis is a cached view keyed by the stages' hashes and completion records and by the
  digest of the analysis code, so an edit there, or a re-measured stage, recomputes it at the next scorecard;
  `cycle report` always recomputes.

Exit codes of `rtp.cli`: 0 ok; 3 sealed split; 4 configuration, data, phase or stage-identity error; 5 the
cycle does not fit its budget; 6 the deadline was reached between two shards (resumable); 7 a cycle ended
with deferred or failed jobs.

After editing the notebook's code cells, update `NOTEBOOK_REV` in cell 1 to `rtp.notebook.notebook_rev(path)`; a
test fails until it matches, so an outdated copy can always be recognised.

## Conventions that keep the loop safe

- Every bug found on Colab gets a test, so the same bug cannot come back silently.
- Checks and warnings come from fixed constants in the code (`rtp/scale/checks.py`, `LOW_COVERAGE` in
  `rtp/measure/ladder.py`). Changing one is a design decision and needs a `DECISIONS.md` entry, made BEFORE
  looking at a model's results; afterwards it is post-hoc and must be reported as such.
- Iterate on dev only. Never pass `--unseal` while the instrument can still change.
- Numerics: stages load models in bfloat16. The ladder stage recomputes the letter logits in float32 from the
  final hidden state, because bf16 rounds a logit near 20 to a grid of 0.125. Integration tests that compare two
  exact computation paths use a float32 copy of the smoke model.
- Batch fixes: a change in a stage's code invalidates that stage, so collect the fixes from one bundle into one
  commit, and never change measurement code in the middle of a cycle.
- The feedback bundle never contains secrets: log records pass through a redaction filter, and only whether
  `HF_TOKEN` is set is reported.
- `--set section.key=value` lets an agent ask the PI to try a change without editing configs. Batch limits are not
  part of any stage's identity, so changing them never forces a recompute.
- The legacy four-option family (`measure`, `open-ended`) and the white-box stages (`extract`, `probe`,
  `localize`) are frozen: their commands and tests stay, but they are not part of any cycle.
