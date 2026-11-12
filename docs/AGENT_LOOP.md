# The improvement loop: Colab run → feedback bundle → coding agent → re-run

The code base is meant to get better with every GPU session. The loop has four steps, and each one
has a tool built for it.

```
 ┌───────────────┐   live output, events.jsonl, status   ┌──────────────────────┐
 │ Colab session │ ────────────────────────────────────▶ │ feedback bundle (.md) │
 └──────▲────────┘                                       └──────────┬───────────┘
        │ git pull; re-run: only stages whose code/inputs             │ given to the agent
        │ changed are recomputed (stage = settings + input             ▼
        │ files + code digest)                                ┌──────────────────────┐
        └──────────────────────────────────────────────────── │ coding agent fixes,  │
                                                              │ adds tests, records  │
                                                              │ decisions, commits   │
                                                              └──────────────────────┘
```

## 1. Run (PI, on Colab)

Use `colab/run_pilot.ipynb`.

- `nb.run(...)` streams every stage live. Each stage logs one progress line per ~10% with an ETA, and marks
  warnings ⚠️ and errors ❌.
- If one model fails, the next model still runs.
- `python -m rtp.cli status --config configs/pilot.yaml` shows what is done, partial, stale or not started.
- `python -m rtp.cli doctor --config ...` checks the environment before anything expensive runs: GPU, bf16,
  library pins, HF token, disks, Drive mount, data, and Hub access.

## 2. Collect (PI)

`nb.feedback(CONFIG)` (or `python -m rtp.cli feedback --config ...`) writes
`runs/<config>/feedback/feedback_<UTC>.md` and a `.json` twin. The bundle contains:

- the environment and the non-PASS doctor checks;
- a status table of all stages;
- suggested next actions, derived from fixed rules;
- every error with its traceback, from `events.jsonl`;
- deduplicated warnings with counts;
- the key results of each finished stage (`_DONE` files);
- log tails of failed and partial stages;
- the last pytest output.

Give the bundle file to the agent. There is no need to paste the notebook output.

## 3. Fix (coding agent)

1. Read `CLAUDE.md` (rules), `DECISIONS.md` (design record), then the bundle.
2. Triage every item. Each one is one of three kinds:
   - **Bug** (error, crash, wrong file, broken resume): reproduce it locally with `python -m rtp.cli smoke`,
     or with a minimal config and `--set` overrides. Fix it, and add a regression test in `tests/` that fails
     before the fix.
   - **Environment** (doctor FAIL, library versions, GPU): fix it in `pyproject.toml` or the notebook. Never
     reinstall torch.
   - **Method-level finding** (for example: the model ignores evidence, K does not transfer, familiarity is not
     captured): this is a research result. Report it to the PI with the numbers. Do not change methods,
     thresholds or data to make the warning go away. If the PI decides on a design change, record it in
     `DECISIONS.md` together with the evidence.
3. Run `pytest -q` and `python -m rtp.cli smoke` locally, then commit with a message that names the bundle.
4. Tell the PI which commands to re-run. Usually it is the same notebook cell.

## 4. Re-run (PI)

Run `git pull` (cell 2), then re-run the failed cell. Stage identity covers the stage settings, the model, the
split, the prompt mode, the sha256 of the input files, and the digest of the source of the stage's import
closure. So:

- Unchanged finished stages are skipped.
- Stages whose code or inputs changed refuse to resume (exit 4) until they are re-run with `--force`. This is
  deliberate: results from two code versions are never mixed.
- A half-finished stage also refuses to resume under a different torch, transformers or tokenizers version,
  GPU, or model revision.

To make a long pilot exactly reproducible, set `PIN_COMMIT` in cell 2.

## Conventions that keep the loop safe

- Every bug found on Colab gets a test, so the same bug cannot come back silently.
- Warnings are produced from fixed constants in the code: `SELECTIVITY_WARN`, `NAME_MASS_GAP_WARN`, the 90%
  consistency threshold, and others. Changing one is a design decision and needs a `DECISIONS.md` entry.
- The feedback bundle never contains secrets: log records pass through a redaction filter, and only whether
  `HF_TOKEN` is set is reported.
- `--set section.key=value` lets an agent ask the PI to try a change without editing configs. Batch limits are not
  part of any stage's identity, so changing them never forces a recompute.
