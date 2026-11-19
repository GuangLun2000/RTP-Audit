# Experiment cycles

One cycle = one question, one GPU session of at most 12 hours on one G4 (RTX PRO 6000, 96 GB), one scorecard
(`runs/<config>/cycles/<name>/scorecard.md`). The mechanics are in docs/AGENT_LOOP.md; this file is the plan
and the log: after every cycle the coding agent adds an entry below (question, result, decision, change).

## Plan

Times for C0, C1, C6 and C7 are predictions of the cost model (`rtp/cost.py`) on the built e1 items, at rates
fitted on the first pilot's legacy readout; they err on the long side, and the first job of C0 recalibrates
them. One 8B model on all three dev tiers is predicted at 27 minutes (core 8, ext 16, ctx 3) plus about 5
minutes to download and load a checkpoint that is not on the VM's disk; the test tiers at 17 minutes.
The times of C2–C4 are guesses: the LoRA training speed is not measured yet. C2 starts with a benchmark job,
and the number of runs per training cycle follows from it.

| Cycle | Phase | Question | Jobs | Predicted GPU time |
|---|---|---|---|---|
| C0 shakedown | develop | Does the forced letter readout work; are the ladders monotone and long enough; what does a pass cost? | Tulu 3 SFT and DPO on `core`; `SFT@scale=2` wiring check; open-set on SFT | 43 min |
| C1 Tulu SFT→DPO | develop | Do the four conditions hold; do rating, price and review evidence rescale alike; what is left after the scale factor; does θ depend on the choice set? | both models on `core`, `ext` and `ctx`; open-set on both | 1.3 h (50 min if `core` is current from C0) |
| C2 neutral seeds, parent SFT | develop | Run-to-run noise band of θ and b; is generic DPO sharpening close to pure scale? | 25-minute training benchmark; 5 seeds × (train + `core`) | guess: 3 to 4.5 h |
| C3 neutral seeds, parent DPO | develop | Noise band for the DPO→RLVR comparison | 5 seeds × (train + `core`) | guess: 2.5 to 3.7 h |
| C4a, C4b dose planting | develop | Dose-response, specificity, detection limit of a planted preference (θ vs raw metrics) | 3 doses × 3 seeds × (planted + twin); seed 1 of every dose first | guess: 8 to 13 h in two sessions |
| C5 freeze | — | Did H2 pass? | record the instrument hash; re-measure only if it changed | 0 to 1.5 h |
| C6 confirmation, dev | confirm | DPO→RLVR in both families; SFT→DPO in Olmo 3 | Tulu 3 final; Olmo 3 SFT, DPO, final: all tiers and open-set | 2.6 h |
| C7 unseal | confirm | H4: held-out choices and concentration | the six stage models on the test split | 2.9 h |

Cycle files exist for C0 and C1 (`configs/cycles/`). C2–C4 need the training code (batch 2); C5–C7 need the
freeze and unseal code (batch 3).

## What a cycle can conclude, and what it cannot

| Cycle | Conclusions its scorecard supports | Not yet, and why |
|---|---|---|
| C0 | Is the letter readout the model's answer; does each evidence type move the model; where the brands sit on each ladder (too short, too coarse); do fictional names stay neutral; the pure-scale wiring identity on an 8B model; raw metrics move under a pure ×2 scale while every level stays; real cost per token | Nothing about SFT → DPO beyond a first look: `core` only, no direct readings, and the instrument may change |
| C1 | Tulu 3 SFT → DPO, exploratory: c per evidence type and its spread (H1); T1 per stratum (did name effects grow by the scale factor); change of level in evidence units; direct vs converted level; two options vs four; raw metrics of the two stages next to c | No claim against training noise (no seed replicates before C2/C3); not confirmatory (this pair was seen in the pilot); the share of a raw-metric change explained by scale is batch 3 |
| C2, C3 | Run-to-run band of c, T1 and the levels under content-neutral training; whether generic DPO sharpening is close to pure scale | The band comes from LoRA runs; the public stages are full fine-tunes (reported at 1× and 2× the band) |
| C4 | H2: a planted preference raises the target's level with the dose, twins and non-targets stay in the band; detection limit of the level vs raw metrics | LoRA runs on a development checkpoint; the design is fixed when batch 2 is written |
| C6 | Confirmatory, both families: DPO → RLVR; SFT → DPO in Olmo 3. H1 and H3 against the band | Dev categories only |
| C7 | H4: held-out categories and open-question concentration predicted from level and scale | Opened once |

Every instrument finding of a develop cycle becomes a proposal for the next instrument version; the table in
docs/AGENT_LOOP.md §3 says which file each finding changes. Estimator and report changes (`rtp/scale`,
`rtp/analyze.py`, `rtp/scorecard.py`) need no GPU time: `cycle report` recomputes them from the stored scores.

Rules that hold for every cycle:

- Development models: Tulu 3 SFT and DPO (already seen in the first pilot). Everything measured on them is
  exploratory. Tulu 3 final (the RLVR stage) and all Olmo 3 stages stay unmeasured until the freeze.
- An instrument change between cycles bumps `ladder.version` in `configs/e1.yaml` and gets a DECISIONS.md line.
  Re-measuring both development models on every tier is predicted at about 1.2 GPU hours, so a change of the
  instrument costs one short session (`AFTER_INSTRUMENT_CHANGE = True` in the notebook).
- Everything that needs no training (C0, C1, C6, C7) is predicted at about 6.5 to 7.5 GPU hours in total and
  would fit one 12-hour session; it is still run as separate cycles, because C6 and C7 come after the freeze.
- Open (PI, when batch 2 is written): the size of the training cycles. As planned, 28 LoRA runs are a guess of
  12 to 21 hours over three sessions. A compact variant, 3 neutral seeds and 2 doses × 2 seeds × (planted +
  twin) = 11 runs, is a guess of 5 to 8 hours and would bring one full develop-phase iteration (training plus
  the Tulu measurements) to 7 to 10 hours, at the price of a 3-seed noise band and a detection limit known
  only as a bracket.
- The cut order of a cycle is written in its file before it runs.

## Log

Entry template:

```
### C0 (date, bundle feedback_<UTC>.md, commit <hash>)
- Question:
- Result (numbers from the scorecard, with intervals):
- Budget: predicted vs actual; calibration k:
- Decision (PI):
- Change (instrument version, DECISIONS.md line, code):
- Next cycle:
```

### C0 (2026-11-19, bundle feedback_20261119T223714Z.md, commit 6d908c4)
- Question: does the forced letter readout work on the Tulu 3 development models, are the evidence ladders
  monotone and long enough to bracket the brands' name effects, and what does a ladder pass cost?
- Result (develop phase, exploratory, no noise band yet; ladder version 1, core tier, Tulu 3 SFT and DPO):
  - Readout check FAILED by its rule: mean letter mass at the forced position 0.675 (SFT) and 0.625 (DPO)
    against 0.9. The rule is not scale-free: the same SFT model with all logits doubled reads 0.874 while its
    decoded answers and all 1188 levels are unchanged. The letter readout and the letters + names readout agree
    (correlation 0.997 / 0.999, median gap 0.06 / 0.05 logits). Where the other 27-36% of the probability goes
    was not stored.
  - Ladders: monotone, but not a graded ruler. The first step already gives 47-58% (rating, 0.1 stars), 74-83%
    (price, 5%) and 92-96% (reviews, 2x) of the response at the top step, and 100% of the 198 brands lie below
    the first step in every cell. The reported level is therefore the ratio name effect / response at the
    first step (median 0.04-0.09 with a tied attribute shown, 0.35-0.56 for a bare name), times the size of
    that step. The response below the first step was never measured. The ladder check passed because it only
    guards the top of the ladder.
  - Fictional anchors FAILED (SD ratio 0.584 / 0.563 against 0.5). With a tied attribute shown the real name
    effects shrink to a median of 0.11-0.35 logits (bare name: 0.84 / 2.02), so the ratio fails where names
    barely matter. No fictional name pulls as hard as a typical real brand (median ratio 0.11 / 0.13).
  - SFT to DPO: c = 2.27 [2.22, 2.33] price, 2.23 [2.19, 2.28] rating, 2.12 [2.09, 2.16] reviews. The curves
    changed shape: the first-step response grew x2.48 / 2.46 / 2.12, the rise beyond it about x1.5. Bare-name
    effects grew x2.53 [2.42, 2.64], the median position bias x1.70, the printed fictional pulls about x3.
    T1 for bare names is +0.10 to +0.17 against c and +0.02 to +0.03 against the first step (price, rating):
    its sign depends on which contrast is called the scale.
  - Pure-scale identity on the 8B model: multipliers 2 on all three curves, largest level change 0.
  - Open question, SFT: N_eff 5.14 (name) and 4.29 (recommend); sampled N_eff 4.51 at T = 1, 2.25 at T = 0.5.
- Budget: predicted 41 min, actual 15 min; calibration k = 0.38. GPU time does not limit instrument iteration.
- Decision (PI): pending. Proposed: do not run C1 on the unchanged ext tier as a decision cycle (one attribute
  differs there too, so the same curve shape is expected); run a short probe cycle first.
- Change (instrument version, DECISIONS.md line, code): no instrument change yet. Operations fixes in 72b1fae
  (peft / torchao on Colab, full results disk is a doctor FAIL, bundle download cell).
- Next cycle: proposed probe C0b on the two development models: store the non-letter tokens at the forced
  position; anchors at smaller steps and on a trade-off grid (two attributes in opposite directions); the ext
  anchors unchanged as a prediction test; the open question on DPO.
