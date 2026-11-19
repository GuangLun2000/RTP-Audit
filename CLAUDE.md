# rtp-audit — read this first every session

Research objective: `research_objective_v1118.md` in the repo root (gitignored, Chinese). It supersedes the old
spec `RTP_AUDIT_AGENT.md` and `local/RESEARCH_PLAN.md`; read those only as history.
Design record: DECISIONS.md (section "Instrument v2" first). Cycle log: docs/CYCLES.md. Loop: docs/AGENT_LOOP.md.
Code, comments, docstrings, commits: English only.

## What the project measures
- Raw propensity metrics (choice share, flip rate, concentration) mix CONTENT (what a model prefers) with SCALE
  (how decisively it says so). Post-training raises scale, so a rise in a raw metric is not a change of preference.
- Per model: name effect Δ_e (logits), anchor curve A(d) between FICTIONAL options (logits per evidence step),
  preference level θ_e = A⁻¹(Δ_e) in evidence units. A pure scale change multiplies Δ and A and leaves θ unchanged.
- Per stage pair (SFT→DPO→RLVR, Tulu 3 and Olmo 3): scale multiplier c per evidence type, content residual
  θ_B − θ_A, and the null model "post-training = uniform sharpening".
- Vocabulary trap: in the LEGACY four-option family `evidence_level` 0–4 is information sufficiency and `handicap`
  is the adverse evidence. In the ladder family `stratum` is information sufficiency and `d` the adverse level.

## Working mode
- Three code batches: 1 = instrument and cycles (can run C0–C1); 2 = LoRA training, noise band, planting
  (C2–C4); 3 = freeze, unseal, prediction (C5–C7). After each batch: `pytest -q` and `python -m rtp.cli smoke`
  pass, push `main`, report to the PI, stop.
- Experiments run as CYCLES (`configs/cycles/*.yaml`, `rtp/cycle.py`): one question, one budgeted GPU session of at
  most 12 h on one G4 (RTX PRO 6000, 96 GB), one scorecard. `cycle plan` refuses a cycle that does not fit.
  After each cycle: read the feedback bundle, add an entry to docs/CYCLES.md (question, result, decision, change).
- A scorecard starts with FINDINGS (`rtp/scorecard.py`: fixed rules on the analysis numbers). Kinds: bug (fix, with
  a test), instrument (develop phase: one collected change, version bump, DECISIONS.md line, re-measure), research
  (report with its interval; never tune away), pending, budget. Read the findings before the tables.
- Anything the objective leaves open: simplest implementation + one line in DECISIONS.md. Do not widen scope.
- Research code: never tune methods, thresholds or data to make results look good; report anomalies as they are.
- IDs marked "to be confirmed" are validated at runtime; on failure raise a clear error, never substitute.

## Hard constraints
1. One model per stage: `load_model` takes ONE model id (a checkpoint, or a variant of it: logit scale, one LoRA
   adapter). Models are compared only on stored per-model tables (`rtp/scale`, `rtp/analyze.py`), never by loading
   two models. `rtp/detect/` (frozen) never loads a base model.
2. Twins: every planted model has a twin with the same data size, templates, seed and hyperparameters; only the
   promoted target differs. Dose runs add seeds per dose and a dose-0 arm.
3. Units are calibrated inside one model (its own anchor curve). Across models compare only θ (evidence units)
   and ratio statistics (c, m), each against a noise band; never raw logits.
4. Develop / confirm: the instrument may change only while the development models (`models` in the config) are the
   only ones measured. Confirmation models (`registry.CONFIRMATION_MODELS`: Tulu 3 final, all Olmo 3 stages; plus a
   config's `confirm_models`) and the sealed test categories are measured after the freeze, once. Before the freeze
   EVERY command that loads a model refuses them (`rtp/phase.py`), legacy stages included; never work around it.
   Reading `test` requires `--unseal` and, for the ladder family, the freeze.
5. Every instrument change in the develop phase bumps `ladder.version` and gets a DECISIONS.md line; thresholds are
   fixed constants (`rtp/scale/checks.py`) and are changed before, not after, looking at a model's results.
6. Anything in the import closure of a GPU stage (`rtp/models.py`, `rtp/infer.py`, `rtp/patching.py`, `rtp/rng.py`,
   `rtp/variants.py`, `rtp/measure/{answers,parse,ladder,open_set}.py`), a hashed config section or a `ModelSpec`
   field invalidates finished stages; a test pins the two digests. Land such changes together, before a campaign.
   Estimators and views live in `rtp/scale` and `rtp/analyze.py`, outside it. A cycle never mixes results: stale
   stages block it until `cycle run --redo-stale` (develop phase only).

## Pitfalls
- Always `tokenizer.apply_chat_template`; never hand-write templates. Base `meta-llama/Llama-3.1-8B` is not a chat
  model: text continuation only.
- The preference level uses ONE readout: the letter-logit difference target − control at the forced position, in
  one token form, averaged over option orders. Answer-path masses, the free position and O5 are diagnostics: they
  are not homogeneous under a scale change.
- bf16 rounds logits near 20 to a grid of 0.125: letter logits are recomputed in float32 from the final hidden state.
- A brand whose name starts with an option-letter token ("Acer" → "A") is flagged `letter_collision`.
- Fictional names can collide with real brands or carry their own pull: check the pulls table; `needs_web_check`.
- First-token argmax can disagree with generation: the decoded sample of the ladder stage reports the agreement.
- Llama repos are gated: on load failure tell the user to accept the licence and set HF_TOKEN.
- Secrets: HF_TOKEN only from env / Colab secrets; never in code, configs, logs, commits.
- Drive is slow: HF_HOME on local disk, results in RTP_WORK_DIR. IDs only in rtp/registry.py; never reinstall torch.
- The repo is public: never commit downloaded results (`v_*/`, `local/`) or the objective file.
- `rtp/report.py` is the LEGACY cross-model report (command `report`); the cycle scorecard is `rtp/scorecard.py`.
