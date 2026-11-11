# rtp-audit — read this first every session

Full spec: RTP_AUDIT_AGENT_BRIEF.md (on disk as `RTP_AUDIT_AGENT.md` in the repo root; gitignored, never commit it).
Open design choices and spec deviations: DECISIONS.md. Code, comments, docstrings, commits: English only.

## Working mode (spec §0)
- Milestones in order M0 → M6; stop and report after each agreed batch (current batch: M0–M2).
- Each milestone: `pytest -q` passes, `python -m rtp.cli smoke` passes, README usage updated.
- Anything the spec leaves open: simplest implementation + one line in DECISIONS.md. Do not widen scope.
- Research code: never tune methods/thresholds/data to make results look good; report anomalies as they are.
- IDs marked "to be confirmed" are validated at runtime; on failure raise a clear error, never substitute.

## Hard constraints (spec §2)
1. `rtp/detect/` never loads a base model: functions take ONE model path. Base/weight-diff methods only in `rtp/baselines/`.
2. Every promoted model has a twin: same data size, templates, seed, hparams; only the promo target differs (fixed vs rotated neutral brands).
3. Calibrate inside one model: identity residuals judged against that model's own brand distribution, never across models.
4. Sealed test split: dev/test split by category; reading `test` requires explicit `--unseal`, else error. Develop on `dev` only.
5. Natural negatives (data/seeds/natural_negatives.csv = Appendix A verbatim) are first-class: measurable directly, evaluated as "must not alarm".

## Pitfalls (spec §11)
- Always `tokenizer.apply_chat_template`; never hand-write templates. Base `meta-llama/Llama-3.1-8B` is not a chat model: text continuation only.
- Option-letter tokens have variants (" A" vs "A"): logsumexp-merge, renormalise over A–D. Always average 4 cyclic rotations.
- Brand names are multi-token: open-ended scoring sums token log-probs and merges aliases with logsumexp.
- First-token argmax can disagree with generation: run the 2% greedy consistency check, warn below 90%.
- Placebo names can collide: case-insensitive substring conflict check, keep `needs_web_check: true`.
- Llama repos are gated: on load failure tell the user to accept the licence and set HF_TOKEN.
- Do not depend on open-instruct launch scripts (reference `open_instruct/dpo.py` + hparams only).
- Secrets: HF_TOKEN only from env / Colab secrets; never in code, configs, logs, commits.
- Drive is slow: HF cache on local disk (HF_HOME), results in RTP_WORK_DIR; estimate activation shard sizes before writing.
- All model/dataset IDs live only in rtp/registry.py. Never reinstall torch.
