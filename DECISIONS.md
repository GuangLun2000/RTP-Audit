# Design decisions not fixed by the spec

One line per decision: what was chosen, and why (simplest option unless noted).
"Spec issue" marks places where the spec is wrong, ambiguous or self-contradictory.

## Spec issues found
- Spec issue: the kickoff prompt calls the spec `RTP_AUDIT_AGENT_BRIEF.md`; the file on disk is `RTP_AUDIT_AGENT.md`, and `.gitignore` excludes it, so it stays out of the repo (a local copy in the worktree is gitignored too).
- Spec issue: §8 example `item_id` `rs-000123-r2` cannot tell the real-name and placebo-name versions of a pair apart; ids are `{pair_id}-r{rotation}` (real) and `{pair_id}-p{rotation}` (placebo), so the example reads as "real, rotation 2".
- Spec issue: §7.1 "two fixed real competitors" cannot mean fixed per category, since smoke has only 3 real brands per category and that would leave a single target. "Fixed" is read as constant within a pair (both name versions, all 4 rotations); competitors are drawn per pair from the other real brands of the category.
- Spec issue: §9 crosses evidence 0–4 with handicap 0–4, but §7.1 says handicap only applies at evidence ≥ 3. Evidence 0–2 therefore only get handicap 0, which avoids duplicate cells that differ only in an unused label.
- Spec issue: §10 lists unit tests for robust z-scores and subspace projection, but those estimators belong to M3/M4 (§7.5, §7.7). They are deferred to those milestones, not implemented this round.
- Spec issue: the dataset ID `allenai/dolci-3-instruct-dpo-with-metadata` (marked "to be confirmed") now redirects on the Hub to `allenai/Dolci-Instruct-DPO`. `validate_repo_id` refuses redirects, so it fails loudly until a human updates the registry. Checked 2026-11-11.
- Spec issue: the card of `allenai/Olmo-3-7B-Instruct-DPO` lists `allenai/Dolci-Think-DPO-7B` as its dataset, not an Instruct DPO set. Only relevant for M5; flagged for review.

## M0 infrastructure
- `torch` is not a declared dependency: Colab ships it and the spec forbids reinstalling it.
- Upper bounds pin the major version of every library whose API we use. `pyarrow` has only a lower bound because its major version moves fast and pinning it would force Colab downgrades.
- Without CUDA (laptops, CI), models load in float32 on CPU; on CUDA they load in bf16 with `device_map={"": 0}` as the spec says.
- `load_model(model_path)` has exactly one parameter (hard constraint 1), and a unit test enforces that.
- Model IDs appear only in `rtp/registry.py` (enforced by a test). YAML configs and the notebook may list IDs as user input, because spec §9 and §13 do.
- Chat/base mode comes from the registry for known IDs (base checkpoints are never chat). Unregistered paths count as chat when the tokenizer has a chat template.
- No system message is passed. Templates that add their own default system prompt (Qwen2.5) keep it, because that is what `apply_chat_template` produces.
- Chat prompts are rendered with `apply_chat_template(tokenize=False)` and tokenised with `add_special_tokens=False`, which is identical to `apply_chat_template(tokenize=True)`. Base-model text prompts use the tokenizer's normal BOS handling.
- Left padding plus explicit `position_ids` (cumsum of the attention mask) make batched and unbatched forward passes agree. An integration test checks this.
- Forward passes use `logits_to_keep` so that full-vocabulary logits are only materialised where they are needed.
- Dynamic batching sorts prompts by token length under two limits: `max_batch_size` and `max_batch_tokens` (batch × longest prompt). On CUDA OOM a batch is halved once; a second OOM propagates.
- Each stage dir stores the hash of the config that created it. Resuming with a different config is refused unless `--force` is passed, which prevents mixing shards from two configs.
- Generated data live in `runs/{config}/data/`, because data are model-independent (spec §9 "generate once").
- `smoke` deletes `runs/<smoke config name>/` first so that every smoke run is a real end-to-end test; `--resume` keeps it.
- IDs marked "to be confirmed" are checked with `HfApi().model_info/dataset_info` before loading. That check is skipped, with a warning, when `HF_HUB_OFFLINE` is set.
- A logging filter masks `hf_...` tokens and the literal value of `$HF_TOKEN`.
- An extra CLI helper `make-placebos` regenerates `data/seeds/placebo_names.yaml`. It is seed maintenance, not a pipeline stage.
- Future stages (`extract`, `probe`, `transplant`, `score`, `plant`, `eval`) are registered and exit with code 2 and a "not implemented in this round" message.
