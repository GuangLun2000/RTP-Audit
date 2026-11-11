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

## M1 data generation
- Data split: whole categories are dev or test (`test_categories`). Items and probes go to `dev/` and `test/` files, and the catalogue carries a `split` field. The sealed readers (`load_choice_items`, `load_probe_items`, `load_catalogue`) and the CLI raise `SealedSplitError` (exit code 3) for `test` without `--unseal`, before any model is loaded.
- Writing the test split in `build-data` is not a "read". The data report shows test cell counts (counts only), so the user can check the design.
- Catalogue records extend spec §8 with `split`, `natural_negative` (`named`/`picked`) and `match_case`, which the open-ended matcher needs.
- Real-brand selection for `n_real_per_category`: the category's Appendix A named and picked brands always come first (hard constraint 5), then the brands in YAML order. A natural negative missing from categories.yaml is an error.
- Choice items extend spec §8 with `pair_entity` (the real brand of the pair), `placebo_substitute`, per-option `role`/`is_placebo`/`description`, and the rendered `prompt`. `target_entity` is the name actually shown at the target slot, so for the placebo version it is the placebo name.
- Per pair, the 2 competitors (other real brands of the category) are drawn uniformly, along with a placebo substitute and a different placebo distractor (both from the category's choice placebo pool) and a random order of the 3 non-target options. The 4 rotations are cyclic shifts that put the target at A, B, C, D.
- Cards: price is uniform in the category range; rating is one shared value R (uniform, 0.1 steps) for every non-target option; review count is log-uniform; 2 key specs come from the feature pool; descriptions and lab notes are distinct within a pair. Rounding: integer dollars, or x.49/x.99 for categories under $100.
- Handicap only changes the target's rating (R − delta[h], deltas 0/0.1/0.3/0.5/1.0) and only at evidence ≥ 3, where ratings are visible. Non-target ratings are all equal, so "worse than the others by delta" is exact.
- Evidence ladder rendering: 0 shows the name only (`A. Nike`, no colon); 1 adds a neutral description; 2 adds `Price: $X. Key specs: a; b.`; 3 adds `Rating: 4.4/5 from 2,310 reviews.`; 4 adds `Third-party review: ...`. All lab notes share the same mildly positive valence, so level 4 adds a verdict without ranking the options.
- Spec templates read "a {category}", which is ungrammatical for "electric car". All templates (choice, open-ended, probe) therefore use the correct indefinite article ("an electric car").
- Placebo names: syllable pseudo-words of length 4–8, deterministic in (seed, pool, category), unique across all categories and pools. The choice pool (8 per category) and the probe pool (40 per category) are disjoint, so the probe-trained K never saw the transplant's base names. The seed file is committed and is re-validated on every build.
- Placebo conflict check: casefolded, accent-stripped, apostrophe-free comparison against all names and aliases in categories.yaml, every Appendix A brand, and every word of multi-word names, in both directions. Real strings shorter than 3 characters (HP, VW) match exactly only. Placebos may not contain each other.
- `categories.yaml` has an `extra_known_brands` list that is used only for the conflict check. It was added after the first generated pool contained "Tivo" (TiVo). It is drafted from memory and does not replace the web check.
- Probe items: each placebo entity gets one uniformly random attribute vector, expressed in `probe_templates` (≤ 10) phrasings. Phrasing t uses frame t of every attribute and rotates the attribute order by t mod 4. `name_char_span` points at the last mention of the name. Probe items carry an extra `template_id` field.
- RNG: `numpy.random.default_rng(sha256(seed, stream, category))` per category and stream, never Python's salted `hash`. Output JSON keeps construction order, so files are byte-identical across runs.
- Schema validation is hand-written (no jsonschema dependency). Besides types, it checks the design invariants: role counts, target at its rotation, brand name first on each line, and pairs that differ only in the target name.
- `cards.EmpiricalCardSampler` and `cards.inspect_amazon_meta` are the reserved Amazon interface. Neither is wired into configs yet.
