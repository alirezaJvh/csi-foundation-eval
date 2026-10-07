---
title: Adversarial review — csi-bench architecture spine
target: ../ARCHITECTURE-SPINE.md (status draft, updated 2026-10-06)
lens: "Two units one level down that obey every AD and convention to the letter yet build incompatibly"
reviewed: 2026-10-06
context: specs/spec-lwm-wifi-har/{SPEC,datasets,preprocessing,lwm-models,candidate-models}.md
---

# Adversarial review — csi-bench spine

## Verdict

**Not ready for story breakdown. Amend first.** The paradigm (ports, registry, immutable contracts, one composition root) is sound, and AD-6/AD-7/AD-11 aim at the right leakage risks. But the spine fixes stage *names* without fixing the *signatures and data* that pass between several of them. Two developers can each follow every AD exactly and still produce results that don't fit together. That breaks the one thing the thesis needs most: a CAP-5 table of like-for-like numbers on identical splits that can be regenerated.

This review lists 18 holes: 6 critical, 7 high and 5 medium. The critical ones are:

1. The `paper_reference` source cannot satisfy AD-4, AD-6 or AD-7 at all.
2. `seeds` and `run_id` and `split_hash` interact in three incompatible ways.
3. Room selection (r1 / r2 / mixed) has no owner, and how it relates to the split is undefined.
4. The `Preprocessor` port has no fit/transform shape, and the spine doesn't say whether "preprocessing" is one adapter or a chain.
5. Validation carving has no owner, and `WindowSet.split` has no `val` value.
6. Code changes never invalidate `run_id` or the cache, so a sweep resumed after a bug fix silently keeps stale results.

Severity scale:

- **Critical**: silently wrong or non-comparable thesis numbers.
- **High**: units that won't integrate, or a leakage path.
- **Medium**: duplicated or wasted runs, or ambiguity that costs rework.

---

## C1 — `paper_reference` cannot pass through the pipeline AD-1 fixes

- **Severity:** Critical
- **Unit A:** `adapters/datasets/paper_reference.py` (story "CAP-5 reference row")
- **Unit B:** `adapters/splitters/random.py` + `core/windowing.py` Windower

**How both comply yet clash.**

- AD-16 says the paper's method runs "through a `paper_reference` dataset source (precomputed `saved_dataset` features) with the `random` splitter".
- AD-4 says a `DatasetSource` emits `Recording`s: complex64 `(antennas, packets, subcarriers)` with a per-file `recording_id`. The `saved_dataset_*.npy` files are rows of 6 real features plus a label. The rows are already windowed (W2 = 10/30/40/50, stride 1) and concatenated across recordings (and across rooms for `r1r2`), with no recording id and no packet index.
- **Dev A** obeys AD-4 literally. They pack each row as a fake `Recording` with `csi` shape `(1, 1, 6)` cast to complex64, `recording_id = "paper_reference/dataset_r1_w2_40"`, and the label of... which row? A `Recording` has one label, so A has to emit one `Recording` per row: 62,227 recordings.
- **Dev B** obeys AD-6 literally ("assigns contiguous packet blocks per recording ... with a gap of at least `max_window_len`"). The `random` splitter cannot do that, because a random split is by definition not contiguous blocks. Its options are to:
  - split each 1-packet recording into a block, which is impossible with a gap, or
  - violate AD-6.
- The Windower (AD-7: "windows lie entirely inside a single split block") then gets 1-packet recordings and a `window_len` of 40. It either fails (AD-7: "a run failure"), or its config gets set to `window_len=1`, so the run is labelled W=1 while it actually used W2=40.
- After that, an extractor has to turn the 6 features back into a vector, which means an `identity` extractor nobody specified.
- A third dev may instead let `paper_reference` emit a `FeatureMatrix` directly to skip all that. That bypasses Splitter, Windower and Extractor, which breaks AD-1 ("stages call each other directly") and AD-7 ("the Windower alone owns window boundaries").

Every path either breaks an AD or records a config that lies about what ran.

**Proposed rule fix (amend AD-16, AD-6):**

> **AD-16 (rewritten).** The paper reference row is produced by recomputing the paper's method, not by loading `saved_dataset`. The `tor_vergata` source feeds a `per_window` extractor `paper_beamforming`, which vendors `processing.py` per AD-15 with its known CSI-distance behaviour preserved and documented. It runs with Windower `window_len = W2`, `stride = 1`, and the `random_window` splitter. `saved_dataset/*.npy` is used only by a test that asserts the recomputed features match those files row-for-row (within tolerance) for antennas 0–1. Runs with `splitter.name == random_window` must have `role: reference`. Config validation rejects any other combination.
>
> **AD-6 (add).** A `SplitPlan` has `unit: packet_block | window`. `packet_block` plans (the `time_block` splitter) obey the contiguous-block-plus-gap rule. `window` plans (only the `random_window` splitter, only for `role: reference`) assign whole windows to train/test after windowing. They are stored and hashed like any plan, and the Windower runs before the splitter in that path. No `DatasetSource` emits anything but `Recording`.

(If recomputation is rejected, the alternative is an explicit second contract `PrecomputedWindowSource → FeatureMatrix` named as an AD-1 port. Either way, the spine must say which.)

---

## C2 — `seeds`, `run_id`, sweep expansion and random-split seeds pull in three directions

- **Severity:** Critical
- **Unit A:** `config/sweep.py` (sweep expander)
- **Unit B:** `runner/execute.py` (per-seed loop, AD-12) + `report`

**How both comply yet clash.**

- AD-5 says `run_id` = hash of the *fully resolved* config, so `seeds: [0,1,2]` is inside `run_id`. It also says sweeps are "a Cartesian product over the listed fields".
- **Dev A** allows `seeds` as a sweepable field (nothing forbids it). A sweep YAML with `seeds: [[0],[1],[2]]` or `seeds: {sweep: [0,1,2]}` expands to 3 configs with 3 `run_id`s, each with one seed and std = 0.
- **Dev B** implements AD-12 ("a run executes once per seed and stores ... mean ± std") and expects one `run_id` with all seeds inside.
- The report gets both shapes. It shows the single-seed runs as 3 separate rows, each with std 0, next to rows that carry a real std.
- **Second clash:** adding a 4th seed later changes `run_id`, so AD-14 resume re-runs seeds 0–2 that were already done. Reruns of identical work are exactly what AD-14 tries to avoid.
- **Third clash:** AD-12 says the `random` splitter draws from `config.seeds`. Then one run has N split plans, so N `split_hash`es and N FeatureCache keys. But AD-13 provenance holds *a* `split_hash`, and AD-7 promises one `window_id` set per `split_hash`. A time_block run and a random run aren't even shaped alike.

**Proposed rule fix (amend AD-5, AD-12):**

> **AD-12 (rewritten).** `config.seeds` is a list consumed only by training-time stochasticity (head initialisation, MLP batching, any stochastic extractor), never by data assignment. A splitter that needs randomness takes its own `split.params.seed: int`. That makes it part of the split and of `split_hash`, and one run has exactly one `split_hash`. Per-seed outputs are written to `runs/<run_id>/seeds/<seed>/{metrics.json,predictions.parquet}`. `runs/<run_id>/metrics.json` holds the aggregate (mean, std, n).
>
> **AD-5 (add).** `run_id` hashes the resolved config **excluding `seeds`**. The seed list is stored in `config.yaml` and the per-seed directory names. A sweep may not list `seeds` as an expanded field (`ConfigError`). Resuming a `done` run with a seed it lacks runs only the missing seeds.

---

## C3 — Room selection (r1 / r2 / mixed) has no owner; split and window ids diverge across selections

- **Severity:** Critical
- **Unit A:** `adapters/datasets/tor_vergata.py`
- **Unit B:** `adapters/splitters/time_block.py` / `report`

**How both comply yet clash.** CAP-3 requires results "for room 1, room 2 and both rooms mixed". The spine never says where room selection lives. AD-10 mentions an undefined "dataset selection". Three AD-compliant designs exist:

1. **Dev A** gives `tor_vergata.Params` a `rooms: list[str]` field, and the source emits only r1 recordings. `time_block` then writes `splits/tor_vergata/<hash>.json` over that subset. The r1 run, r2 run and mixed run get **three different `split_hash`es**, even if the per-recording blocks are byte-identical. AD-7's "same `window_id`s for a given `(split_hash, …)`" no longer guarantees that r1 windows are the same in the r1 run and in the mixed run. AD-10 also stores three copies of every LWM embedding.
2. **Dev B** (report) always runs mixed and derives room-1 metrics by filtering `predictions.parquet` on room. That is a head trained on both rooms and scored on r1, which is a different experiment from "room 1". Both are labelled "room 1".
3. A third dev puts selection in a `Splitter` param, which makes the splitter depend on something other than the recording. That's arguably allowed by AD-6, which only bans model, preprocessing, pooling and window dependence.

On top of this, any train-fitted preprocessing statistic (AD-9 global mean) differs between the r1-only and mixed train sets. So the "r1" embeddings themselves differ by design choice.

**Proposed rule fix (new AD, plus AD-6 tightening):**

> **AD-6a — Split first, select second.** A `SplitPlan` is computed once per dataset over **all** of its recordings (`splits/<dataset>/<split_hash>.json`). Which recordings a run uses is a top-level `ExperimentConfig.selection` (e.g. `{rooms: [r1]}`). The runner applies it as a filter on recordings *after* splitting, and it never changes the plan. For one dataset, every run in the CAP-5 table therefore references the same `split_hash`. `selection` is part of `run_id` and of the FeatureCache key (replacing the undefined "dataset selection"). Train-fitted statistics are fitted on the selected train blocks. "Train on mixed, score on one room" is a separate, explicitly named report view (`score_subset`), never the "room 1" cell.

---

## C4 — `max_window_len` is coupled to every future sweep, and changing it silently re-splits

- **Severity:** High
- **Unit A:** `time_block` splitter
- **Unit B:** a later sweep YAML author adding `window_len: 100`

**How both comply yet clash.** AD-6 says `max_window_len` is "set to the largest window used anywhere in the study".

- **Dev A** implements this faithfully by deriving it: either it scans `experiments/*.yaml`, or the runner passes the max over the current sweep.
- A later sweep adds W=100. The gap grows, so the `split_hash` changes and every earlier run now references an older split. The report mixes rows from two splits under one "identical splits" table, and AD-6 has no rule stopping that.
- **Dev B** (another reading) hard-codes 50. A W=100 run is still legal under AD-7, because the window fits inside one block. But its gap is shorter than one window, which quietly weakens the spec's "gap at least one window long" constraint.

**Proposed rule fix (amend AD-6):**

> `gap_packets` is an explicit, committed param of each dataset's canonical split file, not derived from configs. Config validation raises `ConfigError` if `windower.window_len > plan.gap_packets`. Changing the gap creates a new canonical split, and `report` refuses to put rows with different `split_hash` values in one table group (per dataset) unless `--allow-mixed-splits` is passed. The flag is shown in the table footer.

---

## C5 — `Preprocessor` port shape (fit vs pure, single vs chain, ordering) is unspecified

- **Severity:** Critical
- **Unit A:** `adapters/preprocess/rescale.py` (global mean, AD-9 "fits on train blocks only")
- **Unit B:** `runner/compose.py` + `adapters/preprocess/phase_sanitize.py`

**How both comply yet clash.**

- **Dev A** implements `fit(recordings, plan) → Fitted` + `transform(rec) → Recording`, as AD-9's train-only fitting implies.
- **Dev B** writes `phase_sanitize` as `__call__(rec) → Recording`, since it is stateless per packet. Dev B also has the runner call `preprocessor(rec)` for a *single* configured adapter, which matches AD-1's singular `Preprocessor` port and AD-3's "selects adapters by `name` plus `params`".
- Variant 3 in `preprocessing.md` ("option 1 or 2 plus phase sanitization") plus antenna selection is a **chain of 2–3 steps**. One dev builds a `composite` adapter with a `steps` param. Another makes `preprocess:` a list in config. A third writes `rescale_then_sanitize.py`.
- Order matters. A global mean amplitude computed before antenna selection (over 4 antennas) differs from one computed after (2 antennas). Phase sanitization before or after power normalisation is equivalent, but amplitude-only before rescale is not.
- AD-10 hashes "preprocessing config". If one implementation canonicalises the chain as a sorted-key dict, two orders hash identically and share a cache entry.
- Also unstated: does "train" for fitting include the AD-11 validation carve-out? And are fitted values recorded anywhere?

**Proposed rule fix (amend AD-9, AD-4):**

> The `Preprocessor` port is `fit(recordings: Sequence[Recording], plan: SplitPlan) -> FittedPreprocessor` and `FittedPreprocessor.transform(rec: Recording) -> Recording`. Stateless steps implement `fit` as a no-op. `ExperimentConfig.preprocess` is an **ordered list** of `{name, params}` steps that are applied in order and hashed as a list (order-sensitive). Antenna selection, if present, must be the first step. Fitting uses windows/packets of split `train` (including the validation carve-out, so validation scores are optimistic by one statistic, which is accepted). Fitted values (e.g. the mean amplitude) are written to `provenance.json.preprocess_fitted`.

---

## C6 — Antenna selection and phase are consumed with different assumptions by different extractors

- **Severity:** High
- **Unit A:** `adapters/preprocess/antennas.py` (`keep: [2,3]` or `n: 2`)
- **Unit B:** `adapters/extractors/classical_paolo.py` / `wavesfm` / `imagenet` / `mantis`

**How both comply yet clash.**

- AD-9 puts antenna selection in the Preprocessor, and `Recording.csi` loses the original antenna identity.
- `processing.py` (the likely CAP-4 baseline) beamforms **antennas 0 and 1 specifically, using phase**.
- **Dev B** writes `classical_paolo` to take `csi[0:2]`. Under a 4-antenna config, B silently ignores two antennas. Under `antennas.keep=[2,3]`, B beamforms the wrong pair. Under `amplitude_only` (AD-9 shared step), the 40 phase weights become meaningless. All three runs "succeed" (AD-7 only fails a window the extractor *cannot* process).
- The amplitude-window models face an unowned choice, since the spec says "time × subcarrier amplitude" without an antenna axis. Dev C (WavesFM) averages antennas, Dev D (ImageNet) uses antenna 0 as one channel, and Dev E (Mantis) concatenates antennas as channels. Then "2 vs 4 antennas", a declared experiment variable, means something different per model, so the CAP-5 comparison is no longer like-for-like.

**Proposed rule fix (amend AD-4, AD-8):**

> `Recording` carries `antenna_ids: tuple[int, ...]`, the original receiver indices of the rows of `csi`. Antenna selection updates it. Every extractor declares `requires: InputRequirements(phase: bool, antennas: Literal["any"] | tuple[int, ...] | int)`, and the runner checks it at config validation (`ConfigError` before any stage runs). Extractors that take amplitude windows reduce the antenna axis via one shared, named param `antenna_mode: mean | stack_channels | per_antenna_concat`. Its default is fixed in the spine (proposed: `stack_channels`) and recorded in the resolved config.

---

## C7 — `per_window` extractors that compute cross-window statistics leak test data and depend on batch size

- **Severity:** Critical
- **Unit A:** `adapters/extractors/imagenet.py` / `wavesfm` (image normalisation)
- **Unit B:** `adapters/extractors/classical_paolo.py` (`norm = 1` mean normalisation)

**How both comply yet clash.**

- AD-9 forbids statistics on test, but it is scoped to "a preprocessing step". AD-11 puts feature scaling in the head. Nothing covers an **extractor** that normalises its *input* using statistics over many windows.
- **Dev A** min-max scales amplitude windows to [0, 255] "per dataset" before resizing to 224×224. It computes the min and max over the `WindowSet` it receives. That `WindowSet` contains test windows, which is legal because AD-7 says extractors consume the given `WindowSet`.
- **Dev A′** does it per batch, so features depend on `batch_size`. `batch_size` is in extractor params, so it is in the AD-10 key and in `run_id`. The cache isn't stale, but two `batch_size`s give two "results".
- **Dev B** ports `norm = 1` from `processing.py`, which divides by a mean amplitude over the whole recording, including its test block.
- Each is "model-specific input formatting inside the extractor" (AD-9), so each complies.
- The same question arises for any classical step that "keeps the best weight": is the best weight chosen per window (fine) or with labels across the dataset (training inside an extractor)?

**Proposed rule fix (new rule in AD-8):**

> **Extractors are per-window pure.** The output for a window (or packet) depends only on that window's (packet's) samples, the extractor params and the weights. It never depends on other windows, batch composition, labels or split. Allowed: fixed constants (e.g. ImageNet mean/std) and per-window statistics. Any statistic across windows must be a `Preprocessor` step (fitted on train per AD-9) or live in the `Head`. Every extractor's contract test asserts that extracting a window alone equals extracting it inside a shuffled batch of a different size (within float tolerance). `batch_size` and similar execution-only params are marked `affects_result=False` (see M2).

---

## C8 — Per-packet output has no contract, so LWM and the Pooler disagree on shape and token choice

- **Severity:** High
- **Unit A:** `adapters/extractors/lwm/adapter.py`
- **Unit B:** `adapters/poolers/mean_std.py`

**How both comply yet clash.**

- AD-4 lists the contracts, and the per-packet output is not among them. `FeatureMatrix` is row-aligned to `window_id`, not to packets. AD-8 says the per-packet output "is cached per packet" and the Pooler "reduces each window's packet features".
- CAP-2 asks for **both** CLS `(n,128)` and channel embeddings `(n,59,128)`.
- **Dev A** caches the full `(n_packets, 59+1, 128)` tensor (1.9 GB float32 for 63k packets, borderline at 11 GB RAM), keyed per dataset.
- **Dev B** writes `mean_std` expecting `(n_packets, d)` per recording and indexes rows by `start:end`.
- Who decides CLS vs mean-of-channel-tokens? It could be a LWM param, a Pooler param, or a second pooler stage. A Pooler that has to know about tokens has learned model internals, which goes against AD-8's model-agnostic intent.
- Who maps window `(start, end)` to rows? If A drops gap packets to save time (they belong to no window), row index stops equalling packet index. AD-4's "packet indices into the packet axis as saved" then breaks only for the cache.

**Proposed rule fix (amend AD-4, AD-8):**

> Add contract `PacketFeatures`: per `recording_id`, float32 `(n_packets, d)` with row *i* = packet *i* of the recording as saved, covering every packet (including gap packets). A `per_packet` extractor emits exactly this. Token-level reduction (`cls`, `mean_tokens`, `cls+mean_tokens`) is an extractor param. Poolers take `(PacketFeatures, WindowSet) → FeatureMatrix` and know nothing about models. Raw token tensors are not cached in this phase.

---

## C9 — Validation carving has no owner, and `WindowSet.split` has no `val`

- **Severity:** Critical
- **Unit A:** `adapters/heads/logreg.py`
- **Unit B:** `adapters/heads/mlp.py` (and `time_block` splitter)

**How both comply yet clash.**

- AD-11 says hyperparameter selection uses "a validation part carved from `train` blocks with the same time-block + gap rule". It doesn't say who carves.
- The Head receives a `FeatureMatrix` aligned to `window_id`. To apply a packet gap, a head needs each window's `(recording_id, start, end)` and the gap size, so it needs the `WindowSet` and the `SplitPlan`. Neither is in the head's interface.
- **Dev A** passes the `WindowSet` into the head and carves the last 20% of each recording's train block, with a `window_len` gap.
- **Dev B** uses sklearn `GridSearchCV` with `TimeSeriesSplit` over row order. That carves a different fraction, across recordings, and with no gap, so near-duplicate stride-1 windows leak into validation.
- Both read AD-11 as satisfied. The two heads now pick hyperparameters on different data, which goes against AD-11's own purpose ("not some adapters but not others").
- Edge case: if `time_block` makes **one** train block per recording, then for the shortest file (r2 empty, 2121 packets) with a 70/30 split, gap 50 and W=50, the val carve leaves ~1,400 train packets. A val share below about 1 window yields an empty validation set for that recording. Room 1 "empty" has a single recording, so a carve that drops a class entirely from validation is possible with a large gap.
- Unstated: after selection, does the head refit on train ∪ val?

**Proposed rule fix (amend AD-6, AD-11):**

> The `Splitter` emits a three-way plan `train | val | test`. Inside each recording's train region it carves `val` as a contiguous block with the same `gap_packets`, sized by `split.params.val_fraction`. `val` is part of `split_hash`. The plan fails with `DataError` if any class (per selected room) has zero `val` or zero `test` windows at the largest allowed `window_len`. `WindowSet.split ∈ {train, val, test}`. Heads receive `fit(X_train, y_train, X_val, y_val)` and never carve. After selection, a head refits on train ∪ val with the chosen hyperparameters, and the chosen values are written to `metrics.json.selected_hparams`.

---

## C10 — `window_id` format and Windower anchoring are undefined

- **Severity:** High
- **Unit A:** `core/windowing.py` (Windower)
- **Unit B:** `report` / `results_api` (cross-run joins, per-room views)

**How both comply yet clash.**

- AD-7 promises "the same `window_id`s" for a given `(split_hash, window_len, stride)` but never defines a `window_id`.
- **Dev A** numbers windows `0..n-1` in iteration order, anchors at each block start and drops the trailing partial window.
- **Dev B**, writing a paired comparison (e.g. McNemar between LWM and WavesFM) or the C3 `score_subset` view, assumes `window_id` is meaningful across runs with different `selection`s. Window 0 of the r1-only run is not window 0 of the mixed run.
- A second Windower adapter (AD-1 allows several) could anchor on a global grid (`start % stride == 0`) and produce different windows for the same params.
- Also, one `stride` applies to train and test alike. Stride 1 on test (the paper's choice) makes test windows near-duplicates and inflates the effective n. That should be a deliberate choice, not a default.

**Proposed rule fix (amend AD-7):**

> `window_id = f"{recording_id}:{start}:{end}"`, so it is globally unique and stable across runs. The Windower is a pure function in `core/windowing.py` (there is no Windower adapter choice in this phase). Windows are anchored at each split block's first packet, advance by `stride`, and windows that don't fit are dropped (with counts logged). `stride` may be set per split (`stride_train`, `stride_eval`), and the defaults are fixed in the spine.

---

## C11 — The report picks variants and groups rows by unstated keys (a test-set selection path)

- **Severity:** Critical
- **Unit A:** `runner/cli report` (CAP-5 table)
- **Unit B:** `results_api` / notebooks (the same numbers, viewed differently)

**How both comply yet clash.**

- CAP-5 wants "every method per room and setting". The spine leaves "method", "setting", which head and which preprocessing variant undefined. AD-11 keeps the *head* from tuning on test, but nothing stops the *report* from tuning on test.
- **Dev A** builds one row per extractor and fills each cell with the best preprocessing × pooler × head variant *by test accuracy*. That is test-set model selection at reporting level, and it happens only for methods with many variants (LWM has 4 preprocessings × 2 antennas × poolers, classical has 2). LWM is advantaged by multiple comparisons.
- **Dev B**'s notebook shows all variants, so the thesis table and the notebook disagree on "LWM's number".
- Grouping keys are also unstated: `run_id` changes with any param, so two runs differing only in a no-effect param (M2) appear as two rows.
- Dirty-tree exclusion (AD-13) interacts with resume (H3).

**Proposed rule fix (amend AD-13):**

> The CAP-5 table schema is fixed. The row key is `method = (extractor.name, extractor.params_hash, pooler, head)`. The column key is `(dataset, selection, window_len)`. Variant axes (`preprocess`, `antennas`) are either shown as separate rows, or collapsed by **best validation metric** (`metrics.json.val_macro_f1`), never by a test metric. The collapse rule and the chosen variant are printed per cell. Cells show mean ± std over seeds with n. `report` is the only place that defines this mapping, and `results_api` exposes the same grouping function so notebooks cannot drift.

---

## H1 — Two owners of the FeatureCache read/write path

- **Severity:** High
- **Unit A:** `adapters/extractors/lwm/adapter.py`
- **Unit B:** `runner/execute.py`

**How both comply yet clash.**

- The diagram draws `FX <--> FC`, which suggests extractors talk to the cache.
- AD-2 forbids an adapter importing another adapter, and `FeatureCache` is an adapter (`storage/feature_cache.py`). But an extractor can legally receive a `FeatureCache` *port* by injection.
- **Dev A** does exactly that and caches inside the adapter, computing the key from its own view of the config (it doesn't see `selection` or `split_hash`).
- **Dev B** wraps every extractor call in the runner with get-or-compute.
- The result is a double cache under two key schemes. Worse, A's key omits `split_hash`, so train-fitted preprocessing (C5) can be reused across splits. That's a stale-cache bug that AD-10's "deleting the cache never changes a result" would catch only if someone tried it.

**Proposed rule fix (amend AD-10):**

> Only the runner calls `FeatureCache`. Extractors are pure (C7) and never receive a cache, store or path. Keys are computed only by `core.hashing.feature_key(resolved_config_subset)`, from one function used for both read and write. A contract test runs one experiment with a warm cache and with `.cache/features` deleted, and asserts byte-identical `metrics.json`.

---

## H2 — Code changes never change `run_id` or cache keys

- **Severity:** Critical
- **Unit A:** `adapters/extractors/classical_paolo.py` (bug fix: the CSI-distance loop in `datasets.md`, "Known issues")
- **Unit B:** `runner` sweep resume (AD-14) + `FeatureCache` (AD-10)

**How both comply yet clash.**

- **Dev A** fixes the distance loop, a pure code change. The resolved config is unchanged, so `run_id` is unchanged. AD-10's key has "weights revision" but no code version, so the cache key is unchanged too.
- **Dev B**'s resume logic (AD-14) skips every `done` `run_id`, and the cache serves pre-fix features.
- The thesis table keeps pre-fix numbers, and `provenance.json` shows the *old* commit. That is "traceable", but the table claims to be regenerable with one command, and running that command reproduces the stale result from the cache.
- The same applies to edits in `_vendor/` that `SOURCE.md` records (a "local edit") and to preprocessing/pooler/head code.

**Proposed rule fix (amend AD-3, AD-5, AD-10):**

> Every adapter declares a class constant `VERSION: str`, bumped on any change that can alter its output. The resolved config records `{stage, name, version}` for each adapter, so the version enters `run_id` and every cache key. Extractors with `_vendor/` also include the SHA-256 of the `_vendor/` tree. A contract test fails if `_vendor/` changes without a `SOURCE.md` update. Report rows whose adapter versions differ from the current code are flagged `stale`.

---

## H3 — The dirty-tree filter (AD-13) and sweep resume (AD-14) together create permanent holes

- **Severity:** High
- **Unit A:** `runner` sweep skipper (AD-14)
- **Unit B:** `report` dirty filter (AD-13)

**How both comply yet clash.**

- A run completes on a dirty tree, so it is `done` with `dirty: true`.
- **Dev A** skips it forever, as AD-14 says ("skips any `run_id` already `done`").
- **Dev B** excludes it from the thesis table by default, as AD-13 says.
- After committing, re-running the sweep does nothing, and the CAP-5 cell stays empty until someone remembers `--force` for that specific `run_id`.

**Proposed rule fix (amend AD-14):**

> A sweep skips a `run_id` only if it is `done` **and** its provenance is clean. A `done`+dirty run is re-executed when the tree is clean, and the dirty result is kept under `runs/<run_id>/superseded/<timestamp>/`. `--force` still re-runs clean `done` runs.

---

## H4 — `role` has no defined home, and reference vs comparison is enforced nowhere

- **Severity:** High
- **Unit A:** `adapters/datasets/paper_reference.py`
- **Unit B:** `report`

**How both comply yet clash.**

- AD-16 says results "carry `role: reference`" but not where `role` lives.
- **Dev A** sets it as a `DatasetSource` attribute.
- **Dev B** reads `config.role` from the `ExperimentConfig`, which defaults to `comparison`.
- Every reference run then shows up as a comparison row. Conversely, a user can run `tor_vergata` + `random` with `role: comparison`, which puts leaky numbers in the like-for-like table. That is the exact failure AD-16 exists to prevent.

**Proposed rule fix (amend AD-16):**

> `role: comparison | reference` is a required top-level `ExperimentConfig` field. Validation enforces `role == reference ⇔ splitter.unit == window` (C1). `report` partitions only on this field.

---

## M1 — The Pooler is set for `per_window` extractors, and sweeps produce invalid combinations

- **Severity:** Medium
- **Unit A:** `config/sweep.py`
- **Unit B:** `runner/compose.py`

**How both comply yet clash.**

- A sweep lists `extractor: [lwm, wavesfm]` × `pooler: [mean, mean_std]`.
- The expander builds 4 configs. The runner, per AD-8, bypasses the Pooler for WavesFM. That leaves two WavesFM runs with different `run_id`s and identical results, so they become two report rows.
- `window_len: 1` (per-packet classification) × WavesFM or Mantis gives an image or series of length 1. AD-7 calls that a run failure, so the sweep fills with expected failures that look like bugs.

**Proposed rule fix (amend AD-5, AD-8):**

> `pooler` must be `null` iff the extractor is `per_window`. The sweep expander normalises it to `null` before hashing and de-duplicates by `run_id`. Each extractor declares `min_window_len` (and optional `max_window_len`). The expander drops configs that violate it, logs them as `skipped_invalid`, and doesn't run them. Sweeps may also declare `zip:` groups and `exclude:` predicates in addition to the Cartesian product.

---

## M2 — Execution-only params enter `run_id` and cache keys

- **Severity:** Medium
- **Unit A:** extractor `Params` (`batch_size`, `num_workers`, device)
- **Unit B:** `core/hashing` (`run_id`, cache key)

**How both comply yet clash.**

- The Memory convention requires a `batch_size` param. AD-5 hashes the "fully resolved config including defaults", and AD-10 hashes "extractor name + params".
- Dropping `batch_size` from 64 to 16 to fit RAM changes the `run_id` and the cache key. The result is a cache miss, a full re-embed and a duplicate report row.
- Separately, adding a new param with a default to any adapter changes *every* `run_id` for that adapter, and AD-14 then re-runs the whole study.

**Proposed rule fix (amend AD-3, AD-5):**

> Params fields are tagged `Field(json_schema_extra={"affects_result": False})` for execution-only settings. These are excluded from `run_id` and cache keys, and recorded in provenance. Adding a result-affecting param requires a `VERSION` bump (H2). A default that reproduces the old behaviour is excluded from the hash when equal to the default, so existing `run_id`s survive additive changes.

---

## M3 — `Recording` metadata ownership: sample rate, downsampling, packet counts

- **Severity:** Medium
- **Unit A:** `adapters/datasets/tor_vergata.py`
- **Unit B:** a CAP-1 "counts" story / `report`

**How both comply yet clash.**

- The spec requires per-recording packet counts and, if the professor confirms the rates differ, an equalisation rerun. `csi_saver.py` shows per-file downsample factors (r2_standing_2 by 5, r2_standing_1 by 3, r1 sitting by 2).
- `Recording` has no field for any of this.
- **Dev A** hard-codes the factors in the adapter. **Dev B** re-derives counts by loading the `.npy` files in the report.
- The deferred "equalise sample rates" Preprocessor then has nothing to read, and resampling is forbidden by AD-4 ("no stage may ... resample packets"), which contradicts the Deferred table that puts it in a `Preprocessor`.

**Proposed rule fix (amend AD-4):**

> `Recording` adds `meta: RecordingMeta(n_packets, sample_rate_hz: float | None, downsample_factor: int | None, source_path)`, owned solely by the `DatasetSource`. AD-4's no-resample rule gets an explicit exception: a `Preprocessor` step marked `changes_packet_axis=True` may resample, it must run before splitting, and the split is then computed on its output (its config enters `split_hash`).

---

## M4 — `recording_id` collisions and single-label `Recording`s for public datasets

- **Severity:** Medium (deferred CAP-6, but the contract is fixed now)
- **Unit A:** a future `datasets/wimans.py`
- **Unit B:** `Splitter`, `report`

**How both comply yet clash.**

- `recording_id = "<dataset>/<file stem>"` collides when a public dataset reuses stems across directories (e.g. `user1/act1.mat`, `user2/act1.mat`). It also cannot express a file that contains several labelled segments, because `Recording.label` is one value.
- **Dev A** splits a file into segments named by stem. That gives duplicate ids, and the splitter and window ids silently merge them.

**Proposed rule fix (amend AD-4):**

> `recording_id = "<dataset>/<relative path without extension>[#<segment>]"`, and each `DatasetSource` must assert uniqueness. Per-class metrics are keyed by class **name** (from the source's class list), not by index, so cross-dataset reports cannot mislabel classes.

---

## M5 — The `random` splitter is named in AD-6 but contradicts AD-6's own rule

- **Severity:** Medium (it subsumes into C1, but is worth fixing even if C1 is solved differently)

AD-6 defines splitting as contiguous blocks with gaps and then says "`random` exists only for the paper reference row". A developer implementing `random` must violate the AD's own rule text.

**Fix:** as in C1, rename the splitter to `random_window` and add the `unit: window` plan type, legal only for `role: reference`.

---

## Summary of proposed amendments

| AD | Change |
| --- | --- |
| AD-4 | Add `PacketFeatures`, `antenna_ids`, `RecordingMeta`. Make `recording_id` path-based. Key per-class metrics by name. |
| AD-5 | Exclude `seeds` and `affects_result=False` params from `run_id`. Normalise `pooler=null`. Add sweep `zip`/`exclude`. |
| AD-6 | Add `gap_packets` as a committed param. Split all recordings, then select (new AD-6a). Use three-way `train/val/test`. Add `unit: window` plans for the reference row only. |
| AD-7 | Define `window_id = recording_id:start:end`. Fix anchoring. Allow per-split stride. |
| AD-8 | Extractors are per-window pure. Make token reduction an extractor param. Poolers are model-agnostic. Add `min_window_len`. |
| AD-9 | Preprocessor is `fit/transform`. Preprocessing is an ordered list. Fitted values go to provenance. |
| AD-10 | The runner is the sole cache client. Keys include adapter `VERSION` and the `_vendor` hash. Replace "dataset selection" with `selection`. |
| AD-11 | Heads never carve. Heads get `fit(train, val)` and refit on train∪val. |
| AD-12 | Split seeds live in split params. `config.seeds` covers training only. Store per-seed outputs. |
| AD-13 | Fix the report schema. Collapse variants only by validation metric. Share the grouping function with `results_api`. |
| AD-14 | Re-run `done`+dirty runs on a clean tree. |
| AD-16 | Recompute the paper method from `saved_csi`. Use `saved_dataset` only as a fixture. Make `role` a required config field validated against the splitter. |
