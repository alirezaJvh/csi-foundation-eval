---
reviewed: ARCHITECTURE-SPINE.md (updated 2026-10-06)
against: SPEC-lwm-wifi-har (SPEC.md, lwm-models.md, preprocessing.md, candidate-models.md, datasets.md) + .memlog.md
date: 2026-10-06
kind: reconcile + rubric
---

# Review: reconcile and rubric

## Verdict

The spine is solid on most of the dependency, ownership and provenance structure, and most memlog decisions landed. It is **not ready for epics yet**. Three areas need fixing first. (1) The paper reference row as specified contradicts AD-4, AD-6, AD-7 and AD-11: precomputed features have no packets, a random split has no blocks, and the paper used a random forest. (2) Several real divergence points are left open: the per-packet feature contract, who carves validation, how preprocessing steps compose, how the report picks variants, and how adapters register. (3) `run_id` and cache keys ignore code version, which undermines the "one-command regeneration / every number traceable" promise. Several quiet spec details have no home: packet-count reporting, per-class/per-room counts, the NaN/constant sanity check, the r1/r2/mixed settings, window 40 and per-packet windows, the published-numbers source, and Open Questions.

Severity counts: 1 critical, 7 high, 11 medium, 9 low.

---

## Part 1: Reconcile (spec + companions + memlog → spine)

### Coverage table

| Source item | Status in spine | Finding |
| --- | --- | --- |
| CAP-1: 19 recordings load with class/room/recording labels | Landed (AD-4 `Recording`) | — |
| CAP-1: per-class and per-room counts reported | **Missing** | M1 |
| CAP-1: splits reproducible from a split file | Landed (AD-6, `splits/`) | — |
| CAP-2: CLS **and** channel embeddings, documented shape | **Missing**: no per-packet contract, CLS vs channel not decided | H3 |
| CAP-2: 2-antenna and 4-antenna variants | Partly: antenna selection is a Preprocessor op, but which 2 antennas is not fixed | M2 |
| CAP-2: sanity check (not constant / NaN) | **Missing** | M2 |
| CAP-3: acc, macro-F1, confusion matrix | Landed (AD-11 Evaluator) | — |
| CAP-3: room 1, room 2, mixed | **Missing** from spine (memlog decision lost) | H6 |
| CAP-3: every preprocessing × antenna × window variant | Partly (sweeps), but preprocessing chains can't be expressed | H5 |
| CAP-4: same metrics on same splits | Landed in principle, but classical inner windows and cleaning can break it | M3, M4, H7 |
| CAP-4: classical `norm = 0 / 1` variant | Not mentioned (should be an extractor param) | L8 |
| CAP-5: one table, every method per room and setting | Partly: table shape and variant selection undefined | H4 |
| CAP-5: paper method as reference row | Contradicts AD-4/6/7/11 | C1 |
| CAP-5: published numbers shown alongside | Stated (AD-16), but no source of truth for the numbers | M11 |
| CAP-6: loader only, no eval change | Landed (AD-1/3/4) | — |
| CAP-7: adding a model = new adapter only | Partly: registration mechanism undecided, may touch runner | M5 |
| CAP-7: weights pinned / traceable | LWM and WavesFM only; Mantis and ImageNet unpinned | M8 |
| Constraint: rescaling mandatory for LWM | Covered by variants; nothing guards against unscaled input | L4 |
| Constraint: domain gap reported, not hidden | **No home** | L4 |
| Constraint: frozen weights, same head family | Head family landed (AD-11). "Frozen" isn't enforced | L6 |
| Constraint: identical windows for window and per-packet models | Landed (AD-7, AD-8) | — |
| Constraint: no resampling | Landed (AD-4), but contradicted by Deferred | H7 |
| Constraint (rate guard): report each recording's packet count | **Missing** | M1 |
| Constraint: time-block primary split, gap ≥ one window | Landed (AD-6) | — |
| Constraint: random split only to reproduce paper | Landed (AD-6/16), but inconsistent with AD-6's block wording | C1 |
| Constraint: variants as experiments (2 vs 4 ant., preprocessing, per-packet + several pooled lengths) | Partly: per-packet window semantics undefined | M3 |
| preprocessing.md: window 40 to match paper W2 | **Missing**; also the "40" means different things for classical vs LWM | M3 |
| preprocessing.md: variant list 1–4 (global rescale, per-packet power, +phase sanitize, amplitude-only) | Seed lists modules, but "1 or 2 **plus** 3" needs ordered composition | H5 |
| Constraint: switchable + traceable | Landed (AD-3/5/13), weakened by H1 | H1 |
| Constraint: CPU only, 11 GB | Landed (stack, memory convention, deferred GPU) | L9 |
| Constraint: public datasets only after verification | Landed (Deferred) | — |
| Success signal: each row regenerated with one command from saved config | **No explicit command / guarantee** | M11, H1 |
| Assumption: split + window strategy to confirm with supervisor | **Missing** (no Open Questions section) | M10 |
| Open Q: Paolo instructions, watch-list weights, sample rate | Partly in Deferred; no Open Questions section | M10 |

### Memlog decisions

| Memlog entry | Status |
| --- | --- |
| Paradigm, dependency rule, registry, immutable contracts | Landed (AD-1..4) |
| Functional core: preprocessing, windowing and metrics are pure functions, with I/O only at the edges | **Partly**: core lists windowing, metrics and hashing; preprocessing purity and the "I/O only at edges" rule are not stated (L3) |
| Cache + results store behind repository interfaces, content-hash keys | Landed (AD-10, AD-13) |
| ExperimentConfig, two front ends, run_id = hash | Landed (AD-5) |
| Resumable, failure-isolated sweep | Landed (AD-14) |
| No reported number computed in a notebook | **Narrowed**: AD-13 restricts only the CAP-5 table (L3) |
| Classical = extractor, public = DatasetSource, paper ref = own source + random + role | Landed (AD-16), but internally inconsistent (C1) |
| Builder only as config builder | Landed |
| pydantic + PyYAML + in-house sweep expander, Hydra rejected | Landed; sweep grammar undefined (M9) |
| ACL vendoring, including **LWM patching code** | **Contradicted**: AD-15 says "model definition only, no training or data code" (L2) |
| Pooling option (b) | Landed (AD-8) |
| Split ownership, gap ≥ largest window | Landed (AD-6); evolving max window not handled (M10) |
| Window ownership | Landed (AD-7) |
| Heads, inner validation, scaling on train, seeds | Landed (AD-11/12); validation ownership unclear (H2) |
| Results layout; **each room setting (r1, r2, mixed) is a dataset selection = separate run** | Layout landed; **room-setting rule did not land** (H6) |
| Dirty tree: run, record, exclude | Landed (AD-13); conflicts with AD-14 skip (M7) |
| uv, 3.12, CPU torch, data stays, .cache/runs ignored, ruff, ty | Landed |
| Source tree seed | Landed (+ `splits/`) |
| Model integration; "no git submodules, no separate venvs unless conflict" | Integration landed; the no-submodule/no-venv rule did not (L7) |
| (question) Licenses of LWM / WavesFM code | **Did not land** (M10) |
| Direction: supervisor-facing pipeline explainer deliverable | Not present or linked (`companions: []`) (L7) |

---

## Part 2: Findings

### Critical

**C1: The paper reference row cannot be built without breaking AD-4, AD-6, AD-7 and AD-11**
- **Location:** AD-16, AD-6 (last sentence), AD-4 (`paper_reference` label line), AD-7, AD-11.
- **Issue:** The four rules conflict with the reference row in four places:
  - `paper_reference` reads `saved_dataset/*.npy`. Those files are rows of 6 features plus a label, with no `recording_id`, no packets and no packet indices. They cannot be a `Recording` (AD-4 requires complex `(antennas, packets, subcarriers)`), cannot be split into "contiguous packet blocks" (AD-6), and cannot be windowed by the Windower (AD-7).
  - AD-6 defines the Splitter as assigning packet blocks, but `random` is a row-level split with no blocks or gap, and AD-7 requires windows to lie inside a block.
  - The published 97–99% numbers come from StandardScaler + **random forest** with 4 features chosen by MRMR (`classification.py`). AD-11 limits the head family to `logreg` and `mlp`, so the paper's numbers can't be reproduced.
  - The published numbers use 4 features, while `saved_dataset` has 6.
- **Impact:** Whoever builds the reference row will either break the contracts ad hoc or write a side script, which is exactly the per-model script AD-1 is meant to prevent.
- **Fix:** Amend the spine with one explicit path:
  - (a) Preferred: compute the reference from `saved_csi` with a `paper_processing` per-window extractor (antennas 0 and 1, `processing.py` logic) and a `random_window` splitter that assigns **windows** (not packets) to train/test. Keep it legal by having AD-6 say: "`random` assigns windows; AD-7's block rule applies only to block splitters; results carry `role: reference`."
  - (b) Alternatively, define a `PrecomputedFeatureSource` port that emits `FeatureMatrix` + labels directly. Make it an explicit AD-1 exception used only with `role: reference`.
  - In both cases, add a `random_forest` head adapter (with optional MRMR top-k as a head param). State in AD-11 that `random_forest` is allowed only for `role: reference` rows unless the classical comparison row also adopts it (see H4).

### High

**H1: Code version is missing from `run_id` and cache keys, so fixes are silently skipped and stale features are reused**
- **Location:** AD-5 (run_id), AD-10 (cache key), AD-14 (skip `done`).
- **Issue:** `run_id` hashes only the config, and the cache key hashes only config pieces plus the weights revision. After a bug fix in `lwm/adapter.py` or `time_block.py`, the sweep skips every `done` run and the cache serves features from the old code. AD-10's "deleting the cache never changes any result" is then false in practice. The success signal (rerun one row and get the same number) can't be told apart from "never reran".
- **Fix:** Give every adapter an `impl_version: int` class attribute that is written into the resolved config (and so into `run_id`) and into the cache key. Bump it on any change that affects output, and add a contract test that a changed golden output with an unchanged `impl_version` fails. Also make the cache key include `split_hash` only when the preprocessing fits statistics (optional optimization).

**H2: Who carves the validation set, and from what, is undecided**
- **Location:** AD-6 ("Splitter alone owns train/test assignment") vs AD-11 ("validation part carved from train blocks with the same time-block + gap rule"); AD-4 `FeatureMatrix`.
- **Issue:** Under AD-11 each head carves validation itself, which duplicates split logic in every head. Heads also receive only a `FeatureMatrix` (rows aligned to `window_id`), without `start`/`end`, so they can't apply a gap rule. Two head stories will carve validation differently, or carve it randomly.
- **Fix:** Make the `SplitPlan` carry three roles, `train | val | test`, with gaps between all of them, produced only by the Splitter. Windows get `split ∈ {train, val, test}`. Heads receive train and val matrices from the runner. Store val metrics in `metrics.json` (they are also needed for H4).

**H3: The per-packet feature contract (LWM output) is undefined**
- **Location:** AD-4 (contract list), AD-8, AD-10, CAP-2.
- **Issue:** AD-8 says per-packet output is cached and pooled, but AD-4 defines only `FeatureMatrix (n_windows, d)`. Several things are left open: the Pooler's input type, the cache file shape, and whether the LWM adapter emits CLS `(n,128)`, channel embeddings `(n, 59, 128)`, or a patch-mean. CAP-2 requires that **both** CLS and channel embeddings can be obtained. Full channel embeddings are about 63k × 59 × 128 × 4 B ≈ 1.9 GB per preprocessing variant, which can't be loaded whole into 11 GB alongside other work. The extractor, pooler and cache stories will each guess.
- **Fix:** Add `PacketFeatures` to AD-4: float32 `(n_packets_total, d)` with a `(recording_id, packet_index)` row index, stored as a memory-mapped `.npy`. Add an LWM param `embedding: cls | channel_mean | cls_channel_mean` (part of the cache key). State that full per-patch channel embeddings are produced only by a CAP-2 inspection command, not cached. Poolers take `PacketFeatures + WindowSet → FeatureMatrix`.

**H4: How the CAP-5 table selects variants and lays out rows is undefined, so selection can leak test data**
- **Location:** AD-13, AD-16, CAP-5, AD-11.
- **Issue:** The spec's table has "every method per room and setting". Each method has many runs (preprocessing × antennas × window × pooler × head). The spine doesn't say whether `report` shows all runs or one row per method, or how that row is chosen. If the best variant is picked by test accuracy, this is test-set tuning across runs, the exact leakage AD-11 forbids inside heads. The report story and the sweep story will decide this independently.
- **Fix:** Add to AD-13/16:
  - `report` groups by `(dataset, room_setting, role, extractor)`. It selects the variant with the best **validation** macro-F1 (from H2) and prints its test metrics as mean ± std over seeds, plus the chosen variant's config summary and `run_id`.
  - A `--all-variants` flag prints the full grid.
  - Define the `index.parquet` columns needed for this (see M6).

**H5: A single `Preprocessor` port can't express the variant list (ordered chains)**
- **Location:** AD-1, AD-3 ("config selects adapters by `name` plus `params`"), AD-9; preprocessing.md variants 1–4.
- **Issue:** The spec's variants are compositions: antenna selection → (global rescale | per-packet power) → optional phase sanitization, or → amplitude-only. Order matters, for example sanitizing phase before vs after selecting antennas, and rescaling vs setting phase to zero. AD-3 implies one adapter per stage. One story will build a monolithic `lwm_preproc` adapter, and another will build a chain.
- **Fix:** In AD-3/AD-9, state that `preprocessing` is an **ordered list** of `{name, params}` steps, applied in order and each implementing `Preprocessor`. The list in its given order is part of `run_id` and of the cache key. Ship named presets (`v1_global`, `v2_packet_power`, `v3a_global_phase`, `v3b_packet_power_phase`, `v4_amplitude_only`) as YAML snippets so sweeps reference the spec's variant names.

**H6: The room settings (r1, r2, mixed) were decided in the memlog but are not in the spine**
- **Location:** Memlog "Each room setting (r1,r2,mixed) is a dataset selection = separate run"; spine has only "dataset selection" in AD-10.
- **Issue:** There is no config field, no definition of "mixed", and no guarantee that r1-only and mixed runs use the same per-recording blocks. If someone computes a split per selection (or hashes the split over the selected recordings), r1 and mixed get different splits and the results can't be compared. CAP-3 and CAP-5 both depend on this.
- **Fix:** Add it to AD-6 (or a new AD):
  - `dataset: {name, rooms: [r1] | [r2] | [r1, r2]}`.
  - The `SplitPlan` is computed once over **all** recordings of a dataset, and a room selection only filters it.
  - `split_hash` doesn't depend on `rooms`.
  - `room_setting` (`r1 | r2 | mixed`) is a derived column in `index.parquet`.
  - Preprocessing statistics are fitted on the train blocks of the selected rooms.

**H7: AD-4's no-resample/no-drop rule conflicts with the Deferred rate-equalization item and likely with Paolo's cleaning**
- **Location:** AD-4 ("No stage may reorder, drop or resample packets") vs Deferred "Equalizing sample rates … would be a `Preprocessor` adapter"; CAP-4 "data-cleaning instructions".
- **Issue:** The Deferred item places resampling exactly where AD-4 forbids it. Paolo's pending cleaning will plausibly drop outlier packets or apply temporal filters. Both change packet indices after the SplitPlan has fixed them, which invalidates `split_hash`-based windows.
- **Fix:**
  - Change the Deferred row to: "Rate equalization is a `DatasetSource` variant (e.g. `tor_vergata_equalized`) that emits resampled recordings with their own ids and its own split. It never runs after splitting."
  - Add to AD-4: "Packet-dropping cleaning is likewise a DatasetSource-level variant. Preprocessors are length-preserving."
  - Mark the classical-pipeline Deferred row as "may require an amendment if cleaning isn't length-preserving".

### Medium

**M1: Dataset statistics reporting is missing (per-class and per-room counts, per-recording packet count)**
- **Location:** CAP-1 success; Constraint "report each recording's packet count" (rate-confound guard); spine CLI lists only `run | sweep | report`.
- **Fix:** Add a `csi-bench describe <dataset>` command (runner CLI, reads through `DatasetSource`). It writes `runs/_dataset/<dataset>.parquet` with one row per recording (`recording_id`, room, class, `n_packets`, train/val/test packet counts per `split_hash`) and prints the per-class and per-room totals. `report` includes this table in its output so the rate guard sits next to the results.

**M2: CAP-2 has no sanity check, and the 2-antenna choice is unfixed**
- **Location:** AD-8, AD-9, CAP-2.
- **Fix:**
  - In the runner, after every extractor call, validate that features are finite and that per-dimension std > ε across rows, else raise `ExtractorError`. Add this to the port contract test.
  - Fix the antenna-selection param as an explicit index list. The 2-antenna default is `[0, 1]`, matching `processing.py` and `saved_dataset`; state this in AD-9 or the conventions.

**M3: Window semantics are incomplete (per-packet variant, length 40, stride, classical inner windows)**
- **Location:** AD-7, AD-8, AD-6; preprocessing.md "per-packet classification … including 40 to match the paper's W2".
- **Issue:**
  - "Per-packet" classification is a spec variant, but the spine doesn't say whether it means `window_len = 1`, nor what a per-window extractor (WavesFM 224×224, Mantis) does with a 1-packet window.
  - Stride isn't defaulted. The paper uses stride 1.
  - In `processing.py`, W2 = 40 is counted in **feature** steps, and each feature looks back W = 16 packets, so a "40" classical window spans about 55 packets of CSI. If `classical_paolo` reproduces this, it reads packets outside the Windower's window and also exceeds the gap.
- **Fix:** In AD-7:
  - `window_len` and `stride` are always in packets.
  - `window_len = 1` is valid only for per-packet extractors; the config validator rejects it for per-window ones.
  - Default `stride = 1` for train and test.
  - The standard sweep includes `window_len ∈ {1, 10, 30, 40, 50}` (40 = paper W2; 10/30/50 match `saved_dataset`).
  - Any internal sub-windows of an extractor must fit inside `[start, end)` (see M4).

**M4: Leakage paths through extractors and temporal preprocessing aren't closed**
- **Location:** AD-7, AD-9.
- **Issue:** AD-7 forbids extractors from moving windows, but doesn't forbid an extractor from **reading** packets outside `[start, end)` (e.g. classical lookback, or the known `processing.py` CSI-distance loop over all packets). AD-9 doesn't restrict temporal preprocessing (smoothing, Hampel filters), whose kernels can cross the gap.
- **Fix:**
  - Add to AD-7: "An extractor receives only the packet slice `[start, end)` per window. The runner passes slices, not whole recordings."
  - Add to AD-9: "Preprocessors are per-packet operations, plus statistics fitted on train. Any operation across time is applied separately within each split block, or lives inside an extractor."

**M5: The adapter registration mechanism is undecided**
- **Location:** AD-2 ("Only runner names concrete adapter classes"), AD-3 ("deleting its module and its registration"); CAP-7 "adding a model needs only a new model adapter".
- **Issue:** The registration mechanism isn't chosen. It could be a decorator in the adapter plus auto-import of the package, an explicit table in `runner/compose.py`, or entry points. With an explicit table, adding a model edits the runner. AD-3's wording implies the registration sits somewhere other than the module.
- **Fix:** Choose one. Recommended: the registry lives in `core/registry.py`. Adapters self-register with `@register("extractor", "lwm")`. `runner` imports `csi_bench.adapters` via `pkgutil.walk_packages`, so adding a model means adding a module and nothing else. Reword AD-3: "deleting its module removes it".

**M6: Results schemas are undefined, and `results_api` duplicates the store layout**
- **Location:** AD-13, AD-2 diagram (`results_api → core` only), AD-16 `role`.
- **Issue:** The layout of `metrics.json`, `predictions.parquet` (per seed?) and the `index.parquet` columns isn't fixed. `results_api` may not import `adapters.storage.results_store`, so it re-implements reading the layout, which means two readers. It's also unclear where `role` comes from (config field or derived).
- **Fix:**
  - Define in `core/contracts.py` (or `core/results_schema.py`) the paths and schemas: `metrics.json` = per-seed `{seed, split: val|test, accuracy, macro_f1, per_class, confusion}` plus aggregates; `predictions.parquet` columns `window_id, seed, y_true, y_pred, proba_*`; `index.parquet` columns `run_id, status, dirty, role, dataset, room_setting, extractor, preprocessing_preset, antennas, window_len, stride, pooler, head, test/val acc and F1 mean ± std, created_utc, git_commit`.
  - Make `role` an `ExperimentConfig` field defaulting to `comparison`.
  - Have both `ResultsStore` and `results_api` use the core schema.

**M7: Dirty runs conflict with sweep skipping**
- **Location:** AD-13 vs AD-14.
- **Issue:** A dirty run reaches `done`. Its `run_id` doesn't include the commit, so the next clean sweep skips it, and `report` excludes it, so the row silently disappears unless the user remembers `--force`.
- **Fix:** Add to AD-14: "A sweep treats `done` runs with `dirty: true` as not done when the tree is now clean (rerun and overwrite)." Alternatively, have `report` list the dirty-only rows as missing with the command to regenerate them.

**M8: Mantis and ImageNet weights aren't pinned**
- **Location:** Stack; AD-15; AD-10 (cache key requires "weights revision").
- **Issue:** Only LWM and WavesFM have pinned revisions. `timm` pretrained weights and `MantisV1.from_pretrained("paris-noah/Mantis-8M")` download from HF at `main`, so AD-10's key has no value to use and provenance is incomplete.
- **Fix:**
  - Add Stack rows: `Mantis-8M weights | HF paris-noah/Mantis-8M @ <sha>` and `ImageNet model | timm '<model_name>' via HF @ <sha>`, choosing one concrete ImageNet model (e.g. `vit_small_patch16_224.augreg_in21k_ft_in1k` or `resnet50.a1_in1k`).
  - Require every extractor param model to carry `weights_revision`.

**M9: The sweep grammar can't handle list-valued params**
- **Location:** AD-5 ("Cartesian product over the listed fields"), conventions.
- **Issue:** `seeds: [0, 1, 2]`, `antennas: [0, 1]` and preprocessing lists are legitimately list-valued, so "listed fields" is ambiguous. Two sweep files (or the builder and YAML) will expand differently.
- **Fix:** Use an explicit `sweep:` block of dotted paths, e.g. `sweep: {"extractor.params.embedding": [cls, channel_mean], "window.window_len": [1, 40]}`. Everything outside `sweep:` is literal. Document this in the conventions table.

**M10: There is no Open Questions section; licenses, supervisor confirmations and the evolving max window are lost**
- **Location:** The spine as a whole; memlog `(question)` licenses; spec Assumptions (the split and window protocol were chosen by Claude and need supervisor confirmation) and Open Questions; AD-6 "largest window used anywhere in the study".
- **Issue:** None of these items carries over into the spine. In addition, `max_window_len` is defined by a study that is still evolving: adding a window of 64 later changes every `split_hash` and invalidates every run.
- **Fix:**
  - Add `## Open Questions` with:
    - code licenses before publishing;
    - supervisor confirmation of the time-block split and window strategy;
    - Paolo's instructions;
    - the professor's answer on sample rate;
    - access to the watch-list weights.
  - In AD-6, fix `max_window_len` as an explicit committed constant per dataset (e.g. `tor_vergata: 64`). Any window above it is a config error. Raising it is a deliberate split version bump.

**M11: One-command regeneration and the published numbers have no concrete home**
- **Location:** Success signal; CAP-5 "published numbers shown alongside"; AD-13, AD-16.
- **Fix:**
  - Specify `csi-bench run runs/<run_id>/config.yaml` (the saved resolved config). It must reproduce the same `run_id` (add a test).
  - `report` prints this command per row.
  - Store the published numbers in a committed `experiments/published_reference.yaml` (r1 99.41, r2 99.59, mixed 97.37; 4 features, W2 = 40, random split; source = PIMRC 2025). `report` reads it.

### Low

**L1: The `window_id` format is undefined.** Location: AD-4, AD-7. "Every model sees the same `window_id`s" needs a deterministic id. Fix: set `window_id = "<recording_id>@<start>:<end>"` (a string), with the WindowSet ordered by `(recording_id, start)`.

**L2: AD-15's vendoring rule contradicts the memlog on LWM patching code.** Location: AD-15 "model definition only, no training or data code" vs memlog "vendor lwm_model.py + needed patching code". Fix: allow "input formatting functions needed for inference (e.g. `patch_maker`), copied without DeepMIMO or dataset I/O".

**L3: The functional-core and notebook rules were narrowed.** Location: Design Paradigm / AD-13 vs memlog. Fix:
- Add to AD-2/AD-9: "Preprocessing, windowing, pooling and metrics are pure functions; I/O only in DatasetSource, FeatureCache and ResultsStore adapters."
- Broaden AD-13: "No number or figure in the thesis is computed in a notebook from anything but `results_api` outputs."

**L4: The domain gap and the rescale requirement have no architectural hook.** Location: spec Constraints. Fix:
- Have the LWM adapter log input amplitude statistics into `provenance.json` and warn when the median magnitude is far from O(1).
- Note in the Capability map that the domain-gap finding is evidenced by the LWM-vs-baselines rows plus the preprocessing-variant grid in `report --all-variants`.

**L5: Data provenance and ops hygiene are thin.** Location: AD-13, ops envelope. Fix:
- Record SHA-256 of each input `.npy` in the `SplitPlan` and provenance.
- Note the disk budget for `.cache/features` and `runs/`.
- State the backup stance for the git-ignored `runs/`: regenerable, but back up `index.parquet` and the thesis rows.

**L6: "Frozen" isn't enforced.** Location: Constraint (frozen weights), AD-8. Fix: add an extractor contract test that the model is in `eval()` and under `torch.inference_mode()`, and that parameter checksums are unchanged before and after extraction.

**L7: Memlog items are missing (no submodules / no extra venvs; explainer deliverable).** Fix:
- Add one line to AD-15: "No git submodules or separate venvs unless a dependency conflict forces it (spine amendment)."
- List the supervisor-facing pipeline explainer under `companions` once written.

**L8: The classical `norm = 0 / 1` variant isn't placed.** Fix: state that `norm` is a `classical_paolo` (and `paper_processing`) param, not a Preprocessor.

**L9: CPU device and thread count are implicit.** Fix: add to AD-12 that the runner sets `device = "cpu"` and `torch.set_num_threads(config.threads, default 16)`, recorded in provenance. Adapters never choose a device.

---

## Part 3: Rubric

| Criterion | Judgement |
| --- | --- |
| Fixes the real divergence points for epics and stories, and misses none | **Partly.** Ownership of split, window, cache and results is well fixed. Missed: the per-packet contract (H3), validation ownership (H2), preprocessing composition (H5), room settings (H6), report variant selection (H4), registration (M5), schemas (M6), sweep grammar (M9) and `window_id` (L1). |
| Every AD's Rule is enforceable and prevents its divergence | **Mostly.** AD-2 has an import-linter test. AD-4 and AD-7 are enforceable through contract tests but have leaks (M4). AD-5/AD-10/AD-14 don't prevent stale results because code version isn't keyed (H1). AD-11's validation rule can't be enforced as written because heads lack window positions (H2). AD-6 doesn't hold for `random` (C1). |
| Nothing under Deferred could let two units diverge | **No.** Rate equalization as a Preprocessor contradicts AD-4 (H7). The classical feature set (blocked on Paolo) may need packet dropping, temporal filtering, a random forest head and lookback windows, each of which hits an AD (H7, M3, M4, C1). Fine-tuning and trainable aggregators are correctly fenced. |
| Every owned dimension is decided, deferred or an open question (including the operational envelope) | **Partly.** Environment (single laptop), runtime, memory, logging, resumability and weight download are decided. Missing: an Open Questions section (M10), disk budget and backups (L5), device and threads (L9), dataset-describe ops (M1). |
| No internal contradictions between ADs, conventions and stack | **No.** AD-6 vs `random` and paper_reference vs AD-4/7/11 (C1); AD-4 vs Deferred (H7); AD-13 vs AD-14 (M7); AD-15 vs memlog (L2); AD-2's `results_api → core` only vs needing the store layout (M6); AD-10 needs a weights revision that the Stack doesn't give for Mantis and ImageNet (M8). |

## Suggested order of amendments

1. C1, H2, H3: contracts (`PacketFeatures`, val split role, reference path, `random_forest` head).
2. H1, M5, H5, M9: config, identity and registry mechanics.
3. H6, H4, M6, M11: room settings, report and schemas.
4. H7, M3, M4: leakage and length-preservation rules.
5. M1, M2, M8, M10, then the low findings.
