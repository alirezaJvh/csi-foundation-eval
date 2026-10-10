# Epic 2 Context: Classical ML harness

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher and Paolo express a full classical-ML comparison as a single declarative sweep — window lengths × feature sets or selection × classifiers × room settings — run it with one command, and get a leakage-free results table with feature-selection stability that notebooks can chart. This epic builds the reusable harness (config, runner, cache, reporting) that every later model epic drops into by adding one adapter, so its correctness and its no-leakage guarantees matter more than any individual result.

## Stories

- Story 2.1: Run one classical experiment end-to-end
- Story 2.2: Preprocessing chain with antenna selection and global rescale
- Story 2.3: Full classifier family with seeds
- Story 2.4: Runner-owned feature cache
- Story 2.5: Paper beamforming feature family
- Story 2.6: Feature selection (manual subsets and MRMR)
- Story 2.7: Sweeps and the Python experiment builder
- Story 2.8: Comparison report and notebook results API
- Story 2.9: Faster feature extraction and head training
- Story 2.10: Compute paper_beamforming inner-window metrics once per packet
- Story 2.11: Share paper_beamforming per-position metrics across the sweep

## Requirements & Constraints

- **Identical treatment, no leakage.** Every method gets the same splits, windows and metrics; no window crosses a split block; extractors read only their own window's packets; fitted statistics, feature selection and hyperparameters use train/val only; test is scored exactly once.
- **Reproducibility and traceability.** A run's id is a hash of its fully resolved config plus adapter versions and vendored-code hashes; every result stores its config and provenance (git commit, dirty flag and diff, package versions, split hash, weights revisions, fitted state); any number can be regenerated with one command.
- **Extensibility.** Adding or removing a dataset, preprocessing step, model, feature family, selector or classifier is one adapter module plus its registration; nothing else changes. Import direction is enforced by import-linter.
- **Device and budget.** Runs on the local CPU only (Ryzen 7, 16 threads, 11 GB RAM). Extractors batch their inputs and never hold all inputs at once; the runner owns device and thread settings.
- **Code quality.** `ruff` and `ty` pass (excluding `_vendor/`), and every port's contract test runs each of its adapters.
- **Feature-selection outcome.** The report must expose how stable each feature's selection is across rooms and seeds.
- **Command surface.** One `run` command per config, a resumable `sweep`, a `report` table, and a read-only results API for notebooks.

## Technical Decisions

- **Pipeline is fixed ports with self-registering adapters.** Ports: `DatasetSource`, `Splitter`, `PreprocessStep`, `Windower`, `FeatureExtractor`, `Pooler`, `FeatureSelector`, `Head`, `Evaluator`, `FeatureCache`, `ResultsStore`. Each adapter registers via `@register("<stage>", "<name>")`, declares a pydantic `Params` model and an integer `VERSION` (bumped when output can change), and is found by `pkgutil` discovery. Params that don't affect results (batch size, threads, device) are excluded from the run id.
- **Config is the single construction path.** YAML and `ExperimentBuilder.build()` both produce one `ExperimentConfig` and never build pipeline objects; the config is validated (pydantic) before any stage runs. `role ∈ {comparison, reference}` is required. Sweeps use an explicit `sweep:` block of dotted paths expanded as a Cartesian product; `seeds` cannot be swept.
- **Immutable contracts, fixed shapes.** Stages exchange only frozen dataclasses from `core/contracts.py`: `Recording` (complex64 `(antennas, packets, subcarriers)`), `SplitPlan`, `WindowSet` (`window_id = "<recording_id>@<start>:<end>"`, ordered by `(recording_id, start)`), `PacketFeatures` (`(n_packets, d)`, row-aligned), `FeatureMatrix` (`(n_windows, d)`, row-aligned, with unique family-prefixed `feature_names`), `Predictions`, `Metrics`. Contracts are never mutated.
- **Windowing.** `window_len` and `stride` are in packets, stride default 1. Every window lies inside one split block; a window longer than `gap_packets` raises `ConfigError`. Extractors declare `min_window_len`; `window_len = 1` is valid only for per-packet extractors. For a given (split, selection, window_len, stride) every model sees the same window ids.
- **Extractors and pooling.** Extractors declare `granularity: per_packet | per_window` and `requires: {phase, n_antennas}`. They are pure per input (single-window output equals batched output) and compute no cross-window statistics. Per-packet extractors return `PacketFeatures` and use a `Pooler`; per-window extractors return a `FeatureMatrix` and must set `pooler: null`. Classical features are families (e.g. `amplitude_stats`, `paper_beamforming`) that emit named columns; several families may be listed and their columns concatenated. After extraction the runner checks features are finite and no dimension is constant, raising `ExtractorError` otherwise. Extractors may declare `uses_context = True` and implement `extract_blocks` to share per-packet intermediates across overlapping windows; its output must equal the slice-based `extract` exactly (contract test).
- **Preprocessing** is an ordered chain of fitted steps; order is part of the hash, so reordering yields a different run id. Each step fits on the selected train blocks only and records fitted state in provenance. Steps operate per packet or per split block. Antenna selection is the step `antennas: [ids]` (2-antenna variant `[0, 1]`). Named presets in `config/presets.py` cover the variants. Recording-level normalizations (e.g. `recording_mean_norm`, the paper's `norm = 1`) are steps, never extractor params; its `fit_scope: whole_recording` is allowed only with `role: reference`.
- **Cache.** Only the runner reads/writes `FeatureCache`; extractors never touch it. Key covers dataset, split hash, selection, preprocessing chain plus fitted-state hash, extractor name/params/version/vendor hash/weights revision, and (for per-window extractors) window length and stride. Writes are atomic; deleting `.cache/features` never changes a result. The runner may also cache per-block intermediates of `uses_context` extractors, keyed per recording block without window length, stride or room set (Story 2.11).
- **Heads and evaluation.** One shared head family: `logreg`, `svm`, `knn`, `random_forest`, `mlp`. A head gets `fit(train, val)`, tunes on val, refits on train ∪ val, and predicts test once; feature scaling is fitted inside the head. A single `Evaluator` computes accuracy, macro-F1, per-class scores (keyed by class name) and the confusion matrix for val and test.
- **Seeds.** `config.seeds` drives training randomness; a run executes once per seed under `runs/<run_id>/seeds/<seed>/` and records mean ± std. Adding a seed to a finished run executes only the missing seed.
- **Feature selection** (`feature_selection`, distinct from room `selection`) sits between features/pooler and head: `null` (default), `manual` (column names or families), or `mrmr`. It is fitted on train, with k chosen on val together with head hyperparameters; test columns are filtered with the fitted choice. Selection only picks columns — features are never recomputed per subset — and each run records selected `feature_names` per seed.
- **Results store and report.** Each run writes `runs/<run_id>/{config.yaml, status.json, metrics.json, seeds/<seed>/predictions.parquet, provenance.json, run.log}`; paths and schemas live once in `core/results_schema.py` and are shared by `ResultsStore` and `results_api`. `status.json` moves pending → running → done | failed. `report` produces the comparison table (methods as rows, room settings as columns), choosing each cell by best validation macro-F1 and reporting its test score; it refuses to mix split hashes, excludes dirty runs by default, supports `--all-variants`, shows feature-selection stability, and prints each row's regeneration command. It reads published numbers and shows `role: reference` rows separately.
- **Sweeps** run one at a time in one process, skip `done` runs, rerun clean-tree `done` runs recorded dirty, continue after a failure, and support `--force`.
- **Selection vs split.** Room settings r1/r2/mixed are a post-split `selection: {rooms: [...]}` filter over one `SplitPlan`; each is a separate run. Fitted preprocessing and heads use only selected recordings.

## Cross-Story Dependencies

- Depends on Epic 1: the committed `time_block` split and `split_hash`, the `tor_vergata` dataset, the core contracts, errors, hashing, registry and results-schema foundation.
- Within the epic, Story 2.1 establishes the end-to-end path (config → runner → store) that 2.2–2.8 extend; 2.4 (cache) underpins 2.6 (selection reuses cached features) and 2.7 (sweeps); 2.8 depends on the run/store layout fixed by 2.1.
- Downstream epics build on this harness: Epic 3 (paper reproduction) reuses the splitter, `paper_beamforming` and `report`; Epic 4 (LWM) reuses poolers, heads, cache and report; Epic 5 adds per-window extractors; the full-comparison sweep in Epic 5 depends on the report and sweep machinery here.
