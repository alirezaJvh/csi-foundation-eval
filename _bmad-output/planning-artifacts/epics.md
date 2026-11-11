---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/specs/spec-lwm-wifi-har/SPEC.md
  - _bmad-output/specs/spec-lwm-wifi-har/datasets.md
  - _bmad-output/specs/spec-lwm-wifi-har/preprocessing.md
  - _bmad-output/specs/spec-lwm-wifi-har/lwm-models.md
  - _bmad-output/specs/spec-lwm-wifi-har/candidate-models.md
  - _bmad-output/planning-artifacts/architecture/architecture-master-2026-10-05/ARCHITECTURE-SPINE.md
---

# master - Epic Breakdown

## Overview

This document breaks the work into epics and stories. Its inputs are the spec `SPEC-lwm-wifi-har` (which serves as the requirements document) and the architecture spine (AD-1 to AD-18). The project has no UI, so there is no UX input.

## Requirements Inventory

### Functional Requirements

FR1: Load all 19 recordings of the own dataset (`tor_vergata`) with activity class, room and recording id. [CAP-1]
FR2: A `describe` command reports, per recording, the packet count, and reports counts per class, per room and per split. [CAP-1, rate-confound guard]
FR3: Produce a time-block train/val/test split with a fixed gap between blocks (`gap_packets`, initially 64), saved as a committed split file and reproducible from it. [CAP-1, AD-6]
FR4: Apply preprocessing as an ordered chain of switchable steps: antenna selection (2 = antennas [0,1], or all 4), global rescale, per-packet power normalization, phase sanitization and amplitude-only. Any statistics are fitted on train only. [preprocessing.md, AD-9]
FR5: Cut windows of configurable length (in packets; standard sweep 1, 10, 30, 40, 50) and stride inside split blocks, identical for every method. [AD-7]
FR6: Train the classifiers `logreg`, `svm`, `knn`, `random_forest` and `mlp`, choosing hyperparameters on val. Evaluate accuracy, macro-F1, per-class scores and the confusion matrix on val and on test, with mean ± std over seeds. Also record the val macro-F1 of the chosen candidate before it is refitted on train ∪ val; this is the score used for selection. [CAP-3, AD-11, AD-12]
FR7: Run experiments for room 1, room 2 and both rooms mixed as room selections over one shared split. [CAP-3, AD-17]
FR8: Compute classical features in families with named columns: `paper_beamforming` (the paper's Doppler integral, CSI distance and flatness features, with the `norm` and `fix_distance_loop` params) and `amplitude_stats` (mean power, standard deviation, flatness, …). [CAP-4, AD-8]
FR9: Feature selection: none, a manual subset (by column or family), or MRMR. It is fitted on train and tuned on val. Selected features are recorded per run and seed. [CAP-4, AD-18]
FR10: Define experiments in YAML or with a Python `ExperimentBuilder`; both produce the same `ExperimentConfig`. One `run` command executes a config, and rerunning a saved config reproduces the same run_id. [AD-5]
FR11: A `sweep` command expands a sweep file into the Cartesian product of its options. It runs them one at a time, can resume, skips finished runs, and keeps going after a failure. [CAP-4, AD-14]
FR12: A `report` command produces the comparison table:
- Rows are methods and columns are room settings.
- Each cell is the variant with the best tuning-stage val macro-F1 (from a model not trained on val), reported with its test score.
- An all-variants view, the regeneration command for each row, and feature-selection stability are included.
- The published numbers are shown next to reference rows.

[CAP-5, AD-13]
FR13: Recompute the paper's method from `saved_csi` (all `paper_beamforming` features + random forest). Run it with a window-level random split as the reference row, and with the time-block split as a comparison row. Check it against the `saved_dataset` fixture. [CAP-5, AD-16]
FR14: Run frozen LWM v1.1 on every packet, producing CLS or channel-mean embeddings for the 2- and 4-antenna variants. Embeddings are pooled over windows (`mean`, `mean_std`) and pass the NaN/constant sanity check. [CAP-2, CAP-3, AD-8]
FR15: Run frozen WavesFM, Mantis-8M and one ImageNet vision model as per-window feature extractors, with the same windows, classifiers and metrics. [CAP-7]
FR16: Notebooks read results through a read-only results API to make charts. [AD-13]
FR17: Add the verified activity-recognition datasets (Padova 80 MHz first, then CSI-Bench) by writing a new loader only, with no change to the evaluation code. [CAP-6]
FR18: A `room_holdout` splitter trains and validates on one room and tests on the other room's full recordings (r1→r2, r2→r1). Each direction is its own committed plan. [CAP-8, AD-6]
FR19: Every test score carries a 95% confidence interval from a class-stratified bootstrap over test recordings. [CAP-8, AD-11]
FR20: Run directories have readable names, `runs/<dataset>/<splitter>/<slug>--<run_id>/`, and `report` prints one section per (dataset, splitter). [AD-13]

### NonFunctional Requirements

NFR1: No leakage:
- Every method gets identical splits, windows and metrics.
- No window crosses a split block.
- Extractors read only their window's packets.
- Fitted statistics, feature selection and hyperparameters use train/val only, and test is scored once.

[AD-6, AD-7, AD-9, AD-11, AD-18]
NFR2: Reproducibility and traceability:
- `run_id` is a hash of the resolved config, adapter versions and vendored code.
- Every result stores its config and provenance (git commit, dirty flag, package versions, split hash, weights revisions).
- Every number can be regenerated with one command.

[AD-5, AD-13]
NFR3: Extensibility. Adding or removing a dataset, preprocessing step, model, feature family, selector or classifier means adding or deleting one adapter module; nothing else changes. Dependency rules are enforced by import-linter. [AD-1, AD-2, AD-3]
NFR4: Pretrained models stay frozen: run in eval mode under inference mode, and a checksum test proves the weights are unchanged. No fine-tuning in this phase. [AD-8]
NFR5: Runs on the local CPU (Ryzen 7, 16 threads, 11 GB RAM). Extractors work in batches, and the runner owns device and thread settings. [AD-12, conventions]
NFR6: Code quality: ruff and ty pass (`_vendor/` excluded), and every port has a contract test that each of its adapters runs. [conventions]
NFR7: Recordings are used at their saved rate, with no resampling, and the domain gap is reported (LWM input amplitude statistics go into provenance). [SPEC constraints, AD-13]
NFR8: Public datasets are used only after the supervisor verifies them. [SPEC constraints]

### Additional Requirements

- **No starter template.** This is a greenfield `uv` project: Python 3.12, CPU-only torch/torchvision from the PyTorch CPU index via `tool.uv.sources`, and a `pandas==3.0.6` override (mantis-tsfm caps pandas but never imports it). It also sets up ruff, ty, import-linter and pytest. All versions are pinned per the spine's Stack table. This becomes Epic 1, Story 1.
- Source layout per the Structural Seed: `src/csi_bench/{core, adapters/*, config, runner, results_api}`, `experiments/`, `splits/`, `notebooks/`, `tests/`. The data stays in `data/csi_proc_python/`, and `.cache/` and `runs/` are git-ignored.
- Core contracts are frozen dataclasses with the fixed shapes, ids and label coding of AD-4: `Recording` with `antenna_ids` and `RecordingMeta`, `window_id = recording_idstart:end`, and `FeatureMatrix` with `feature_names`.
- Adapters register themselves via `register` and are found automatically by `pkgutil`. Each declares a pydantic `Params` model and a `VERSION`; params marked `affects_result=False` are left out of run_id. [AD-3]
- Sweep files use an explicit `sweep:` block of dotted paths. `seeds` cannot be swept. [AD-5]
- The feature cache has a single client (the runner), complete keys and atomic writes. Deleting it never changes a result. [AD-10]
- Results store layout and schemas are defined in `core/results_schema.py`. A sweep reruns finished runs that were dirty once the tree is clean. [AD-13, AD-14]
- Vendored research code:
  - LWM: `lwm_model.py` plus a `patch_maker` excerpt only, with the `module.` prefix stripped, `map_location="cpu"`, and `make_sample` never called.
  - WavesFM: `models_vit.py` and `pos_embed.py` at the pinned commit, with the key remap `vis_patch_embed.0.*` → `vis_patch_embed.*`.
  - Each vendored folder has a `SOURCE.md`, and a strict-load test checks the weights.

  [AD-15]
- Weights are fetched with `huggingface_hub` at pinned full shas into `.cache/weights/`. The ImageNet backbone is chosen and pinned when its story is built.
- The published reference numbers go in `experiments/published_reference.yaml`: room 1 99.41, room 2 99.59, mixed 97.37 (4 features, W2 = 40, random split, PIMRC 2025).
- Build order (SPEC constraint):
  1. Harness foundation
  2. Data
  3. Classical harness
  4. Paper reproduction
  5. Pretrained models (LWM may start right after Data)

### UX Design Requirements

Not applicable: the project has no user interface. It is driven from the command line, and results are read in notebooks.

### FR Coverage Map

FR1: Epic 1 - Load the 19 own recordings with labels
FR2: Epic 1 - `describe` counts per recording, class, room and split
FR3: Epic 1 - Committed time-block train/val/test split with a gap
FR4: Epic 2 - Preprocessing chain, antenna selection and global rescale; Epic 5 - LWM-specific variants (per-packet power, phase sanitization, amplitude-only)
FR5: Epic 2 - Windowing inside split blocks
FR6: Epic 2 - Shared classifiers, metrics and seeds
FR7: Epic 2 - Room selections r1/r2/mixed over one split
FR8: Epic 2 - Classical feature families with named columns
FR9: Epic 2 - Feature selection (manual, MRMR)
FR10: Epic 2 - ExperimentConfig from YAML or builder; `run` command
FR11: Epic 2 - Resumable, failure-isolated `sweep`
FR12: Epic 2 - `report` comparison table
FR13: Epic 3 - Paper method under random and time-block splits
FR14: Epic 5 - Frozen LWM per-packet embeddings and pooling
FR15: Epic 7 - WavesFM, Mantis-8M, ImageNet extractors
FR16: Epic 2 - Read-only results API for notebooks
FR17: Epic 6 - Public dataset loaders (Padova 80 MHz, CSI-Bench)
FR18: Epic 4 - `room_holdout` cross-room splitter
FR19: Epic 4 - Recording-bootstrap confidence intervals
FR20: Epic 4 - Readable run layout and a report section per (dataset, splitter)

## Epic List

### Epic 1: Reproducible dataset access
The researcher can load all 19 own recordings, generate the committed time-block split, and inspect counts per recording, class, room and split with `describe`. Story 1 sets up the project (uv, pinned CPU stack, quality tooling, core contracts, registry).
**FRs covered:** FR1, FR2, FR3

### Epic 2: Classical ML harness
The researcher and Paolo can write a sweep file (window lengths × feature sets or selection × classifiers × room settings), run one command, and get a leakage-free results table with feature-selection stability; notebooks chart the results.
**FRs covered:** FR4 (chain, antenna selection, global rescale), FR5, FR6, FR7, FR8, FR9, FR10, FR11, FR12, FR16

### Epic 3: Paper reproduction and leakage measurement
The researcher can reproduce the paper's published accuracy with its random split and show what the same method scores under the time-block split.
**FRs covered:** FR13

### Epic 4: Generalisation protocols
The researcher can score any method across rooms (train on one room, test on the other) and see a confidence interval on every test score, with all results kept in one readable results store.
**FRs covered:** FR18, FR19, FR20

### Epic 5: LWM evaluation
The researcher can embed every packet with frozen LWM (2/4 antennas, preprocessing, embedding type, pooling, window variants) and get LWM rows in the same table. The embedding story needs only Epic 1; scoring needs Epic 2.
**FRs covered:** FR14, FR4 (LWM-specific variants)

### Epic 6: Public datasets (activity recognition)
The researcher can rerun the study on the supervisor-verified activity-recognition datasets (Padova 80 MHz, then CSI-Bench) by adding only their loaders.
**FRs covered:** FR17

### Epic 7: Other pretrained models
The researcher can add WavesFM, Mantis-8M and an ImageNet model as rows in the same table, each as one new extractor.
**FRs covered:** FR15

**Execution order** (sprint-change-proposal-2026-10-11): Epic 4 → Epic 5 → Epic 6 → Epic 7, in number order. Epic 4 comes first because every later score uses it. Epic 6 comes before Epic 7 so the new models are scored on every verified dataset. Epics 4–7 were renumbered on 2026-10-11 so numbers follow execution order (old 7 → 4, old 4 → 5, old 5 → 7; Epic 6 kept its number).

**Remaining order** (sprint-change-proposal-2026-11-02): Story 4.5 → Story 6.2 → Story 6.3 → Story 6.4 → Epic 7. Story 4.5 goes first so no further sweep is selected on the in-sample val score. The Epic 6 stories were renumbered on 2026-11-05 so numbers follow execution order (old 6.3 → 6.2, old 6.4 → 6.3, old 6.2 → 6.4); that proposal still uses the old numbers.

**Remaining order** (sprint-change-proposal-2026-11-10): Story 6.5 → Story 6.6 → Story 6.7 → Epic 7. LWM is scored on Padova and CSI-Bench before the other pretrained models, so Epic 7 adds rows to tables that already hold LWM on every dataset. Story 6.5 (device option, cache race fix) comes first because 6.7's embedding cost depends on it; 6.6 commits the long-window plans 6.7 uses.

## Epic 1: Reproducible dataset access

The researcher can load all 19 own recordings, generate the committed time-block split, and inspect counts per recording, class, room and split with `describe`.

### Story 1.1: Project setup with the pinned CPU stack and an adapter registry

As a researcher,
I want a uv project with the pinned CPU-only stack, quality tooling, and a registry where components register themselves,
So that every later component plugs in by adding one module, and the architecture rules are enforced from day one.

**Acceptance Criteria:**

**Given** a fresh clone
**When** I run `uv sync`
**Then** Python 3.12 and every Stack-table pin install, with torch/torchvision from the PyTorch CPU index and no `nvidia-*` packages
**And** `pandas==3.0.6` resolves through `override-dependencies`, and `uv.lock` is committed

**Given** the project
**When** I run `ruff check`, `ruff format --check`, `ty check` and `pytest`
**Then** all pass, and `_vendor/` folders are excluded from ruff and ty

**Given** the import-linter contracts for AD-2 (`core` imports nothing internal; adapters import only `core`; adapters never import each other; only `runner` imports adapters)
**When** a test module breaks one of them
**Then** `lint-imports` fails

**Given** an adapter module decorated with `register("<stage>", "<name>")` that declares a pydantic `Params` model and a `VERSION`
**When** the runner's discovery imports `csi_bench.adapters` via `pkgutil.walk_packages`
**Then** the adapter can be looked up by `(stage, name)` without being named anywhere else
**And** registering the same `(stage, name)` twice raises `ConfigError`

**And** `core/` provides `errors.py` (`ConfigError`, `DataError`, `ExtractorError`) and `hashing.py` (SHA-256 of canonical JSON, shown as 12 hex chars), each with unit tests
**And** `.cache/` and `runs/` are git-ignored, and `data/csi_proc_python/` is left unchanged

### Story 1.2: Load the own recordings as labelled `Recording`s

As a researcher,
I want the `tor_vergata` dataset source to load every saved recording with its room, activity and metadata,
So that every later stage works from one consistent, labelled view of the data.

**Acceptance Criteria:**

**Given** `data/csi_proc_python/saved_csi/`
**When** the `tor_vergata` source loads it
**Then** exactly 19 `Recording`s are returned, each a frozen dataclass with:
- `csi`: complex64, shape `(4, n_packets, 234)`
- `antenna_ids = (0, 1, 2, 3)`
- `recording_id = "tor_vergata/<file stem>"`
- `room ∈ {r1, r2}`
- `label` as an index into the declared class list `[empty, sitting, standing, walking]`

**Given** the file-name variants in the folder (`r1_sitting_*`, `r2_sit_*`, `r1_walking_*`, `r2_walk_*`, `r1_empty` with no index)
**When** they are parsed
**Then** they map to the correct class and room
**And** an unrecognised file name raises `DataError` naming the file

**Given** each loaded recording
**Then** `meta` holds `n_packets`, a `source_sha256` of the file, and `downsample_factor`, taken from the `csi_saver.py` history (r2_standing_2 = 5, r2_standing_1 = 3, r1 sitting = 2, otherwise `None`)
**And** `sample_rate_hz` is `None`

**And** a `DatasetSource` port contract test, which every future dataset adapter will run, checks shapes, dtypes, id uniqueness and label validity

### Story 1.3: Generate the committed time-block train/val/test split

As a researcher,
I want a time-block split that places contiguous train, val and test blocks in every recording, separated by a gap,
So that no window can ever span two parts, and every method uses exactly the same split.

**Acceptance Criteria:**

**Given** the 19 recordings and `time_block` params (`gap_packets = 64`, block fractions in the order train | val | test, default 0.6 / 0.2 / 0.2)
**When** the split is generated
**Then** each recording gets one contiguous block for each part, in packet indices (0-based, end-exclusive)
**And** adjacent blocks are separated by at least `gap_packets`
**And** the blocks never overlap

**Given** the same inputs and params
**When** the split is generated twice
**Then** the `SplitPlan` content and `split_hash` are identical
**And** the plan is written to `splits/tor_vergata/<split_hash>.json` together with each input's `source_sha256`

**Given** a configuration that leaves some class with no val or no test packets
**When** the split is generated
**Then** `DataError` names the class and the missing part

**Given** a recording file whose content changes after the split was committed
**When** that split is loaded
**Then** loading fails with `DataError` because the `source_sha256` doesn't match

### Story 1.4: Inspect the dataset with `csi-bench describe`

As a researcher,
I want one command that summarises the dataset and its split,
So that I can check class balance and per-recording packet counts, which guard against sample rate standing in for the activity, before running any experiment.

**Acceptance Criteria:**

**Given** the installed project
**When** I run `csi-bench describe tor_vergata`
**Then** it prints one row per recording (`recording_id`, room, class, `n_packets`, `downsample_factor`, and train/val/test packet counts for the current split)
**And** it prints totals per class, per room and per split

**And** it writes the same table to `runs/_dataset/tor_vergata.parquet`, using a schema defined in `core/results_schema.py`

**Given** an unknown dataset name
**When** I run `describe`
**Then** the command exits non-zero, with a message that lists the registered dataset names

## Epic 2: Classical ML harness

The researcher and Paolo can write a sweep file (window lengths × feature sets or selection × classifiers × room settings), run one command, and get a leakage-free results table with feature-selection stability; notebooks chart the results.

### Story 2.1: Run one classical experiment end-to-end

As a researcher,
I want to run one experiment from a YAML config and get its metrics saved,
So that the whole path (windows → features → classifier → score → stored result) works before anything is widened.

**Acceptance Criteria:**

**Given** a YAML `ExperimentConfig` with:
- dataset `tor_vergata`
- the committed split
- `selection.rooms: [r1]`
- `window_len: 40`, `stride: 1`
- extractor `amplitude_stats` (at least `mean_power`, `std`, `flatness`)
- head `logreg`
- `role: comparison`

**When** I run `csi-bench run <config.yaml>`
**Then** the config is validated (pydantic) before any stage runs, and an invalid field fails within seconds with `ConfigError` naming that field

**Given** the split and the selection
**When** the Windower runs
**Then**:
- every window lies inside one split block
- `window_id = "<recording_id><start>:<end>"`
- windows are ordered by `(recording_id, start)`
- only recordings from the selected rooms appear
- a `window_len` larger than `gap_packets` raises `ConfigError`

**Given** the windows
**When** `amplitude_stats` runs
**Then** it receives only each window's packet slice and returns a `FeatureMatrix` with `feature_names` like `amplitude_stats.mean_power`
**And** extracting one window alone gives the same values as extracting it in a batch (contract test)

**Given** the features
**When** the head is fitted
**Then** it receives `fit(train, val)`, tunes its regularisation on val, refits on train ∪ val, and predicts test once
**And** the Evaluator reports accuracy, macro-F1, per-class scores (keyed by class name) and the confusion matrix for val and test

**And** the run writes `runs/<run_id>/{config.yaml, status.json, metrics.json, seeds/<seed>/predictions.parquet, provenance.json, run.log}` in the schemas of `core/results_schema.py`
**And** `run_id` hashes the resolved config plus adapter `VERSION`s and excludes `affects_result=False` params
**And** running `csi-bench run runs/<run_id>/config.yaml` reproduces the same `run_id` (test)
**And** provenance records the git commit, the dirty flag (plus the diff when dirty), package versions and `split_hash`

### Story 2.2: Preprocessing chain with antenna selection and global rescale

As a researcher,
I want to switch preprocessing steps on and off per experiment, with any statistics learned from training data only,
So that I can compare the 2- and 4-antenna variants and rescaling without leaking test information.

**Acceptance Criteria:**

**Given** a config with `preprocessing: [{antennas: [0, 1]}, {global_rescale: {}}]`
**When** the run executes
**Then** the steps apply in the listed order, and `Recording.antenna_ids` becomes `(0, 1)`
**And** `global_rescale` fits its mean amplitude on the selected **train** blocks only and records the fitted value in provenance

**Given** two configs that list the same steps in a different order
**Then** they get different `run_id`s

**Given** named presets in `config/presets.py`
**When** a config references a preset by name
**Then** it expands to the same step list as writing the steps out explicitly

**Given** the step `recording_mean_norm` (the paper's `norm = 1`)
**When** it is fitted
**Then** each antenna's mean amplitude is computed from that recording's train block only and recorded in provenance
**And** `fit_scope: whole_recording` is accepted only with `role: reference`; with `role: comparison` it raises `ConfigError`

**And** a `PreprocessStep` port contract test checks that `transform` is pure and that `fit` sees only train blocks

### Story 2.3: Full classifier family with seeds

As a researcher,
I want SVM, KNN, Random Forest, logistic regression and a small MLP, all tuned on validation and repeated over seeds,
So that I can compare classifiers fairly and report mean ± std.

**Acceptance Criteria:**

**Given** `head: svm | knn | random_forest | logreg | mlp`, each with a pydantic hyperparameter grid in its params
**When** the run executes
**Then** each head scales features inside itself (fitted on train), selects hyperparameters on val only, refits on train ∪ val, and predicts test once

**Given** `seeds: [0, 1, 2]`
**When** the run executes
**Then** each seed writes `seeds/<seed>/predictions.parquet`, and `metrics.json` holds per-seed scores plus mean ± std
**And** adding a seed to a finished run executes only the missing seed

**And** the runner sets `device="cpu"`, the torch thread count and the determinism flags, and records them in provenance. Adapters never set them.
**And** a `Head` port contract test runs every head on a small synthetic dataset and checks that no head sees test labels during fitting

### Story 2.4: Runner-owned feature cache

As a researcher,
I want computed features cached and reused,
So that sweeping classifiers or feature subsets over the same windows doesn't recompute the features.

**Acceptance Criteria:**

**Given** two runs that differ only in head or head params
**When** the second run executes
**Then** features load from `.cache/features` without calling the extractor (verified with a spy)

**Given** a change to any key part (dataset, `split_hash`, selection, preprocessing chain + fitted state, extractor params, `VERSION`, window length/stride for per-window extractors)
**Then** the cache misses

**And** cache writes are atomic (temp file + rename), and only the runner calls `FeatureCache`. An import contract forbids adapters from importing it.
**And** deleting `.cache/features` and rerunning gives identical metrics (test)

### Story 2.5: Paper beamforming feature family

As a researcher,
I want the paper's beamforming features available as a feature family,
So that classical experiments can use them alone or together with the amplitude statistics.

**Acceptance Criteria:**

**Given** `extractor: paper_beamforming` (antennas 0 and 1, 40 phase weights, inner windows W = 8, 16 and 1 as in `processing.py`, param `fix_distance_loop: false`; normalization comes from the `recording_mean_norm` preprocessing step, not from the extractor)
**When** it runs on a window
**Then** it returns named columns for the paper's features (Doppler integral, CSI distance, flatness and their window statistics), computed only from that window's packets
**And** its `min_window_len` covers the inner-window lookback, and shorter windows are rejected at config validation

**Given** a test-only oracle `tests/oracles/paper_processing.py` that copies `processing.py`'s per-window computation verbatim, with a source note (`processing.py` is a script with no importable functions and stays unchanged)
**When** the port and the oracle run on the same packets
**Then** the feature values match within numerical tolerance, both with the original distance loop (`fix_distance_loop: false`) and with the fixed one

**Given** a config listing several families (`extractor: [amplitude_stats, paper_beamforming]`)
**Then** the runner concatenates their columns, and the names stay unique

### Story 2.6: Feature selection (manual subsets and MRMR)

As a researcher,
I want to choose features by name or let MRMR pick the best k,
So that Paolo and I can find which features matter and how stable that choice is.

**Acceptance Criteria:**

**Given** `feature_selection: {manual: [amplitude_stats.mean_power, paper_beamforming]}` (a column name or a whole family)
**Then** the head receives exactly those columns

**Given** `feature_selection: {mrmr: {k: [2, 3, 5]}}`, implemented with `mrmr-selection` 0.2.8 (`mrmr_classif`, the same function `classification.py` uses)
**When** the run executes
**Then** MRMR is fitted on train, k is chosen on val together with the head's hyperparameters, and test columns are filtered with the fitted choice
**And** the selected `feature_names` are recorded per seed in `metrics.json`

**And** with `feature_selection: null` all columns are used; this is the default
**And** a `FeatureSelector` contract test checks that a selector never receives test rows

### Story 2.7: Sweeps and the Python experiment builder

As a researcher,
I want to list options in a sweep file and run every combination in one command that survives crashes,
So that I can queue a large comparison and come back to it later.

**Acceptance Criteria:**

**Given** a YAML file with a base config and a `sweep:` block of dotted paths (e.g. `window.window_len: [10, 30, 40, 50]`, `head.name: [svm, knn, random_forest]`, `selection.rooms: [[r1], [r2], [r1, r2]]`)
**When** I run `csi-bench sweep <file>`
**Then** it expands to the Cartesian product, validates every config before running any, and runs them one at a time
**And** sweeping `seeds` is rejected with `ConfigError`

**Given** an interrupted sweep
**When** it is relaunched
**Then** `done` runs are skipped, and a failed run is recorded as `failed` with its traceback while the sweep continues
**And** `done` runs recorded with `dirty: true` are rerun when the tree is now clean, and `--force` reruns everything

**Given** `ExperimentBuilder().dataset(...).features(...).head(...).build()`
**Then** it returns an `ExperimentConfig` equal to the one loaded from the equivalent YAML (test), and it never builds pipeline objects

**And** an example sweep, `experiments/classical_baseline.yaml`, covers Paolo's request: window lengths × feature sets/selection × SVM/KNN/RF × rooms

### Story 2.8: Comparison report and notebook results API

As a researcher,
I want a report command and a read-only results API,
So that my supervisor gets the comparison table and I can chart results in Jupyter.

**Acceptance Criteria:**

**Given** finished runs
**When** I run `csi-bench report`
**Then**:
- `runs/index.parquet` is refreshed
- a table is produced with methods as rows and room settings (r1, r2, mixed) as columns
- each cell shows test macro-F1 and accuracy (mean ± std) for the variant with the best **val** macro-F1, plus the command that regenerates it

**And** `--all-variants` lists every run's variant
**And** a feature-selection stability view shows how often each feature was selected, per room and seed
**And** the report refuses to mix different `split_hash`es in one table, and excludes dirty runs by default
**And** `role: reference` rows are shown separately, next to the values in `experiments/published_reference.yaml` (r1 99.41, r2 99.59, mixed 97.37)

**Given** a notebook
**When** it imports `csi_bench.results_api`
**Then** it can load the index, metrics and predictions using the shared schema, but cannot write
**And** a test fails if any notebook imports anything other than `results_api` from `csi_bench`
**And** `notebooks/01_classical_overview.ipynb` shows an example chart (macro-F1 vs window length per classifier)

### Story 2.9: Faster feature extraction and head training

As a researcher,
I want `paper_beamforming` extraction and the heads to use several cores within a RAM budget,
So that the 216-run classical baseline sweep finishes in hours, not days.

**Acceptance Criteria:**

**Given** `paper_beamforming` and a set of windows
**When** features are extracted with several workers
**Then** the feature matrix is identical to the single-process result (same values, same row order)
**And** the paper-parity test against `saved_dataset/*.npy` still passes
**And** the worker count is configurable, defaults to a value that keeps the run within about 4 GB on the development machine, and `1` restores the old behaviour

**Given** the `random_forest` and `knn` heads
**When** they are fitted
**Then** they use more than one core, and the metrics for a fixed seed are unchanged

**Given** `experiments/classical_baseline.yaml`
**When** the first `paper_beamforming` run finishes
**Then** the measured extraction time and peak memory before and after are recorded in the story's spec
**And** a run left in `running` state by an interrupted sweep is rerun on resume

**Out of scope:** running several sweep processes at once, which needs a run-directory lock. Reusing overlapping inner windows moved to Story 2.10 (the measured extraction time showed multi-core alone does not help).

### Story 2.10: Compute paper_beamforming inner-window metrics once per packet

As a researcher,
I want `paper_beamforming` to compute each inner-window metric once per packet instead of once per window,
So that the classical baseline sweep's extraction time falls from ~16 min to a few minutes per run.

**Acceptance Criteria:**

**Given** the stage timings of one `paper_beamforming` run on `experiments/classical_baseline.yaml` (load, preprocess, extract, head fit)
**Then** they are logged and recorded in the story's spec before the change, so the gain is measured and the next bottleneck is known

**Given** a split block and window offsets
**When** `paper_beamforming` extracts through the block-aware path
**Then** the feature matrix is exactly equal (same values, same row order) to the one from the slice-based `extract`, for `fix_distance_loop` true and false, on synthetic and real windows
**And** the paper-parity test (`tests/test_paper_beamforming.py` vs `tests/oracles/paper_processing.py`) still passes
**And** if bit-equality proves impossible, the tolerance decision goes to the human and `VERSION` is bumped

**Given** an extractor with `uses_context = True`
**Then** the runner calls it block by block, never across a split gap, so every position a window uses lies inside that window
**And** a contract test asserts block-aware output equals slice-based output for every registered extractor that offers both
**And** the feature-cache key and cached matrix are unchanged, and `VERSION` stays the same when values are identical

**Given** the same run as before
**Then** the measured extraction time and peak RSS before and after are recorded in the story's spec, and peak RSS stays within about 4 GB

**Out of scope:** caching per-position arrays across window lengths and room sets (moved to Story 2.11), and running several sweep processes at once.

### Story 2.11: Share paper_beamforming per-position metrics across the sweep

As a researcher,
I want the per-position inner-window metrics computed once per recording block and reused by every window length and room set,
So that the baseline sweep's `paper_beamforming` extraction falls from about 1.6 h to minutes.

**Acceptance Criteria:**

**Given** two runs of `paper_beamforming` that differ only in `window_len`, `stride` or room set
**When** the second run extracts
**Then** it reads the per-block arrays from the runner-owned cache and recomputes none that exist
**And** r1+r2 is built from the r1 and r2 entries

**Given** any window length, stride or room set, and `fix_distance_loop` true and false
**When** features are extracted through the shared arrays
**Then** the matrix equals the Story 2.10 block-aware matrix exactly, including against a matrix cached by the old path
**And** the paper-parity test still passes

**Given** a preprocessing chain with fitted state
**Then** the per-block entries are keyed by that state, so values are never shared across different state
**And** the feature-matrix cache key, `VERSION` and run ids are unchanged, and deleting `.cache` never changes a result

**Given** the `paper_beamforming` runs of `experiments/classical_baseline.yaml`
**Then** total extraction time and peak RSS before and after are recorded in the story's spec, and peak RSS stays within about 4 GB

**Out of scope:** running several sweep processes at once, and rewriting the extractor maths.

## Epic 3: Paper reproduction and leakage measurement

The researcher can reproduce the paper's published accuracy with its random split and show what the same method scores under the time-block split.

### Story 3.1: Window-level random splitter for reference runs

As a researcher,
I want a splitter that assigns individual windows to train/val/test at random, as the paper did,
So that I can reproduce the published setup.

**Acceptance Criteria:**

**Given** `splitter: random_window` with `seed` and fractions in its params
**When** a run uses it with `role: reference`
**Then** windows are assigned at random, the plan records `unit: window`, and its `split_hash` differs from every time-block split
**And** the same seed gives the same assignment

**Given** `random_window` with `role: comparison`
**When** the config is validated
**Then** validation fails with `ConfigError`

### Story 3.2: Reproduce the paper and measure leakage

As a researcher,
I want the paper's method run under both splits, next to the published numbers,
So that the thesis can show how much the random split inflates accuracy.

**Acceptance Criteria:**

**Given** `experiments/paper_reproduction.yaml`, which uses:
- `paper_beamforming` (antennas 0 and 1, the paper's W2 = 40 setting) with the paper's normalization: `recording_mean_norm` with `fit_scope: whole_recording` on the reference rows, and the train-fitted default on the comparison rows
- a `random_forest` head
- room selections r1, r2 and mixed, with mixed rooms using the paper's 4 features via MRMR

**When** the sweep runs
**Then** reference rows (`random_window`) and comparison rows (`time_block`) are produced

**Given** the `saved_dataset/*.npy` fixture
**When** the port computes features with the paper's original windowing
**Then** the values match the saved rows for r1, r2 and r1r2 at W2 ∈ {10, 30, 40, 50} within tolerance
**And** any remaining mismatch is documented in a test note

**And** `report` shows the reference rows next to the published 99.41 / 99.59 / 97.37, plus the difference between the random and time-block results per room

## Epic 4: Generalisation protocols

The researcher can score any method across rooms and see a confidence interval on every test score, with all results kept in one readable results store. Runs first, because every later score uses it (sprint-change-proposal-2026-10-11).

### Story 4.1: Readable run layout and a report section per protocol

As a researcher,
I want run directories named after what they contain and a report grouped by dataset and split protocol,
So that every result, old and new, stays in one store and I can find runs by eye.

**Acceptance Criteria:**

**Given** any run
**Then** its directory is `runs/<dataset>/<splitter>/<slug>--<run_id>/`, with `slug = <extractors>-<selector>-<head>-<rooms>-w<window_len>` sanitised to `[a-z0-9_+-]`
**And** lookup by `run_id`, deduplication and resuming still work, and existing `run_id`s don't change
**And** the layout is defined only in `core/results_schema.py`

**Given** the existing `runs/<run_id>/` directories
**When** the one-off migration runs
**Then** every run is moved to the new layout, nothing is deleted, the run count before equals the count after, and the index is rebuilt
**And** running the migration again changes nothing

**Given** runs from several datasets and splitters
**When** `report` runs
**Then** it prints one section per (dataset, splitter), and refuses only a mix of split hashes within one column
**And** the leakage block pairs the `random_window` and `time_block` sections of the same dataset
**And** `results_api` and the notebooks keep working

### Story 4.2: `room_holdout` splitter

As a researcher,
I want a splitter that trains and validates on one room and tests on the other,
So that every method gets a strict out-of-distribution score.

**Acceptance Criteria:**

**Given** params `train_rooms`, `test_rooms`, `fractions` and `gap_packets`
**When** the plan is generated
**Then** train-room recordings are split into time blocks for `train` and `val` (same gap rule), and test-room recordings go entirely to `test`
**And** validation never uses the test room

**Given** the plans for r1→r2 and r2→r1
**Then** both are committed under `splits/tor_vergata/`, and their hashes differ from every other plan
**And** a missing class in `val` or `test` raises `DataError`

**Given** a config with `room_holdout`
**When** `selection.rooms` differs from `train_rooms ∪ test_rooms`
**Then** config validation rejects it

**And** the Splitter contract test passes and `describe --split` works
**And** `report` shows a cross-room section with the columns r1→r2 and r2→r1

### Story 4.3: Confidence intervals from a recording bootstrap

As a researcher,
I want a confidence interval on every test score,
So that I can tell real differences between methods from noise.

**Acceptance Criteria:**

**Given** a run's test predictions
**When** the bootstrap runs
**Then** a pure function in `core/metrics` resamples test recordings with replacement, stratified by class, with a fixed seed and `B` as a parameter, and returns 95% intervals for accuracy and macro-F1
**And** the intervals are stored in `metrics.json`

**Given** existing runs
**Then** their intervals are computed from the saved `predictions.parquet`, with no retraining

**And** `report` cells show the score with its interval
**And** a class with a single test recording is flagged

### Story 4.4: Cross-room classical baseline

As a researcher,
I want the classical grid and the paper method scored across rooms,
So that the first out-of-distribution numbers exist before the pretrained models arrive.

**Acceptance Criteria:**

**Given** `experiments/classical_cross_room.yaml` (the classical grid and the paper method under both `room_holdout` plans)
**When** `sweep` and then `report` run
**Then** the report shows the cross-room section next to the within-room one, with confidence intervals

### Story 4.5: Honest validation score for selection

As a researcher,
I want variants selected on a validation score from a model that did not train on val,
So that every comparison table picks its best variant fairly.

**Acceptance Criteria:**

**Given** a finished run
**When** `metrics.json` is written
**Then** it records the tuning-stage val macro-F1 (`head.val_score` of the chosen candidate) per seed, with its mean and std over seeds
**And** the existing `val` metrics remain, documented as in-sample diagnostics of the refitted model

**Given** the run index and `report`
**When** the best variant per method and room setting is chosen, including for the leakage rows
**Then** selection uses the tuning-stage score
**And** a done run without that score is excluded, with a note giving how many were excluded and the command that fixes them

**Given** a synthetic case where a memorising head wins on the in-sample val but loses on the tuning-stage score
**When** `report` selects
**Then** the head with the better tuning-stage score wins, and a test fails if selection reverts to the in-sample score

**Given** the existing store
**When** the 5,084 reported runs are rerun
**Then** a copy of every old `metrics.json` is kept first, outside `runs/`, and no run or result is lost
**And** run ids are unchanged, and each run's test metrics match the old ones, with any mismatch listed rather than ignored
**And** a before/after table per `(dataset, splitter)` section shows every cell whose selected variant or score changed
**And** `_bmad-output/reports/lwm-vs-classical.html` is regenerated

## Epic 5: LWM evaluation

The researcher can embed every packet with frozen LWM (2/4 antennas, preprocessing, embedding type, pooling, window variants) and get LWM rows in the same table. Story 5.1 needs only Epic 1; scoring needs Epic 2.

### Story 5.1: Embed every packet with frozen LWM v1.1

As a researcher,
I want frozen LWM embeddings for every packet,
So that I can evaluate LWM on our data.

**Acceptance Criteria:**

**Given** the `lwm` adapter
**Then** `_vendor/` contains only `lwm_model.py` and a `patch_maker` excerpt, with no DeepMIMO imports
**And** `SOURCE.md` records the repo URL, sha `1b68512dbe97de56841301206eacb780fcd76011`, local edits and constructor arguments
**And** weights are fetched at that sha into `.cache/weights/`

**Given** the checkpoint
**When** it is loaded
**Then** the `module.` prefix is stripped, it loads with `map_location="cpu"` and strict key matching, and the strict-load test passes
**And** `make_sample` is never called

**Given** 4-antenna or 2-antenna input (the 2-antenna input is padded by `patch_maker`)
**When** it runs with `embedding: cls | channel_mean`
**Then** it returns `PacketFeatures` of shape `(n_packets, 128)` for every packet, which pass the finite/non-constant check

**And** the model runs in `eval()` under `torch.inference_mode()`, and a test confirms parameter checksums don't change
**And** the adapter reports input amplitude statistics alongside the embeddings, with a warning if the median magnitude isn't of order 1
**And** embedding the full dataset finishes on CPU, and the time taken is logged (about 2 minutes expected)

### Story 5.2: LWM preprocessing variants

As a researcher,
I want the remaining preprocessing variants,
So that the domain-gap experiments in `preprocessing.md` can run.

**Acceptance Criteria:**

**Given** the steps `per_packet_power`, `phase_sanitize` (subtract each packet's linear phase fit across subcarriers) and `amplitude_only`
**When** the `PreprocessStep` contract test runs
**Then** each step passes

**And** presets cover the spec's 4 variants:
- global rescale
- per-packet power
- global rescale or per-packet power, each followed by phase sanitization
- amplitude-only

**Given** synthetic CSI with an added linear phase slope
**When** `phase_sanitize` is applied
**Then** the slope is removed, to within tolerance

### Story 5.3: Pool and score LWM in the comparison

As a researcher,
I want LWM embeddings pooled over the shared windows and scored with the shared classifiers,
So that LWM gets rows in the same table.

**Acceptance Criteria:**

**Given** the poolers `mean` and `mean_std`
**When** an LWM run uses `window_len ∈ {1, 10, 30, 40, 50}`
**Then** per-packet embeddings come from the cache (one entry per recording) and are pooled per window
**And** LWM is never re-run because the window length changed

**And** `window_len: 1` is accepted for LWM and rejected for per-window extractors
**And** `feature_selection` defaults to null for LWM
**And** the LWM input amplitude statistics from Story 5.1 are written to provenance
**And** `experiments/lwm.yaml` sweeps antennas × preprocessing presets × embedding type × pooler × window length × classifier × rooms, and `report` shows the LWM rows
**And** `experiments/lwm.yaml` also runs under both `room_holdout` plans, and `report` shows the LWM rows in the cross-room section with confidence intervals

## Epic 6: Public datasets (activity recognition)

The researcher can rerun the study on the supervisor-verified activity-recognition datasets (verified 2026-10-08) by adding only their loaders. Padova 80 MHz comes first, as the likely easiest import, followed by its cross-room and SHARP-comparable runs; CSI-Bench follows. Long-window runs (1, 2, 3, 5 s) and frozen LWM then cover both public datasets, with a runner device option so LWM embedding can use a GPU.

### Story 6.1: Padova 80 MHz loader

As a researcher,
I want a loader for the Padova 80 MHz CSI dataset (Meneghello et al.),
So that the whole study can be repeated on independent data from the same hardware family.

**Acceptance Criteria:**

**Given** the downloaded dataset
**When** the story starts
**Then** its format, class list, environments and recording layout are confirmed and recorded in the story spec before any code is written

**Given** the confirmed format
**When** a new `DatasetSource` adapter is added
**Then** it declares its own class list, passes the `DatasetSource` contract test, and gets its own committed `time_block` split
**And** `describe` works for it
**And** if the data has several environments, they map to `room`, so `room_holdout` gives a cross-environment test with no extra code

**Given** the classical sweep
**When** only its `dataset` field is changed
**Then** the sweep runs on Padova and `report` shows a Padova section
**And** `git diff` shows no change outside `adapters/datasets/`, `splits/` and `experiments/`

### Story 6.2: Padova cross-room baseline

As a researcher,
I want Padova scored across environments,
So that the supervisor's same-room question has a measured answer and the result is comparable to SHARP's S6 and S7 tests.

**Acceptance Criteria:**

**Given** the `padova` adapter
**When** `room_holdout` plans are generated
**Then** the plans `bedroom→living_room` and `bedroom→laboratory` are committed under `splits/padova/`, with validation from the train room only

**Given** `experiments/padova_cross_room.yaml`, using the Padova baseline grid with `stride: 9` (72 runs per direction, 144 in total)
**When** `sweep` and then `report` run
**Then** a `padova / room_holdout` section appears with one column per direction and confidence intervals
**And** `git diff` shows no change outside `splits/` and `experiments/`

**And** the report states that this protocol trains on the whole bedroom (S1–S5), whereas SHARP trains on S1 only

### Story 6.3: Padova SHARP-comparable run

As a researcher,
I want classical results on Padova under SHARP's closest protocol,
So that the gap to the paper is measured against its class set, data subset and window length.

**Acceptance Criteria:**

**Given** the `padova` adapter module
**When** a second dataset `padova_sharp` is registered in `adapters/datasets/padova.py`, sharing its loading code
**Then** it loads only the S1 sets, and its classes are `empty, sitting, walking, running, jumping`
**And** it passes the `DatasetSource` contract test, has its own committed `time_block` split (0.6/0.2/0.2), and works with `describe`
**And** its results appear in their own report section, separate from `padova`

**Given** `experiments/padova_sharp.yaml`
**When** `sweep` runs
**Then** it covers `window_len` 346 (about 2 s) as well as the 20–50 packet windows, with the existing feature families and heads
**And** `amplitude_stats` also runs on all 4 antennas, while `paper_beamforming` stays on 2 antennas because it requires exactly 2

**Given** the results
**When** the story closes
**Then** the spec records, next to SHARP's S1 number, how much of the gap each change closed (class set, data subset, window length)
**And** it records whether a Doppler family is worth proposing

### Story 6.4: CSI-Bench loader

As a researcher,
I want a loader for the CSI-Bench activity-recognition data,
So that the study covers a large multi-device benchmark too.

**Acceptance Criteria:**

**Given** the downloaded dataset
**When** the story starts
**Then** its format, class list and recording layout are confirmed and recorded in the story spec
**And** if the data is amplitude-only or otherwise does not fit the `Recording` contract (complex CSI, AD-4), the story stops and asks for a spine amendment instead of working around the contract

**Given** the confirmed format
**When** a new `DatasetSource` adapter is added
**Then** it declares its own class list, passes the `DatasetSource` contract test, gets its own committed `time_block` split, and `describe` works for it

**Given** the classical sweep
**When** only its `dataset` field is changed
**Then** the sweep runs on CSI-Bench and `report` shows a CSI-Bench section
**And** `git diff` shows no change outside `adapters/datasets/`, `splits/` and `experiments/`

**Format confirmed 2026-11-02 (recorded here so the story does not stop on its rule).** Use `Multitask/sub_Human_mat/` (`csi_trace.csi`, complex128, `[tx, rx, subcarriers, packets]`, plus `mactimer`). The H5 files are amplitude-only and are not used. There are 1,778 HAR files, 5 classes (`walking-case1`–`4` merge into `walking`), environments E01–E06 and E27 as rooms, and no empty class. No file has non-finite values, all-zero packets or all-zero subcarriers. Scope for this story: the HP devices (438 files, 2×2×58, about 122 Hz), using `tx0` with `rx0` and `rx1` as antennas `[0, 1]`, as adapter `csi_bench_hp`. The 1×4×14 devices are a later, separate adapter. Exclude `EightreePlug` (bursty timing) and the 6 files under 1,000 packets. Do not use `mactimer` for timing. The story must confirm the antenna pairing across more than the one file sampled so far, and check whether `paper_beamforming`, `amplitude_stats` and LWM accept 58 subcarriers.

### Story 6.5: Device option and parallel-safe dataset cache

As a researcher,
I want LWM embedding on a GPU (MPS or CUDA) and dataset caches that survive parallel sweeps,
So that the public-dataset LWM runs finish in hours, not days.

**Acceptance Criteria:**

**Given** `--device cpu|mps|cuda` or `CSI_BENCH_DEVICE` (default `cpu`)
**When** a run starts
**Then** the runner sets the device and records it in provenance, adapters never choose it, and `run_id` does not include it

**Given** the LWM per-packet embedding cache
**Then** its key includes the device, so CPU and GPU embeddings are never mixed

**Given** the same config on one device
**When** it is rerun
**Then** it reproduces the same metrics; across devices only `run_id` is guaranteed equal, and this is documented

**Given** the torch sources in `pyproject.toml`
**Then** CPU stays the default, macOS wheels provide MPS, and Linux can select a CUDA index

**Given** two sweep processes loading `csi_bench_hp` (and `padova`) at the same time
**When** both convert or open the same cache entry
**Then** no run fails with `cache file unreadable`, and a test reproduces the race before the fix

### Story 6.6: Padova long-window plan and classical baseline

As a researcher,
I want Padova scored at 1, 2, 3 and 5 s windows,
So that it matches the CSI-Bench long-window runs and the LWM long-window rows have a classical comparison.

**Acceptance Criteria:**

**Given** the `padova` adapter
**Then** a `time_block` plan (fractions 0.6/0.2/0.2, `gap_packets` 865) is committed under `splits/padova/`

**Given** `experiments/padova_long_windows.yaml`, the Padova classical grid with `window_len` [173, 346, 519, 865], stride window/2 and the 3 rooms
**When** `sweep` and then `report` run
**Then** the runs appear in their own section (`padova / time_block, gap_packets 865`)

**Given** the CSI-Bench long-window runs made on the Mac (commit `05237c4`)
**Then** their YAML and plan `d6b97e4f25b3` are committed with an expansion test, and `report` shows their section

### Story 6.7: LWM on Padova and CSI-Bench

As a researcher,
I want frozen LWM rows for Padova and CSI-Bench,
So that the LWM-vs-classical comparison covers every dataset before Epic 7.

**Acceptance Criteria:**

**Given** the shared reduced grid (presets `lwm_rescale_phase` and `lwm_amplitude`, embedding `cls`, pooler `mean_std`, heads `svm`, `mlp` and `logreg`, seed 0, `role: comparison`, `feature_selection: null`), justified by the `tor_vergata` medians in sprint-change-proposal-2026-11-10
**Then** these sweeps exist, each with a header giving its run count, stride rule and why each grid value was kept, and an expansion test:

| File | Dataset and plan | Antennas | Windows and stride | Columns | Runs |
|---|---|---|---|---|---|
| `experiments/padova_lwm.yaml` | `padova`, `time_block` `b6010d87ce4d` (gap 64) and `room_holdout` `725a12be6b83`, `99d235eb2f0e` | `ant2`, `ant4` | 10, 30, 50; stride 9 | 3 rooms + 2 directions | 180 |
| `experiments/padova_lwm_long.yaml` | `padova`, the Story 6.6 plan (gap 865) | `ant2`, `ant4` | 173, 346, 519, 865; stride window/2 | 3 rooms | 144 |
| `experiments/padova_sharp_lwm.yaml` | `padova_sharp`, `7640fb8ad320` (gap 346) | `ant2`, `ant4` | 346 and 50 (control, as in Story 6.3) | S1 | 24 |
| `experiments/csi_bench_hp_lwm.yaml` | `csi_bench_hp`, `d6b97e4f25b3` | `ant2` | 122, 244, 366, 610; stride window/2 | 7 rooms + all 7 pooled | 192 |

**When** the sweeps and then `report` run
**Then** each (dataset, protocol) section shows the LWM rows next to the classical rows, with confidence intervals

`room_holdout` uses short windows only, because its plans have gap 64 and the gap must exceed the window. CSI-Bench uses long windows only, because its short-window classical runs are incomplete.

## Epic 7: Other pretrained models

The researcher can add WavesFM, Mantis-8M and an ImageNet model as rows in the same table, each as one new extractor.

### Story 7.1: WavesFM extractor

As a researcher,
I want frozen WavesFM features per window,
So that a wireless foundation model with a built-in WiFi sensing task appears in the comparison.

**Acceptance Criteria:**

**Given** the `wavesfm` adapter
**Then** `_vendor/` contains `models_vit.py` and `pos_embed.py` at commit `0279d3b11afd7691f2565a8361a5d0d60b3ae363`, with the `pos_embed` import changed to a relative import
**And** `SOURCE.md` records that edit and the model arguments (`vit_multi_small`, 1 channel, 224/16, conditional LN)

**Given** weights pinned at `6a63011e7cbc4ea05ec664d02643e7330c9cc503`
**When** they are loaded
**Then** keys `vis_patch_embed.0.*` are remapped to `vis_patch_embed.*`
**And** the strict-load test passes: only head keys may be missing, and no keys are unexpected

**Given** a window
**When** the adapter runs
**Then** the time × subcarrier amplitude window is resized to 224×224 inside the adapter, and one feature vector per window is returned
**And** `min_window_len` is declared

**And** the frozen-checksum and NaN/constant checks pass, and timm stays pinned at 1.0.30

### Story 7.2: Mantis-8M extractor

As a researcher,
I want frozen Mantis-8M features per window,
So that a general time-series foundation model appears in the comparison.

**Acceptance Criteria:**

**Given** the `mantis` adapter using `mantis-tsfm`
**When** it loads `paris-noah/Mantis-8M` at revision `bc7d5ab40c02133386a28e2c127f35c17c86901d`
**Then** the model runs on the device set by the runner

**Given** a window
**When** the adapter runs
**Then** the subcarrier amplitude series over the window are the input channels, with a param for Mantis's channel-reduction adapter
**And** one feature vector per window is returned

**And** the frozen-checksum and NaN/constant checks pass, and the weights revision is recorded in provenance

### Story 7.3: ImageNet vision-model extractor

As a researcher,
I want a frozen ImageNet-pretrained vision model as a per-window extractor,
So that the comparison includes the classic transfer-learning baseline.

**Acceptance Criteria:**

**Given** one timm backbone chosen in this story
**Then** it is added to the spine Stack table with its pinned weights revision (spine amendment: Stack only)

**Given** a window
**When** the adapter runs
**Then** the amplitude window is treated as an image, using fixed ImageNet normalisation constants inside the adapter
**And** one feature vector per window is returned

**And** the frozen-checksum and NaN/constant checks pass

### Story 7.4: Full comparison sweep

As a researcher,
I want one sweep covering every method,
So that the supervisor's final table can be regenerated with one command.

**Acceptance Criteria:**

**Given** `experiments/full_comparison.yaml` (classical, paper reference, LWM, WavesFM, Mantis, ImageNet), run under `time_block` and both `room_holdout` plans on `tor_vergata`, and under `time_block` on each verified public dataset
**When** `sweep` and then `report` run
**Then** each (dataset, protocol) section lists every method with its best variant chosen on validation, plus confidence intervals
**And** the reference rows sit next to the published numbers

**And** every row shows its regeneration command
**And** rerunning one of them reproduces the same `run_id` and metrics
