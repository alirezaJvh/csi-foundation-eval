# Epic 1 Context: Reproducible dataset access

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

The researcher can load all 19 own recordings (`tor_vergata`), generate a committed time-block train/val/test split, and inspect counts per recording, class, room and split with `describe`. Story 1.1 sets up the greenfield uv project, pinned CPU stack, quality tooling, core contracts and the adapter registry.

## Stories

- Story 1.1: Project setup with the pinned CPU stack and an adapter registry
- Story 1.2: Load the own recordings as labelled Recordings
- Story 1.3: Generate the committed time-block train/val/test split
- Story 1.4: Inspect the dataset with csi-bench describe

## Requirements & Constraints

- Every method gets identical splits, windows and metrics; no leakage; reproducible runs with hashed configs.
- Adding or removing a component means adding or deleting one adapter module; dependency rules are enforced by import-linter.
- Runs on local CPU only (no nvidia packages); ruff and ty pass with `_vendor/` excluded; each port has a contract test.
- `data/csi_proc_python/` stays unchanged; `.cache/` and `runs/` are git-ignored.

## Technical Decisions

- Python 3.12, uv; torch/torchvision CPU index via `tool.uv.sources`; `pandas==3.0.6` via `override-dependencies`; pins per the architecture Stack table (numpy 2.5.3, scikit-learn 1.9.1, pyarrow 25.0.1, pydantic 2.13.5, PyYAML 6.0.3, huggingface-hub 2.1.1, timm 1.0.30, mantis-tsfm 1.1.0, mrmr-selection 0.2.8, ruff 0.16.10, ty 0.0.84, import-linter 2.15, pytest 9.1.1).
- Layout: `src/csi_bench/{core, adapters/<stage>, config, runner, results_api}`, `experiments/`, `splits/`, `notebooks/`, `tests/`.
- Dependency direction: core imports nothing internal; adapters import only core; adapters never import each other; only runner imports adapters (by discovery via `pkgutil.walk_packages`).
- Adapters register via `@register("<stage>", "<name>")`, declare a pydantic `Params` and integer `VERSION`; `affects_result=False` params are excluded from run_id; duplicate registration raises ConfigError.
- Frozen dataclass contracts (Recording: csi complex64 `(antennas, packets, subcarriers)`, antenna_ids, recording_id, room, label, meta); window_id `recording_id@start:end`; labels `0 empty, 1 sitting, 2 standing, 3 walking`.
- Hashes: SHA-256 of canonical JSON (sorted keys, no whitespace), first 12 hex chars. Errors: ConfigError, DataError, ExtractorError. stdlib logging only.
- Time-block splitter with `gap_packets` (initial 64), SplitPlan committed at `splits/<dataset>/<split_hash>.json` with source_sha256 per input; raises DataError if a class lacks val/test windows.
- `describe` writes `runs/_dataset/<dataset>.parquet` (per-recording packet counts; per class, room and split counts).

## Cross-Story Dependencies

- 1.1 provides the registry, errors, hashing and tooling used by 1.2-1.4; 1.2 provides Recordings for 1.3; 1.3 and 1.2 feed 1.4.
