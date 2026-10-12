# Epic 4 Context: Generalisation protocols

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher score any method across rooms (train on one room, test on the other) and see a 95% confidence interval on every test score, with all results — old and new, across datasets and split protocols — kept in one readable results store. Within-room classical baselines already sit at ~98.5–99.9% test accuracy with only 1–1.5 points of headroom, and all numbers are single-split, so within-room scores can no longer separate methods. A strict out-of-distribution protocol plus uncertainty estimates is what makes later comparisons meaningful. This epic runs first among the remaining epics because every later score (LWM, public datasets, other pretrained models) depends on it.

## Stories

- Story 4.1: Readable run layout and a report section per protocol
- Story 4.2: `room_holdout` splitter
- Story 4.3: Confidence intervals from a recording bootstrap
- Story 4.4: Cross-room classical baseline

## Requirements & Constraints

- **Additive only.** Keep every existing run, result, code path and epic. Nothing in the results store is deleted. All existing runs (225) must survive the layout migration with unchanged `run_id`s and metrics.
- **Readable run layout.** Run directories are `runs/<dataset>/<splitter>/<slug>--<run_id>/`, with `slug = <extractors>-<selector>-<head>-<rooms>-w<window_len>` sanitised to `[a-z0-9_+-]`. `run_id` remains the sole identity: lookup, deduplication and resume use it, and the slug is only for reading. `runs/index.parquet` and `runs/_dataset/` stay where they are.
- **Migration** of existing runs is one-off, idempotent (a second run changes nothing), deletes nothing, preserves the run count, and rebuilds the index.
- **Report grouping.** `report` prints one section per `(dataset, splitter)`. `time_block` and `random_window` sections have room columns (r1, r2, mixed); `room_holdout` sections have direction columns (r1→r2, r2→r1). The mixed-split-hash refusal applies only within one column. The leakage block pairs the `random_window` and `time_block` sections of the same dataset. `results_api` and notebooks must keep working.
- **Strict OOD.** Cross-room validation uses the train room only; the test room is never seen before scoring. There is no oracle-selection row.
- **Confidence intervals.** Every test score (accuracy and macro-F1) carries a 95% interval from a class-stratified bootstrap over test *recordings* (not windows). Existing runs are backfilled from saved predictions with no retraining. A class with a single test recording (e.g. r1 `empty`) contributes no spread and must be flagged in the report.
- Leakage rules still hold: fitted state, feature selection and hyperparameters use train/val only, and test is scored once.
- Success signal: one `runs/` store produces separate `tor_vergata / time_block`, `random_window` and `room_holdout` sections; every cell shows an interval; the cross-room classical baseline yields the first OOD numbers to compare against LWM.

## Technical Decisions

- **No new port.** `room_holdout` is a new adapter behind the existing `Splitter` port (`adapters/splitters/room_holdout.py`, registry name `room_holdout`). Other changes are confined to `core/results_schema.py`, `core/metrics`, and `runner/report.py`.
- **Layout lives only in `core/results_schema.py`**, shared by `ResultsStore` and `results_api`. No other module hard-codes run paths.
- **`room_holdout` splitter.** Params: `train_rooms`, `test_rooms`, `fractions`, `gap_packets`. Train-room recordings are split into time blocks for `train` and `val` using the same gap rule as `time_block`; every test-room recording goes to `test` in full. Each direction (r1→r2, r2→r1) is its own committed plan under `splits/tor_vergata/`, with a hash distinct from every other plan. Valid for `role: comparison`. A class missing from `val` or `test` raises `DataError`. It must pass the Splitter contract test and work with `describe --split`.
- **Selection rule.** With `room_holdout`, `selection.rooms` must equal `train_rooms ∪ test_rooms`; config validation rejects anything else. Cross-room is a split protocol of its own, not a room selection.
- **Bootstrap.** A pure function in `core/metrics`: resample test recordings with replacement, stratified by class, with a fixed resampling seed and `B` as a parameter; returns 95% intervals for accuracy and macro-F1. Intervals are stored in `metrics.json` alongside the point scores. They capture which recordings landed in test, which seeds cannot.
- Report cells still show the variant with the best **validation** macro-F1, reporting its test score — now with its interval.
- Errors use `core.errors` (`ConfigError`, `DataError`); hashes are SHA-256 of canonical JSON, first 12 hex chars.

## Cross-Story Dependencies

- 4.1 comes first: the new layout and per-`(dataset, splitter)` report sections are what let cross-room runs (4.2) and later datasets share one store.
- 4.2 needs 4.1's report sectioning to show the cross-room section with direction columns.
- 4.3 is independent of 4.2 but must backfill intervals for all runs in the migrated layout from saved `predictions.parquet`.
- 4.4 depends on 4.1–4.3: `experiments/classical_cross_room.yaml` runs the classical grid and the paper method under both `room_holdout` plans, and the report shows the cross-room section next to the within-room one, with intervals.
- Downstream: Story 5.3 runs LWM under both `room_holdout` plans; Epic 6 datasets map environments to `room` so `room_holdout` gives cross-environment tests with no extra code; Story 7.4 reports every `(dataset, protocol)` section.
