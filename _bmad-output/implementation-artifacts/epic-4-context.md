# Epic 4 Context: Generalisation protocols

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher score any method across rooms (train on one room, test on the other) and see a 95% confidence interval on every test score, with all results kept in one readable store and every table choosing its best variant on an honest validation score. Within-room scores sit too close to the ceiling to separate methods, so a strict out-of-distribution protocol and uncertainty estimates are what make later comparisons meaningful. Reported val metrics come from a model refitted on train ∪ val, so they are in-sample and favour heads that memorise. Test scores are not contaminated, but which variant each table shows, and therefore its number, may change. Story 4.5 runs next, ahead of all other remaining work, so no further sweep is selected on the inflated score.

## Stories

- Story 4.1: Readable run layout and a report section per protocol
- Story 4.2: `room_holdout` splitter
- Story 4.3: Confidence intervals from a recording bootstrap
- Story 4.4: Cross-room classical baseline
- Story 4.5: Honest validation score for selection

## Requirements & Constraints

- **Additive only.** Keep every existing run, result and code path. Nothing in the results store is lost, by migration or by rerun.
- **Readable run layout.** Run directories are `runs/<dataset>/<splitter>/<slug>--<run_id>/`, with `slug = <extractors>-<selector>-<head>-<rooms>-w<window_len>` sanitised to `[a-z0-9_+-]`. `run_id` is the sole identity for lookup, deduplication and resume; the slug is only for reading. `runs/index.parquet` and `runs/_dataset/` stay where they are.
- **Migration** of old-layout runs is one-off and idempotent, deletes nothing, preserves the run count, and rebuilds the index.
- **Report grouping.** One section per `(dataset, splitter)`. `time_block` and `random_window` sections have room columns (r1, r2, mixed); `room_holdout` sections have direction columns (r1→r2, r2→r1). The mixed-split-hash refusal applies only within one column. The leakage block pairs the `random_window` and `time_block` sections of the same dataset. `results_api` and notebooks must keep working.
- **Strict OOD.** Cross-room validation uses the train room only; the test room is never seen before scoring. There is no oracle-selection row.
- **Confidence intervals.** Every test accuracy and macro-F1 carries a 95% interval from a class-stratified bootstrap over test *recordings*, not windows. Existing runs are backfilled from saved predictions with no retraining. A class with a single test recording (e.g. r1 `empty`) adds no spread and is flagged in the report.
- **Honest selection.** Every report cell, including the leakage rows, shows the variant with the best tuning-stage val macro-F1, reported with its test score. A done run lacking that score is excluded, and the report notes how many were excluded and the command that fixes them. A test must fail if selection reverts to the in-sample score.
- **Rescoring the store.** All 5,084 reported runs are rerun (`tor_vergata/time_block` 2,919, `tor_vergata/room_holdout` 1,946, `padova/time_block` 216, `tor_vergata/random_window` 3). Measure the rerun time on a sample before starting.
  - First copy every old `metrics.json` outside `runs/`.
  - Run ids must not change, and test metrics must match the old ones; list any mismatch rather than ignoring it.
  - Produce a before/after table per section showing every cell whose selected variant or score changed, and regenerate `_bmad-output/reports/lwm-vs-classical.html`.
- Leakage rules still hold: fitted state, feature selection and hyperparameters use train/val only, and test is scored once.
- **Success signal:** one store produces separate sections per dataset and protocol, every cell shows an interval, the cross-room classical baseline gives OOD numbers to compare with LWM, and `report` selects on the tuning-stage score in every section.

## Technical Decisions

- **No new port.** `room_holdout` is a new adapter behind the `Splitter` port, registry name `room_holdout`.
- **Layout and schemas live only in `core/results_schema.py`**, shared by `ResultsStore` and `results_api`. No other module hard-codes run paths.
- **`room_holdout` splitter.** Params: `train_rooms`, `test_rooms`, `fractions`, `gap_packets`. Train-room recordings are split into time blocks for `train` and `val` with the same gap rule as `time_block`; test-room recordings go to `test` in full. Each direction is its own committed plan under `splits/<dataset>/`, with a hash distinct from every other plan. A class missing from `val` or `test` raises `DataError`. It must pass the Splitter contract test and work with `describe --split`.
- **Selection rule.** With `room_holdout`, `selection.rooms` must equal `train_rooms ∪ test_rooms`; config validation rejects anything else. Cross-room is a split protocol of its own, not a room selection.
- **Bootstrap.** A pure function in `core/metrics` with a fixed resampling seed and `B` as a parameter. Intervals are stored in `metrics.json` with the point scores. They capture which recordings landed in test, which seeds cannot.
- **Which val score counts.** Variant selection uses the tuning-stage val score: the val macro-F1 of the chosen candidate, fitted on train only (`head.val_score`). The val metrics of the refitted model are in-sample diagnostics and never select anything. The head refit on train ∪ val itself stays unchanged.
- **Recording the score.** `metrics.json` holds the tuning-stage score per seed, with its mean and std over seeds. The existing `val` metrics remain, documented as in-sample.
- **`run_id` stability.** `run_id` hashes the resolved config, adapter `VERSION`s and `_vendor/` hashes, so the 4.5 change must leave all of these unchanged to keep ids stable.
- Errors use `core.errors` (`ConfigError`, `DataError`); hashes are SHA-256 of canonical JSON, first 12 hex chars.

## Cross-Story Dependencies

- 4.1 came first: its layout and report sections let cross-room runs (4.2) and later datasets share one store.
- 4.3 backfills intervals for all runs in the migrated layout. 4.4 depends on 4.1–4.3.
- 4.5 changes selection for every section built by 4.1–4.4, so the cross-room and within-room tables may change after its rescore.
- Remaining order is 4.5, then 6.3, 6.4, 6.2, then Epic 7. Epic 5 and earlier tables need no code change beyond 4.5, but their selected variants may change; Epic 7 inherits the fix.
- Padova numbers already quoted to the supervisor stay provisional until 4.5 is done.
- Epic 6 datasets map environments to `room`, so `room_holdout` gives cross-environment tests with no splitter code (Story 6.3 commits plans under `splits/padova/`). Story 7.4 reports every `(dataset, protocol)` section.
