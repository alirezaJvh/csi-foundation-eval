---
title: 'Confidence intervals from a recording bootstrap'
type: 'feature'
created: '2026-10-12'
status: 'done'
route: 'dispatch'
baseline_commit: '0c330255adc17c0654391683e1e47cdb74270f8e'
baseline_tree_pre_story: '729330075ba6ba898f136d5c2bf29b37ae863506' # working tree incl. uncommitted 4.1 + 4.2
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-4-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every test score is a single number from one split, so a 0.5-point gap between methods cannot be told apart from which recordings happened to land in `test`. Seeds do not capture this.

**Approach:** Add a pure bootstrap in `core/metrics`. It resamples test *recordings* with replacement, stratified by class, and returns 95% percentile intervals for test accuracy and macro-F1. New runs store the interval in `metrics.json`. Existing runs get it from their saved `predictions.parquet`, with no retraining. The report shows each cell's score with its interval and flags classes that have only one test recording.

## Boundaries & Constraints

**Always:** `bootstrap_intervals(predictions, n_classes, n_resamples=1000, seed=0, level=0.95)` is pure numpy (no pandas, no I/O). Its input is one `(recording_id, y_true, y_pred)` triple of arrays per seed, all over the same test recordings. A recording's class is its `y_true`. Each recording has a single label; a recording with mixed labels → `DataError`. Each resample draws, per class, as many recordings as that class has, with replacement. The same draw is used for every seed, and the statistic is the seed mean of accuracy and macro-F1 (macro over all `n_classes`, `zero_division=0`), so the interval belongs to the score the report shows. Compute from per-recording confusion matrices, never by re-scoring windows. Use percentile bounds. The result is `{"accuracy": [lo, hi], "macro_f1": [lo, hi], "n_resamples", "seed", "level", "unit": "recording", "single_recording_classes": [class names]}`. It is stored as the top-level `intervals.test` of `metrics.json`; every other key stays unchanged. `run` computes it after all seeds finish, reading every seed's `predictions.parquet`, so a resumed run is covered too. The index gains `test_macro_f1_lo/hi`, `test_accuracy_lo/hi` and `single_recording_classes` (comma-joined, `""` if none); they are null when a run has no intervals. The same seed and `n_resamples` give bit-identical intervals.

**Decisions:** (1) Backfill extends `csi-bench migrate-runs`: after moving dirs, it fills `intervals` for every `done` run that lacks them, rewriting only `metrics.json`, and is idempotent. `report` does not write run files; it prints a stderr note with the count of runs without intervals, pointing to `migrate-runs`. (2) A comparison cell reads `F1 98.50 [97.12, 99.40] acc 99.10 [98.03, 99.81]` (percent, 2 decimals). A cell whose run has `single_recording_classes` gets a trailing `*`, and the table is followed by one footnote line per flagged cell, e.g. `* r2→r1 / <method>: single test recording for empty`. A run with no intervals shows `[-]`. The seed std leaves the comparison cell and stays in the `--all-variants` table, which also gains a `test F1 95%` column.

**Never:** Retrain or re-predict any run. Change `run_id`s, the point scores, or `mean_std`. Add intervals for `val`. Delete run directories, or move them beyond what `migrate-runs` already does. Touch the leakage block's numbers. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| All correct | every window right | interval `[1.0, 1.0]` for both scores | — |
| Spread | 4 classes × 3 recordings, mixed errors | `lo ≤ point ≤ hi`, `lo < hi` | — |
| Single recording | class `empty` has 1 test recording | it appears in `single_recording_classes`; the report flags the cell | — |
| Two seeds | same recordings, different predictions | the interval contains the seed-mean score | — |
| Seeds disagree on recordings | seed 1 is missing a recording | rejected | `DataError` |
| Mixed-label recording | one recording with two `y_true` values | rejected | `DataError` |
| Old run | `metrics.json` without `intervals` | backfilled from predictions; point scores byte-identical | — |
| Re-backfill | every run already has intervals | nothing is rewritten | — |
| Missing predictions | a seed's parquet is gone | run is skipped with a warning; cell shows no interval | — |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/metrics.py` -- add `bootstrap_intervals` next to `evaluate`. Core imports only numpy/sklearn and `core.*` (`tests/test_layering.py`). Use `DataError` from `core.errors`.
- `src/csi_bench/core/results_schema.py:191` `metrics_document` -- leave it unchanged; intervals are added afterwards as `doc["intervals"] = {"test": ...}`. Add `INTERVAL_FIELDS` to `INDEX_TABLE_SCHEMA` (line 227); this is additive, and `results_api.load_index` just reads the parquet. `predictions_path`/`PREDICTIONS_TABLE_SCHEMA` already have `recording_id`, `split`, `y_true` and `y_pred`.
- `src/csi_bench/runner/run.py:716` -- after `per_seed` and before `_write_json(metrics_path…)`, build the intervals from the `test` rows of every seed's `predictions.parquet` (they are all on disk by then, both fresh and previous) and set `document["intervals"]`. Put the reading in one helper (`run_intervals(run_id, seeds, n_classes, root)`, e.g. in a new `runner/intervals.py`), so the backfill reuses it.
- `src/csi_bench/runner/migrate.py` -- after moving dirs, walk `rs.iter_run_dirs`. For a `done` run whose `metrics.json` has no `intervals`, compute them and rewrite only that file (load, add the key, `json.dumps(…, indent=2, sort_keys=True)` as `run._write_json` does). Report the count. Idempotent: runs that already have intervals are not touched. `refresh_index` runs when anything changed.
- `src/csi_bench/runner/report.py` -- `_index_row` reads `metrics["intervals"]["test"]` into the new fields (missing → null). `_cell` formats as decided. `_comparison` collects flagged cells for the footnote. `_variants` adds the `test F1 95%` column. `_leakage` is unchanged. `report` counts runs with null intervals and prints a note to stderr.
- `src/csi_bench/runner/cli.py:141` -- update the `migrate-runs` help text.
- `notebooks/01_classical_overview.ipynb` -- reads the index and `mean_std`. Additive columns keep it working; no edit is needed.
- Store facts: 222 runs, all `done`, all single-seed, all with predictions. `tor_vergata` r1 has one `empty` recording, so r1 and r2→r1 test sets flag `empty`.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/metrics.py` -- `bootstrap_intervals` -- the pure statistic
- [x] `src/csi_bench/runner/intervals.py` (new) -- read a run's test predictions → intervals -- shared by `run` and the backfill
- [x] `src/csi_bench/runner/run.py` -- store `intervals.test` in `metrics.json` -- new runs
- [x] `src/csi_bench/runner/migrate.py`, `src/csi_bench/runner/cli.py` -- idempotent backfill -- existing runs
- [x] `src/csi_bench/core/results_schema.py`, `src/csi_bench/runner/report.py` -- index fields, cell format, single-recording flag, missing-interval note -- report AC
- [x] `tests/test_metrics.py`, `tests/test_run.py`, `tests/test_migrate.py`, `tests/test_report.py`, `tests/test_results_schema.py` -- the I/O matrix, determinism, a run writes intervals, backfill leaves point scores and `run_id`s unchanged and is idempotent, cell and footnote rendering -- coverage

**Acceptance Criteria:**
- Given a finished tiny-fixture run, when `metrics.json` is read, then `intervals.test` exists and its bounds contain the `mean_std` test means.
- Given the real store, when the backfill runs twice, then the first pass fills all 222 runs, the second changes no file, and `report` shows an interval in every cell with the same point scores as before.

### Review Findings

- [x] [Review][Patch] Pin the multi-seed shared-draw seed-mean interval with an exact test — the only multi-seed checks assert `lo <= mean <= hi`; mutants that use seed 0's statistic alone or average per-seed percentile bounds still pass. Extend the window-rescoring test to two seeds: same `draws` for both, average per-resample sklearn scores, `pytest.approx` on the percentiles [tests/test_metrics.py:93]
- [x] [Review][Patch] Missing-interval note always points to `migrate-runs`, even for runs it cannot fill — a run with missing predictions (I/O matrix row) stays without intervals, so the note repeats forever; reword to say runs with missing predictions stay without intervals [src/csi_bench/runner/report.py:314]

**Rejected:**
- `false` — Interval failure in `run` fails a fully trained run (4 layers): unreachable — `labels` come from `by_id[rec].label` so recordings can't mix labels, every seed shares the same `windows`, resumed seeds need their parquet on disk (`_completed_seeds`), and empty test is a `ConfigError` earlier.
- `false` — Null `y_true`/`y_pred` give a misleading error: `_write_predictions` stores `int(y_true)`/`int(y_pred)`, never null.
- `false` — `class_names` kwarg deviates from the frozen signature: additive with a default, and the spec's result schema needs class names in `single_recording_classes`.
- `false` — Transient `.metrics.json.tmp` violates "rewrite only metrics.json": it is renamed onto `metrics.json`; the end state is compliant.
- `spec` — Percentile interval may not contain the point; tests assert it: the AC requires containment, and the fixtures are deterministic, so the tests aren't flaky.
- `spec` — Settings hard-coded and `test F1 95%` header is a literal: spec fixes the defaults and the column name.
- `spec` — Comparison cell drops the seed std: Decision (2).
- `spec` — Comma-joined `single_recording_classes` / commas in class names: the spec mandates the comma join; class names are fixed identifiers.
- `spec` — Footnote order `room / method` differs from command lines: the spec's footnote format.
- `spec` — Backfilled intervals carry no provenance; no per-class recording counts or zero-recording-class flag: beyond the spec's result record.
- `low` — `intervals._write_json` duplicates `run._write_json`: the spec says to mirror it, and `run` importing `intervals` blocks the reverse import.
- `low` — Backfill counts missing/unreadable `status.json` as "could not be filled": `run` writes `status.json` right after `mkdir`, so only a hand-damaged dir hits it.
- `low` — `metrics.json` read twice in `_index_row`; malformed `intervals` silently ignored and never backfilled: needs a hand-corrupted file.
- `low` — `single_recording_classes` stored as a string, or bounds with != 2 entries / lo > hi, accepted: hand-corrupted input only.
- `low` — An uncaught backfill exception skips `refresh_index` after moves: `ArrowInvalid` is a `ValueError` and is caught; others are unlikely.
- `low` — `.tmp` left behind if the write fails midway: rare, harmless.
- `low` — Untested `level` != 0.95, null recording ids, backfill after a partial migrate failure, multi-seed "disagree" through `migrate-runs`: unreachable or unlikely paths.

## Design Notes

Each recording has one label, so the per-class stratum is a set of recordings, and a resample is a sum of per-recording confusion matrices: `C[b] = Σ counts[idx_b]`. Accuracy is `trace/sum`, and macro-F1 comes from the matrix's row and column sums. Over the whole store, B=1000 then costs milliseconds per run instead of re-scoring about 13k windows 1000 times. One draw shared across seeds gives the interval of the seed mean, the number the report shows. A class with one recording always redraws that same recording, so it adds no spread. That is why it is flagged rather than silently narrowing the interval.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests` -- expected: clean
- `uv run csi-bench migrate-runs --runs runs` twice -- expected: the first fills 222 runs, the second reports 0
- `uv run csi-bench report --runs runs` -- expected: intervals in every cell, `empty` flagged in r1 columns, point scores unchanged
