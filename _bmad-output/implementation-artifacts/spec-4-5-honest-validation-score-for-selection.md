---
title: 'Honest validation score for selection'
type: 'bugfix'
created: '2026-11-02'
status: 'in-progress'
baseline_commit: '1441d521a3ee965bc160b1c5d150b78b29aa5d50'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-4-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `report` picks each table's best variant on val metrics from a model refitted on train ∪ val, so the score is in-sample and favours heads that memorise. Test scores are valid, but which variant each table shows, and so its number, may be wrong.

**Approach:** Record the tuning-stage val macro-F1 (`head.val_score` of the chosen candidate) in `metrics.json` and the index, select on it everywhere, then rerun the 5,084 reported runs to fill it in, with old metrics kept, ids stable, test metrics checked, and a before/after table per section.

## Boundaries & Constraints

**Always:** `seeds.<seed>.tuning_val_macro_f1` holds the winning candidate's score and `mean_std.tuning_val.macro_f1` holds `[mean, std]`, written only when every seed has a score. `val` metrics stay and are documented as in-sample. Selection in `report`, the leakage rows and the notebook uses the tuning score only. A done run without it is excluded, with a stderr note giving the count and `csi-bench rescore-runs --runs <root>`. `rescore-runs` backs up every `metrics.json` and `provenance.json` outside `--runs` before any rerun, never overwrites a backup, skips runs that already have the score (so it resumes), refuses a run whose recomputed `run_id` differs from its directory, and continues past a failed run, listing it and exiting 1. Run ids and test metrics must match; any mismatch is listed, and the new metrics stay in place with the old ones in the backup.

**Never:** Change the refit on train ∪ val, a `run_id`, an adapter `VERSION`, `_vendor/`, a split plan, or `tune_and_refit`. Backfill the score from `selection.candidates` instead of rerunning. Make `sweep` overwrite old runs: `_completed_seeds` stays as it is. Delete any run. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Memorising head | A: in-sample 1.00, tuning 0.70; B: in-sample 0.90, tuning 0.85 | B selected, in the table and the leakage rows | — |
| Old run | done, no tuning score | excluded, counted in the note | exit 0 |
| Only old runs | all excluded | note plus `no finished runs to report` | exit 0 |
| Selector run | three candidates | score of the chosen candidate | — |
| Resumed seed | new seed added to a run whose old seed has no score | no `mean_std.tuning_val`; run excluded until rescored | — |
| Rescore resumed | interrupted, then rerun | scored runs skipped; existing backups untouched | — |
| Bad backup path | `--backup` inside `--runs` | nothing rerun | `ConfigError`, exit 2 |
| Test mismatch | new test metrics differ from the backup | listed in the comparison output | run kept |

**Decision (2026-11-02, rescore timing):** the full rescore runs only on a committed tree (recorded time about 67 h run one at a time; a dirty tree would mark every run `dirty: true`, hide it from `report`, and make a later `sweep` re-execute it). This session implements and verifies on a stratified sample in a scratch copy, then stops. The human commits and starts `rescore-runs`. A later session writes the before/after table and regenerates the HTML; the story stays `in-progress` until then.

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/run.py:632-668` `_run_seed` -- `best_score` is already the honest score; return it too. `:794-822` `_execute` -- carry scores from `previous` the way `seed_selection` does and pass them to `metrics_document`. `:304` `_completed_seeds` -- do not touch.
- `src/csi_bench/core/results_schema.py:191` `metrics_document` -- new optional arg; do not add to `SPLIT_NAMES` (code iterates it as metric splits). `:251` `INDEX_TABLE_SCHEMA` -- add `tuning_val_macro_f1_mean/_std`.
- `src/csi_bench/runner/report.py:158` `_best` -- select on the tuning column (also reached by `_leakage`, `_comparison`). `:93` `_index_row` -- read it, in its own try so old runs still index. `:298` `usable` -- exclude and note; count only runs that would otherwise be shown. `:233` `_variants` -- add the tuning column and label `val F1` as in-sample. `:388` heading wording.
- `src/csi_bench/runner/rescore.py` (new) -- `rescore_runs` iterates `rs.iter_run_dirs`, using each run's own `config.yaml`, `load_config`, `compute_run_id`, `_execute(force=True)`. `rescore_report` compares backup and store. Old selection replays the in-sample rule over backup rows in `iter_run_dirs` order; reuse `_index_row` fields, `_cell`, `_grid`.
- `src/csi_bench/runner/cli.py:140` -- add `rescore-runs --runs --backup --workers` and `rescore-report --runs --backup --out`. Default backup is `results/metrics-before-4-5/` (gitignored, outside `runs/`).
- `notebooks/01_classical_overview.ipynb` -- the source cell selects with `val_macro_f1_mean.idxmax()`; switch to the tuning column and drop null rows.
- `tests/test_run.py:210,451` -- seed key sets gain the score. `tests/test_report.py:59` -- the `_write_run` helper needs the score, plus the memorising case. `tests/test_results_schema.py` -- document shape.
- `_bmad-output/reports/lwm-vs-classical.html` -- hand-written, with no generator in the repo and prose conclusions tied to its numbers. Regenerate after the rescore with a scratch script reading `runs/`, keep the layout, and rewrite any prose the numbers no longer support.
- Store facts (checked 2026-11-02): 5,084 run dirs, all `done`, none dirty, all recomputed ids match. `runs/index.parquet` is stale at 4,868 rows (no Padova) until `report` runs. Recorded time: LWM 50.0 h, Padova 14.9 h, other classical 2.2 h. The packet cache is present. RAM is 11 GB and one LWM run peaks at about 3 GB, so reruns stay sequential.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/results_schema.py` -- document fields, index columns, in-sample docstring -- one schema
- [x] `src/csi_bench/runner/run.py` -- record the score per seed, carrying it across resume -- AC1
- [x] `src/csi_bench/runner/report.py` -- index, selection, exclusion note, variants table -- AC2
- [x] `notebooks/01_classical_overview.ipynb` -- select on the tuning column -- no table left on the old score
- [x] `src/csi_bench/runner/rescore.py`, `src/csi_bench/runner/cli.py` -- backup, rerun, comparison commands -- AC3
- [x] `tests/test_report.py`, `tests/test_run.py`, `tests/test_results_schema.py`, `tests/test_rescore.py` (new) -- matrix cases, including the guard that fails if selection reverts -- coverage
- [x] scratch copy of a stratified sample (each dataset, splitter, head, family) -- run `rescore-runs` and `rescore-report`; record seconds per run and test-metric matches in Implementation Notes -- required before the full rescore
- [ ] real store, after the human commits -- `rescore-runs`, `rescore-report`, regenerate `_bmad-output/reports/lwm-vs-classical.html` -- not in this session, per the rescore timing decision

**Acceptance Criteria:**
- Given a finished run, when `metrics.json` is written, then it has the per-seed score and its mean and std, and the `val` metrics are unchanged.
- Given a store of old and new runs, when `report` runs, then only runs with the score are selected from, in every section, and old runs are counted in the note.
- Given the rescored store, when `rescore-report` runs, then it shows every changed cell per `(dataset, splitter)`, plus test-metric mismatches and run-count and id checks: 5,084 before and after, same ids.

### Review Findings

- [x] [Review][Decision] (resolved 2026-11-02: refuse and repair, applied) Dirty tree only warns, and a dirty rescore cannot be repaired — The timing decision says the rescore runs only on a committed tree, but `rescore-runs` warns once at startup and continues. Runs finished while dirty count as already scored, so a later `rescore-runs` skips them while `report` hides them. `rescore-report` compares them anyway and exits 0, so its table can show cells `report` will not show. A `--backup` inside the repo that is not gitignored also makes every run dirty, after the startup check.
- [x] [Review][Decision] (resolved 2026-11-02: back them up, applied) Old predictions are overwritten with no copy — The rerun replaces `seeds/<seed>/predictions.parquet`, and only `metrics.json` and `provenance.json` are backed up. If a run's test metrics mismatch, the old predictions that could explain it are gone. Copying them costs 1.3 GB (247 GB free).
- [x] [Review][Patch] A later backup can record rescored metrics as old, so the mismatch check compares new with new and passes [src/csi_bench/runner/rescore.py:50]
- [x] [Review][Patch] `rescore-report` passes scored runs that are not `done`, and runs whose config or status is unreadable [src/csi_bench/runner/rescore.py:239]
- [x] [Review][Patch] A rerun that loses its intervals passes as no mismatch, and no test covers an interval mismatch [src/csi_bench/runner/rescore.py:165]
- [x] [Review][Patch] A mistyped or empty `--runs` reports success, against the documented exit 2 [src/csi_bench/runner/rescore.py:74]
- [x] [Review][Patch] No test covers a real rerun failure: status `failed`, old metrics kept, retried later [tests/test_rescore.py:133]
- [x] [Review][Patch] No test shows a failed backup stops the rerun [tests/test_rescore.py:124]
- [x] [Review][Patch] No test shows the before column reads the backup and not the store [tests/test_rescore.py:204]
- [x] [Review][Patch] No test shows `rescore-runs --workers` is rejected when invalid and reaches `_execute` when valid [tests/test_rescore.py]
- [x] [Review][Defer] Ties on the tuning score go to the first run in directory order [src/csi_bench/runner/report.py:180] — deferred: pre-existing rule; needs no rerun to change, so measure tied cells on the rescored store first.
- [x] [Review][Defer] `rescore-report` does not show the deciding scores or a test-score delta, and has no leakage before/after [src/csi_bench/runner/rescore.py:174] — deferred: reads the backup only, so it can be added after the rescore with no rerun.
- [x] [Review][Defer] `rescore-report` does not check `val` metrics or the score against the old `selection` score [src/csi_bench/runner/rescore.py:155] — deferred: the backup holds the data, so it can be added after the rescore with no rerun.
- [x] [Review][Defer] Notebook: saved output is stale, its test only matches source text, a stale index raises `KeyError`, and a part-way rescore plots nothing without a message [notebooks/01_classical_overview.ipynb] — deferred: re-execute after the rescore; `csi-bench report` clears the `KeyError`.

Rejected:
- `low` Never-finished run fails `rescore-report` after `rescore-runs` exits 0 -- the store has none (all 5,084 `done`), and failing loudly is acceptable.
- `low` Old rows read post-rescore status -- every stored run was `done` before the rescore.
- `low` `rescore-report` does not apply the split-hash refusal -- `report` applies it to the real store, which would fail there first.
- `low` `_check_backup` rejects only one nesting direction -- needs an unusual `--runs results/runs --backup results`.
- `low` Duplicate run ids collapse in the report dict -- none in the store; `rescore-runs` already fails such a run through `run_dir`.
- `low` No lock against concurrent `sweep` or `rescore-runs` -- one user, and only runs with the same id collide.
- `low` Null std in `mean_std.tuning_val` crashes `--all-variants` -- `metrics_document` never writes one.
- `low` Partial multi-seed failure leaves new predictions beside old metrics -- every stored run has one seed.
- `false` Run-id refusal skipped for already-scored runs -- the refusal guards a rerun, and these are not rerun.
- `false` `_git` returning `None` hides a dirty tree -- provenance uses the same call, so those runs are recorded clean and `report` shows them.
- `low` No README or runbook for the new commands -- no such doc exists for any command; `--help` and the spec cover them.
- `low` Other untested branches (`--out` failure, stdout output, `_listed` truncation) -- small and loud on failure.
- `spec-edit` Code Map names `_write_run`, and the Verification line has no pass condition -- the fix edits this spec.

## Implementation Notes

**Sample rescore (2026-11-02), scratch copy, real store untouched.** 123 runs covering all 66 strata of (dataset, splitter, head, extractors, selector): the fastest run of each, plus one random run per stratum with mean recorded time under 120 s. Run one at a time with default workers, on the dirty tree.
- Time: 1,092 s of reruns, 8.9 s per run. The fastest-per-stratum pick makes this mean unrepresentative, so scale by stratum: rerun/recorded time per stratum times its recorded total gives about 70 h for the full store (67.1 h recorded).
- Checks: exit 0, 123 before and after, same ids, 0 test-metric mismatches (exact comparison of per-seed test metrics, `mean_std.test` and `intervals.test`), 0 `val` metric differences, and on selector runs the new score equals the old `selection` score of the chosen candidate.
- Cells: 6 of 100 changed in the sample. This only shows the comparison works: the sample holds few variants per cell, so it says nothing about the real tables.
- Resume: a second `rescore-runs` reran nothing and left the backups byte-identical. `--backup` inside `--runs` gave exit 2.

**Choices the spec left open.**
- `rescore-runs` leaves a run with no `metrics.json` alone (it never finished, so it is not a reported run), and reruns any run with a `metrics.json` that is not `done` with the score, which covers interrupted reruns.
- A run whose rerun fails keeps its old `metrics.json` but has status `failed`, so `report` hides it until a later `rescore-runs` succeeds.
- Superseded by code review (2026-11-02): `rescore-runs` refuses a dirty tree unless `--allow-dirty`, checks again after the backup, and on a clean tree reruns scored runs that were made dirty. It also copies old predictions, and copies nothing for a run that already has the score.
- `rescore-report` exits 1 if a check fails (mismatch, id difference, run without the score), and 2 if there is nothing to compare. Superseded by code review (2026-11-02): dirty runs are hidden from the after side and fail the check, unless `--allow-dirty`; runs not `done` and unreadable runs fail it too.

**Before the full rescore.** Commit everything, including this spec and `sprint-status.yaml`. Keep the tree clean for the whole run: provenance reads git state as each run ends, so an edit during the roughly 70 h marks later runs dirty.

**Code review skipped at the user's request (2026-11-02); nothing staged or committed.** The diff was read against the tasks and the I/O matrix: every matrix row has a passing test. The spec stays `in-progress` until the full rescore, the before/after table and the HTML are done.

**After code review (2026-11-02).** 569 tests pass; ruff clean. A 3-run scratch copy (Padova knn, LWM logreg, amplitude svm) confirmed: dirty tree refused with exit 2, `--allow-dirty` run exit 0, strict `rescore-report` exit 1 on 3 dirty runs, `--allow-dirty` report exit 0, 0 test-metric mismatches. The 123-run sample was not repeated. The earlier note that code review was skipped no longer holds.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests` -- expected: clean
- `uv run csi-bench rescore-runs --runs <scratch-sample> --backup <scratch-backup> --allow-dirty` -- expected: exit 0, ids unchanged
- `uv run csi-bench rescore-report --runs <scratch-sample> --backup <scratch-backup> --out <scratch>/compare.md --allow-dirty` -- expected: exit 0
