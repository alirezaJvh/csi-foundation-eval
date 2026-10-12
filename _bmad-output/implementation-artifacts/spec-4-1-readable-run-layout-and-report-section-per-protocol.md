---
title: 'Readable run layout and a report section per protocol'
type: 'feature'
created: '2026-10-11'
status: 'done'
route: 'dispatch'
baseline_commit: '0c330255adc17c0654391683e1e47cdb74270f8e'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-4-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Run directories are bare `runs/<run_id>/` hashes, and `report` refuses any store mixing split hashes. So time-block, random-window and (soon) cross-room or other-dataset runs cannot share one store, and nobody can find a run by eye.

**Approach:** Store runs at `runs/<dataset>/<splitter>/<slug>--<run_id>/`, with the layout defined only in `core/results_schema.py` and `run_id` still the only identity. Add a one-off, idempotent `csi-bench migrate-runs` that moves legacy run directories into the layout and rebuilds the index. Make `report` print one section per `(dataset, splitter)`, refusing a split-hash mix only within one room column.

## Boundaries & Constraints

**Always:** `slug = <extractors>-<selector>-<head>-<rooms>-w<window_len>`. Here extractors are sorted names joined with `+`, the selector is `none` when unset, and rooms is `rs.room_setting(...)`. The slug is lowercased, and every character outside `[a-z0-9_+-]` becomes `_`. Lookup, dedup/skip and seed resume go by `run_id`, found by globbing `<root>/*/*/*--<run_id>`. More than one match raises `DataError`; none raises `FileNotFoundError`. The splitter name comes from the committed plan's `splitter` field. Migration resolves the plan through the run's `provenance.split_hash`, then `config.split`, then `find_split(dataset)`. If none resolves, the splitter is `unknown`. Migration only moves (`rename`) and never deletes or rewrites run files. A second run moves nothing and leaves the index untouched. `report` takes the section key from the run's directory and adds a `splitter` column to `index.parquet`. The leakage block pairs `random_window` and `time_block` sections of the same dataset. Reference/comparison tables and the published row behave as today inside each section. The `results_api` signatures stay unchanged.

**Decisions (2026-10-11):**
- **Recursive search.** Migration picks up legacy run dirs anywhere under `--runs`. Today that means `runs/ml-classic-backup/` (216) and `runs/run_paper/` (7), which merge into one store. The emptied folders stay where they are.
- **Duplicate `run_id`s.** When copies share a `run_id`, the copy with the newest `status.finished_at` goes into the layout, unless a layout dir already exists. Every other copy moves to `runs/_duplicates/<origin parent relative to root>/<run_id>/`, which `report` ignores. Nothing is deleted.
- **Real store.** This story runs the migration on the real `runs/`. First it writes `../runs-backup-2026-10-11.tar.gz` (in the project's parent dir). Then it migrates twice; the second pass must move 0. Expected result: 222 runs in the layout, 1 parked (`c5a7175e39e0` from `run_paper`), and `report` shows the sections.
- **Spec size.** The full spec is kept even though it is over 1600 tokens, because layout, migration and report are coupled.

**Never:** Change `run_id`s, metrics or any run file's content. Fall back to the old `runs/<run_id>/` layout for reading or writing. Hard-code run paths outside `core/results_schema.py`. Add the `room_holdout` splitter (4.2) or intervals (4.3). Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| New run | config on the time-block plan | `runs/tor_vergata/time_block/amplitude_stats-none-svm-r1-w40--<id>/` | — |
| Resume / skip | sweep rerun after migration | finished runs are skipped and missing seeds resume in place | — |
| Migrate | legacy `runs/<id>/` dirs | every dir moved and the index rebuilt; the summary prints moved/skipped counts | unreadable run → `unknown` splitter, still moved |
| Migrate again | already migrated | 0 moved and no file touched | — |
| Duplicate | same `run_id` in two legacy dirs, or a layout dir already present | newest (or existing layout dir) kept; others parked under `_duplicates/` and listed | — |
| Report | runs on 2 splitters | two sections, `tor_vergata / random_window` and `tor_vergata / time_block`, plus leakage | — |
| Mixed column | two hashes in one section's `r1` column | refused, naming the section and column | exit 2 |
| Legacy present | unmigrated `runs/<id>/` at report time | note on stderr: run `csi-bench migrate-runs` | report continues |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/results_schema.py:47-72` -- `run_dir` and the path helpers return `root/run_id/...`. Add `run_slug(config: dict)`, `new_run_dir(root, dataset, splitter, slug, run_id)`, a glob-resolving `run_dir`, `iter_run_dirs(root)` and `parse_run_dir(path, root) -> (dataset, splitter, run_id)`. Add `splitter` to `INDEX_TABLE_SCHEMA`. Every other helper keeps its signature and builds on `run_dir`.
- `src/csi_bench/core/splitplan.py:51` -- `read_plan_unit`; add the sibling `read_plan_splitter(path)`.
- `src/csi_bench/runner/run.py:573` -- `_execute` creates the directory. Before `mkdir`, resolve the splitter via `find_split(config.dataset, config.split)`, with `unknown` on `DataError` so the real error is still logged in the run dir. Then use `new_run_dir` only when `run_dir` finds no existing directory. `_completed_seeds`/`_sweep_action` already catch `OSError`, so a missing run reads as "not started".
- `src/csi_bench/runner/report.py` -- `refresh_index` iterates `root.iterdir()`; switch it to `iter_run_dirs`. `_index_row` fills `dataset`/`splitter` from `parse_run_dir`. In `report`, group usable rows by `(dataset, splitter)`, move the hash check into each section's room column, and print sections in sorted order with a `<dataset> / <splitter>` heading. `_leakage` replaces `_on_window_plan` with the section's splitter.
- `src/csi_bench/runner/migrate.py` (new) -- `migrate_runs(root, split_root="splits") -> int`: finds legacy dirs (name matches `^[0-9a-f]{12}$`, has `config.yaml`, not under a `_`-prefixed or layout dir), moves them, then calls `refresh_index` if anything moved.
- `src/csi_bench/runner/cli.py:134` -- add the `migrate-runs --runs runs` parser next to `report`.
- `src/csi_bench/results_api/__init__.py` -- no change needed. It goes through `rs` path helpers.
- `tests/test_report.py:31` `_make_run` builds `rs.run_dir(...)`. Give it a `splitter` argument (default `time_block`) and use `new_run_dir`. The leakage tests monkeypatch `find_split` and should switch to splitter dirs. `tests/test_run.py:85` `_run_dirs` iterates root children; switch it to `rs.iter_run_dirs`.
- `notebooks/01_classical_overview.ipynb` -- reads only `load_index` and must keep running.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/results_schema.py` -- layout helpers, slug and the index `splitter` column -- the single home of the layout
- [x] `src/csi_bench/core/splitplan.py` -- `read_plan_splitter` -- splitter name without loading recordings
- [x] `src/csi_bench/runner/run.py` -- create the layout dir in `_execute` -- new runs land in the layout
- [x] `src/csi_bench/runner/migrate.py`, `src/csi_bench/runner/cli.py` -- migration and its command -- one-off move
- [x] `src/csi_bench/runner/report.py` -- index from the layout, sections, per-column hash refusal, leakage by section -- AC 3
- [x] `tests/test_results_schema.py`, `tests/test_migrate.py` (new), `tests/test_report.py`, `tests/test_run.py`, `tests/test_sweep.py` -- slug sanitising, lookup (none/one/many), the I/O matrix, idempotency and the run count -- coverage

**Acceptance Criteria:**
- Given a fresh run, when it finishes, then `rs.run_dir(run_id)` resolves to the layout path and a sweep rerun skips it.
- Given a migrated store, when `report` runs, then the `run_id` set and every `metrics.json` are byte-identical to before.
- Given the notebook, when it is executed against a migrated store, then it runs without errors.

### Review Findings

- [x] [Review][Decision] A run first created while its split plan cannot be resolved stays under `<dataset>/unknown/` for good — `_execute` creates `<dataset>/unknown/<slug>--<id>` so the error is logged. Once the plan is fixed, `run_dir` finds that directory and the run finishes there, so it is reported under `<dataset> / unknown` and never reaches its real section or the leakage table. Neither `run` nor `migrate-runs` moves it out of `unknown`. The spec says to use `new_run_dir` "only when `run_dir` finds no existing directory", so moving it is a change to the spec's intent. (medium; blind-hunter+edge-case-hunter) — resolved: option 1, `_execute` moves the dir out of `unknown` once the plan resolves; test `test_run_leaves_the_unknown_section_once_the_plan_resolves`
- [x] [Review][Decision] The split-hash check now also refuses reference runs, not only comparison runs — `report` loops `for role in ("comparison", "reference")`. Before, only comparison runs were checked. A store whose reference runs mix plans in one column now exits 2 where it used to report. The real store still reports (exit 0). It is unclear whether "refuse only within one room column" was meant to cover reference runs too. (low; acceptance-auditor) — resolved: option 1, keep checking both roles (the `random_window` section holds only reference runs); recorded in the Spec Change Log
- [x] [Review][Patch] When `migrate-runs` fails partway, it does not list what it already moved or parked, and prints no summary [src/csi_bench/runner/migrate.py:775] — the failure path is now tested too (`test_failure_midway_lists_what_moved_and_refreshes_index`), which closes the deferred item below
- [x] [Review][Patch] The `_duplicates/<origin>/<run_id>` path is built in `migrate.py`, which breaks the never-rule "Hard-code run paths outside `core/results_schema.py`". Move it into an `rs` helper [src/csi_bench/runner/migrate.py:735]
- [x] [Review][Patch] The leakage block prints when one side is empty: it checks only that both sections exist, not that both have rows. Skip it unless `random_side` and `block_side` are both non-empty [src/csi_bench/runner/report.py:304]
- [x] [Review][Patch] Add a test for the new `run_sweep` `DataError` branch, where one `run_id` has several directories: exit 1, `<id> failed`, and the other runs continue [src/csi_bench/runner/run.py:817]
- [x] [Review][Patch] Pin the per-column hash scoping. `test_different_hashes_in_other_columns_or_sections_are_fine` only builds a different section; add an r2 run on `h2` in `time_block` [tests/test_report.py:431]
- [x] [Review][Defer] The `migrate_runs` failure path (exit 2, partial index refresh) is untested [src/csi_bench/runner/migrate.py:775] — deferred: a one-off tool, reruns are idempotent, and the next `report` rebuilds the index

**Rejected**
- false — The notebook (AC3) has no proof. I ran its code cells against the migrated store and they finished without errors.
- false — Real-store byte identity (AC2) is unproven. I compared the store with `../runs-backup-2026-10-11.tar.gz`: the same 222 run_ids, and every `metrics.json` hash is identical.
- false — Docs still show the old `runs/<run_id>/` paths. No README or docs file references them.
- false — Dataset names go into paths unsanitised. `config.dataset` is validated against the dataset registry, and splitter names come from committed plans.
- false — `finished_at` is compared as a string, which could pick the wrong copy. Every value is written by `_now()` as UTC ISO-8601, so string order is time order.
- false — `new_run_dir` does not validate `run_id`. Run ids are computed 12-hex hashes.
- false — An unreadable run is dropped. It is moved, as the I/O matrix requires; the index cannot describe a run whose config is unreadable.
- low — Each path lookup globs the whole store, so the index rebuild grows with the square of the run count. A full `report` on the 222-run store takes 2.8 s; fixing it needs new directory-based helpers.
- low — When a `run_id` has two layout dirs, both disappear silently. Neither `run` nor `migrate` can produce that state, and the sweep reports it as a failure.
- low — The parked name gets a `.N` suffix when the target exists. This is a no-delete safeguard and does not happen in the real store.
- low — There is no `--splits` or `--dry-run` flag. `splits` is resolved from the working directory, as in every other command.
- low — A plan whose `splitter` is null, symlinked run dirs, and a run directory vanishing while the report runs. All are unreachable in practice.
- low — `migrate` stops when a `run_id` already has several layout dirs. That is a loud failure on a state the tools never produce.
- spec — The duplicate kept is chosen by `finished_at` alone, not by status. This was the decision of 2026-10-11.
- spec — The random side is hard-coded as `random_window` rather than the plan unit. The spec pairs `random_window` with `time_block`.
- spec — The hash check is weaker than before; print the hash in each heading. Per-column refusal is the spec's rule.
- spec — The dataset comes from the path rather than the config. The spec says to take the section from the directory.
- spec — A window run with a missing plan falls back to the block plan. That is the spec's resolution order.

## Implementation Notes

## Spec Change Log

- 2026-10-11 (code review), clarification: the per-column split-hash refusal applies to each role (comparison and reference) separately. A `random_window` section holds only reference runs, so without this it would have no check at all.
- 2026-10-11 (code review), amendment to "use `new_run_dir` only when `run_dir` finds no existing directory": when the existing directory sits under the `unknown` splitter and the plan now resolves, `_execute` renames it into `<dataset>/<splitter>/<slug>--<run_id>`. The run keeps one directory, and nothing is deleted.

## Review Triage Log

## Design Notes

The directory carries `dataset` and `splitter`, so the index and the report read the section from the path instead of looking up the plan per run. That also covers runs whose plan file is missing, such as test fixtures with `split_hash: h1`. A run whose plan cannot be resolved still gets a section (`unknown`) and is never dropped.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests` -- expected: clean
- `uv run csi-bench migrate-runs --runs runs && uv run csi-bench migrate-runs --runs runs` -- expected: the second call reports 0 moved (after the tar backup)
- `uv run csi-bench report --runs runs` -- expected: separate `tor_vergata / random_window` and `tor_vergata / time_block` sections plus a leakage block
