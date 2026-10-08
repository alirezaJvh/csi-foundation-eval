---
title: 'Inspect the dataset with csi-bench describe'
type: 'feature'
created: '2026-10-08'
baseline_commit: 'd0d5520aec8dad657c40b00a9dbe7d8724f1fc33'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-1-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** There is no way to check class balance and per-recording packet counts (the guard against sample rate standing in for the activity) before running experiments, and no CLI exists yet.

**Approach:** Add a `csi-bench` console script (`runner/cli.py`) with a `describe <dataset>` subcommand. It loads the dataset through the registry, joins the committed split, prints one row per recording plus totals, and writes the table to `runs/_dataset/<dataset>.parquet` using a schema defined in the new `core/results_schema.py`.

## Boundaries & Constraints

**Always:**
- Entry point `csi-bench = "csi_bench.runner.cli:main"` in `[project.scripts]`; `main(argv=None) -> int` so tests call it directly. Use stdlib `argparse`; only the `describe` subcommand now (`run`/`sweep`/`report` come later).
- Dataset lookup: `discover()` then `registry.get("dataset", name)`, built with default `Params`. Unknown name → exit code 2, stderr message listing `registry.names("dataset")`.
- "Current split" = the single `splits/<dataset>/*.json`, loaded with `load_split_plan` against the loaded recordings. Optional `--split <split_hash>` picks one file; zero files, several files without `--split`, or an unknown hash → `DataError` message, exit code 2.
- Per-recording row: `recording_id`, `room`, `class` (label name via the source's `classes`), `n_packets`, `downsample_factor` (nullable), `train`, `val`, `test` packet counts (block end − start).
- Totals: summed `n_packets`/`train`/`val`/`test` per class, per room and overall.
- Parquet: one table, columns `scope` (`recording|class|room|total`), `recording_id` (null for totals), `room`, `class` (null when not applicable), `n_packets`, `downsample_factor`, `train`, `val`, `test`, `split_hash`. Column names and dtypes are defined once in `core/results_schema.py` (also the `runs/_dataset/<dataset>.parquet` path helper); the printed text shows the same rows.
- Rows sorted by `recording_id`, so output is deterministic. `runs/` is created if absent.

**Never:**
- No config, windower, `run`/`sweep`/`report` code; do not modify `data/`, `splits/`, or any split logic; no `git add`/commit.
- `core/` must not import adapters or runner; the CLI must not import adapter modules directly (only via `discover`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Real data | `describe tor_vergata` | 19 recording rows + class/room/total rows printed; parquet written; exit 0 | N/A |
| Totals | train+val+test per recording ≤ `n_packets` | class/room/total rows equal sums of recording rows | N/A |
| Unknown dataset | `describe nope` | nothing written | exit 2, message lists `tor_vergata` |
| No split file | `splits/<dataset>/` empty | nothing written | exit 2, `DataError` text |
| Several split files | two plans, no `--split` | nothing written | exit 2, message lists hashes |
| Stale split | recording sha differs from plan | nothing written | exit 2, `DataError` from loader |
| Rerun | same inputs twice | identical parquet content | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/discover.py` -- `discover()` imports adapters so they self-register; call before lookup
- `src/csi_bench/core/registry.py` -- `get("dataset", name)`, `names("dataset")`; `ConfigError` on unknown
- `src/csi_bench/core/splitplan.py` -- `load_split_plan(path, recordings)`; blocks are `(start, end)` per part
- `src/csi_bench/core/contracts.py` -- `Recording`/`RecordingMeta` (`n_packets`, `downsample_factor`); `ports.DatasetSource.classes`
- `src/csi_bench/adapters/datasets/tor_vergata.py` -- source; default root is CWD-relative (see `deferred-work.md`), so run from repo root; do not change
- new `src/csi_bench/core/results_schema.py` -- dataset-table columns/dtypes + path helper; pandas/pyarrow are allowed in core
- new `src/csi_bench/runner/cli.py` -- argparse `main`, table building, printing, parquet writing
- `pyproject.toml` -- add `[project.scripts]`; import-linter contracts already allow runner → adapters
- `tests/test_tor_vergata.py` -- registry-reset fixture style to copy; `tests/test_time_block.py` -- builds synthetic recordings and plans

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/results_schema.py` -- dataset-table schema and `dataset_table_path(dataset, root="runs")`
- [x] `src/csi_bench/runner/cli.py` -- `main`, `describe` (table from recordings + plan, text output, parquet write), error → exit 2
- [x] `pyproject.toml` -- `[project.scripts] csi-bench`
- [x] `tests/test_describe.py` -- matrix cases on synthetic recordings in `tmp_path` (monkeypatched cwd and dataset root), parquet schema and round trip, rerun identical, real-data run (skip if data absent)

**Acceptance Criteria:**
- Given the installed project and real data, when I run `uv run csi-bench describe tor_vergata`, then stdout has 19 recording rows and class, room and total rows, and `runs/_dataset/tor_vergata.parquet` holds the same table with the schema from `core/results_schema.py`.
- Given an unknown dataset name, when I run `describe`, then the exit code is non-zero and the message lists the registered dataset names.
- `ruff check`, `ruff format --check`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench describe tor_vergata` -- expected: table printed, parquet written, exit 0

### Review Findings

- [x] [Review][Patch] Scope values are hardcoded in `build_table` while `SCOPES` in the schema module is dead code — the one-place definition the spec asks for is not actually used [src/csi_bench/runner/cli.py:48]
- [x] [Review][Patch] CLI error boundary catches only `DataError`; `discover()` import failures, adapter `Params`/`ConfigError`, and `runs/` mkdir/parquet-write `OSError` escape as tracebacks (exit 1, after the table is already printed) instead of a clean exit 2 [src/csi_bench/runner/cli.py:98-110]
- [x] [Review][Patch] The full dataset is loaded before the split is located, so a missing/ambiguous split only surfaces after the expensive load — validate `_find_split` first [src/csi_bench/runner/cli.py:101-102]
- [x] [Review][Patch] The `csi-bench` console-script entry point is never exercised by any test; a wrong `[project.scripts]` target ships green [pyproject.toml:21]
- [x] [Review][Patch] `build_table`'s absent-class branch (`if c in present`) has no test coverage; a regression there would emit spurious all-zero class rows undetected [src/csi_bench/runner/cli.py:73]
- [x] [Review][Defer] Repo-wide `ruff check` / `ruff format --check` fail (181 errors, all under untracked `.agents/skills/**`), so AC3's verification command does not pass in this workspace [pyproject.toml:52] — deferred: pre-existing/environmental, not caused by this change

#### Rejected

- `false` — "Totals sum raw `n_packets` across recordings with different `downsample_factor`" (blind-hunter): the spec's Totals row mandates exactly these raw sums; not a defect.
- `false` — "Only `split_hash`, not `splitter`/`params`/`gap_packets`, is emitted" (blind-hunter): the spec fixes the full column set; adding columns would violate it.
- `false` — "Output path is hardcoded, no `--out` option" (blind-hunter): the spec fixes `runs/_dataset/<dataset>.parquet`.
- `false` — "`classes[r.label]` has no bounds validation" (blind-hunter): labels are produced by the adapter as `CLASSES.index(...)`, so an out-of-range label is unreachable with the registered sources.
- `false` — "No `--version`/`--verbose`/logging hook" (blind-hunter): out of scope per spec ("only the `describe` subcommand now").
- `false` — "`_find_split` does not check the file stem equals the plan's internal `split_hash`" (blind-hunter): `write_split_plan` names files by hash and rows report the internal `split_hash`, so no wrong output results.
- `low` — non-atomic parquet write (blind-hunter): crash-mid-write is rare and a rerun overwrites; fix adds temp-file + `os.replace` complexity.
- `low` — no confirmation of the written path (blind-hunter): cosmetic omission, not in spec.
- `low` — `test_rerun_identical` compares raw parquet bytes (blind-hunter): byte-identity is a valid determinism check; only a hypothetical cross-pyarrow-version concern.
- `low` — assorted coverage gaps (column order, null totals cell, `runs/` creation, argparse-exit) (blind-hunter): low-value; overlapping the two filed verification gaps.
- `low` — left-justified numeric columns / no dataset-split header (blind-hunter): cosmetic formatting.
- `low` — "monkeypatched cwd and dataset root" only partially implemented (acceptance-auditor): a Tasks-note detail, not an AC; behavior is correct.
