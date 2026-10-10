---
title: 'Comparison report and notebook results API'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: 'b24a93258e854a3b1e9f23a616827ec071741981'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Finished runs sit in `runs/<run_id>/` with no way to compare them, so the supervisor cannot get the methods × rooms table and notebooks cannot read results without re-implementing the file layout.

**Approach:** `csi-bench report` refreshes `runs/index.parquet` and prints the comparison table, an optional all-variants list and a feature-selection stability view. A read-only `csi_bench.results_api` exposes the index, metrics and predictions through the shared schema; a notebook shows an example chart.

## Boundaries & Constraints

**Always:** Paths and column schemas live in `core/results_schema.py`; the report and API both use them. A **method** is (sorted extractor names, feature-selection name or `none`, head name); a **variant** is the rest of the resolved config (window_len, stride, preprocessing, params). **Room setting** from `selection.rooms`: `[r1]`→`r1`, `[r2]`→`r2`, both→`mixed`; any other set is listed under its joined name but outside the three columns. Only `status: done` runs enter the table. Each cell = the variant with best mean **val** macro-F1; shows test macro-F1 and accuracy as mean ± std plus `csi-bench run runs/<run_id>/config.yaml`. Dirty runs (`provenance.git.dirty`) are excluded unless `--include-dirty`. `role: reference` rows go in a separate block beside `experiments/published_reference.yaml` (r1 99.41, r2 99.59, mixed 97.37). Stability = per method and room setting, per feature: selected count / (runs × seeds), from `seeds.<seed>.feature_names`; runs without feature selection are omitted.

**Never:** The report and API write nothing except `runs/index.parquet` (report only); `results_api` has no write function and imports only `core` and third-party libraries. No new dependencies (notebook is plain nbformat JSON). No cell mixes split hashes.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Table | done runs across r1/r2/mixed | rows = methods, columns r1/r2/mixed, best-val variant per cell | N/A |
| All variants | `--all-variants` | one line per run: method, room, variant, val and test scores, run id | N/A |
| Mixed splits | runs with different `split_hash` | none printed | `ConfigError`-style message naming the hashes, exit 2 |
| Dirty | run recorded dirty | excluded by default, included with `--include-dirty` | N/A |
| Reference | `role: reference` run | separate block with published value alongside | Missing reference file -> exit 2 |
| No runs | empty/absent `runs/` | message, empty index written | exit 0 |
| Notebook write | `results_api` | read functions only | `test` fails on any other `csi_bench` import in notebooks |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/results_schema.py` -- add `INDEX_TABLE_SCHEMA`, `index_path(root)`; paths for config/status/metrics/provenance/predictions already here.
- `src/csi_bench/results_api/__init__.py` -- empty; add `load_index`, `load_metrics`, `load_predictions` (pandas/pyarrow, read-only).
- `src/csi_bench/runner/cli.py` -- add `report` subcommand next to `run`/`sweep`.
- `src/csi_bench/runner/report.py` -- new: build index from `runs/*/` (config.yaml, status.json, metrics.json, provenance.json), table/variant/stability rendering.
- `src/csi_bench/runner/run.py` -- `_write_json` and `status.json` / `metrics.json` shapes (`mean_std.<split>.{accuracy,macro_f1}=[mean,std]`, `seeds.<n>.feature_names`); do not change.
- `tests/test_run.py` -- `_make_project`, `CONFIG`, `main([...])` fixtures to reuse; `tests/test_results_schema.py`, `tests/test_layering.py` (import-linter contracts in `pyproject.toml`; add `results_api` imports only core).
- `experiments/published_reference.yaml` -- new; `notebooks/` -- new.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/results_schema.py` -- index schema and path -- single source for report and API
- [x] `src/csi_bench/results_api/__init__.py` -- read-only loaders -- notebook entry point
- [x] `src/csi_bench/runner/report.py` -- index refresh, table, `--all-variants`, stability, split-hash and dirty rules, reference block -- the report
- [x] `src/csi_bench/runner/cli.py` -- `report` subcommand (`--all-variants`, `--include-dirty`, `--runs`) -- command surface
- [x] `experiments/published_reference.yaml` -- published r1/r2/mixed values -- reference comparison
- [x] `notebooks/01_classical_overview.ipynb` -- macro-F1 vs window length per classifier via `results_api` only -- example chart
- [x] `pyproject.toml` -- import-linter contract: `results_api` imports only core -- read-only guarantee
- [x] `tests/test_report.py` -- matrix rows (synthetic run dirs), notebook-import test, no-write check

**Acceptance Criteria:**
- Given finished runs, when `csi-bench report` runs, then `runs/index.parquet` is refreshed and the table prints.
- Given a notebook under `notebooks/`, when its code cells are parsed, then the only `csi_bench` import is `csi_bench.results_api`.

## Implementation Notes

- Results-API tests live in `tests/test_report.py` (one file). The report uses `results_api.load_metrics` for stability, so `runner/report.py` imports `results_api`. Index reads `config.yaml`/`status.json`/`provenance.json`/`metrics.json` directly in `runner/report.py`. Stability groups by method × room setting over all comparison runs (variants pooled). Notebook uses matplotlib, which the notebook environment must provide (not added as a dependency). Ruff and ty pass; new tests and layering pass.

## Spec Change Log

## Review Triage Log

### Review Findings

- [x] [Review][Patch] Notebook chart mixes room settings, methods, dirty runs and split hashes — decided: `room = "mixed"` variable, filter `dirty == False` and a single `split_hash`, group by `(method, window_len)`, one line per method [notebooks/01_classical_overview.ipynb]
- [x] [Review][Patch] Done run with unreadable metrics crashes the report — `split_hash` is set before `mean_std` is parsed, so a failed metrics read leaves `None` means on a row that passes `usable`; `_comparison`/`_pm` then raise TypeError. Require non-null `val_macro_f1_mean` in `usable` (or fill row only after all reads succeed) [src/csi_bench/runner/report.py:63-75,198-202]
- [x] [Review][Patch] One malformed run dir aborts the whole report — `config["role"]`, `config["selection"]["rooms"]`, `config["extractors"]` etc. sit outside the `try`; non-dict `status.json`/`config.yaml` raise uncaught TypeError. Move row construction inside the guard, catch TypeError/KeyError, return None [src/csi_bench/runner/report.py:43-62]
- [x] [Review][Patch] "Report writes nothing but index.parquet" is untested — `test_results_api_reads_and_cannot_write` only checks function-name prefixes; add a directory-tree before/after test for `report` [tests/test_report.py]
- [x] [Review][Defer] Sweep `--force`/dirty rerun tests only check the `force` kwarg, not that seeds recompute — deferred: story 2.7 code/tests [tests/test_sweep.py:87-121]
- [x] [Review][Defer] Sweep issues (duplicate run_ids, `_sweep_action` outside per-run try, still-dirty stale skip, stale seed artifacts on force) — deferred: story 2.7 [src/csi_bench/runner/run.py]
- [x] [Review][Defer] Notebook cells never executed by any test — deferred: exploratory output; cheap fix is a test applying the same pandas expressions to the `runs` fixture index [notebooks/01_classical_overview.ipynb]

#### Rejected

- Other room sets (e.g. `r1+r3`) get no table column — low: matches spec ("outside the three columns"); still shown in command and stability lists.
- Regeneration command printed under the table, not in the cell — low: information present, spec-intent met.
- Missing `--reference` ignored when no reference runs — low: nothing to compare, harmless.
- Reference rows not hash-checked — low: reference runs use random split hashes by design.
- Stability denominator counts only seeds with `feature_names`; `" | none | "` string match fragile — low: runner always writes names for FS methods; fix needs new index columns.
- Stability pools window_len/k — false: spec defines stability per method and room setting.
- import-linter contract forbids only three packages — low: those are all other first-party packages; the name overstates, no breakage.
- Non-atomic index write / ArrowException not caught / NaN or tie in best-variant pick — low: sorted directory order is deterministic; NaN and arrow type errors not reachable from validated configs.
- Mixed seed counts compared equally; same-name extractors collapse; `load_metrics` run_id not validated; missing `seeds` key; empty-report newline; `metric` key unused; hardcoded `"runs"`; trailing newline; brittle notebook import test; skipped-run warning — low/negligible, fixes add complexity for unlikely cases.
