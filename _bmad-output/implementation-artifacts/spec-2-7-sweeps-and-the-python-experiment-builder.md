---
title: 'Sweeps and the Python experiment builder'
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

**Problem:** Each comparison cell needs its own hand-written config and a manual `run`, so a large comparison cannot be queued, resumed after a crash, or built from a notebook.

**Approach:** `csi-bench sweep <file>` expands a base config plus a `sweep:` block of dotted paths into a Cartesian product, validates all of it up front, then runs each config in-process, one at a time, with skip/rerun/failure rules. `ExperimentBuilder` builds the same `ExperimentConfig` the YAML loader does. Ship `experiments/classical_baseline.yaml`.

## Boundaries & Constraints

**Always:** Dotted paths address the real config shape (`window_len`, `selection.rooms`, `head.name`, `extractors`, `feature_selection`, `preprocessing`). Config construction has one path: dict -> `ExperimentConfig.model_validate` -> `validate_registry`. Sweep expansion order is file order (first key slowest). Combinations resolving to the same `run_id` run once. A sweep run reuses `_execute`, so seeds, cache and per-seed resume behave as in `run`.

**Never:** Sweep `seeds` (path `seeds` or `seeds.*`) -> `ConfigError`. No pipeline objects or registry discovery in the builder. No parallel runs. No new dependencies. Do not touch the report/results API (2.8).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Expand | `window_len: [10,30]`, `head.name: [svm,knn]` | 4 configs, all validated before any run | Any invalid combo -> `ConfigError` naming its overrides; nothing runs, exit 2 |
| Bad sweep | `seeds` swept; empty value list; non-mapping `sweep`; missing base field | rejected before any run | `ConfigError` |
| Resume | status `done`, provenance `dirty: false` | skipped | N/A |
| Dirty rerun | `done` with `dirty: true`, tree now clean | rerun | Tree still dirty -> skipped |
| Failure | one run raises | `status.json` `failed` with trace; sweep continues; exit 1 after the last run | Summary lists failed run ids |
| Force | `--force` | every run re-executed, previous seeds ignored | N/A |
| Builder | `ExperimentBuilder().dataset("tor_vergata")...build()` | equals YAML-loaded config | Missing required field -> `ConfigError` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/config/experiment.py` -- `load_config` parses YAML then `model_validate` + `validate_registry`; extract `config_from_dict(raw, source)` for reuse by sweep and builder. `_expand_*` shorthand validators already normalize `feature_selection`/`preprocessing`.
- `src/csi_bench/runner/run.py` -- `_execute(config, config_path, run_id, root, cache_root)` writes status (`failed` + `trace`) then re-raises; `_completed_seeds` gates skipping; `collect_provenance` records `git.dirty`; `_git` helper. Add `force` to `_execute` (ignore `previous`).
- `src/csi_bench/runner/cli.py` -- `main` argparse; add `sweep` subcommand (`file`, `--force`).
- `src/csi_bench/core/results_schema.py` -- `status_path`, `provenance_path` for the skip decision.
- `experiments/r1_amplitude_stats_logreg.yaml` -- config style to mirror; `config/presets.py` for preprocessing preset names (`ant2`, ...).
- `tests/test_run.py` -- `_make_project`, `CONFIG`, `_config_file`, `main([...])` fixtures to reuse; `tests/test_experiment_config.py` for config tests.
- Layering (`pyproject.toml` import-linter): `config` imports only `core`; only `runner` imports adapters.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/config/experiment.py` -- extract `config_from_dict` -- single construction path
- [x] `src/csi_bench/config/sweep.py` -- `expand_sweep(raw, source) -> list[tuple[dict overrides, ExperimentConfig]]`: split `sweep`, reject `seeds`, set dotted paths (dict keys, list indices; create missing mappings), Cartesian product, validate each, dedupe by `compute_run_id`
- [x] `src/csi_bench/config/builder.py` -- `ExperimentBuilder` fluent setters (`dataset`, `rooms`, `preprocessing`, `window`, `features`, `feature_selection`, `head`, `seeds`, `role`), `build()` via `config_from_dict`; export from `csi_bench.config`
- [x] `src/csi_bench/runner/run.py` -- `force` flag on `_execute`; `run_sweep(path, force, root, cache_root) -> int` (discover, expand, skip rule, catch per-run exceptions, print one `run_id status` line each plus summary; exit 0 / 1 if any failed / 2 on config error)
- [x] `src/csi_bench/runner/cli.py` -- `sweep` subcommand
- [x] `experiments/classical_baseline.yaml` -- window_len [10,30,40,50] x extractor sets (`amplitude_stats`; `paper_beamforming` with `ant2`; both) x `feature_selection` [null, mrmr] x head [svm,knn,random_forest] x rooms [[r1],[r2],[r1,r2]]; every combination must validate (respect `min_window_len`, mrmr `k <= columns`)
- [x] `tests/test_sweep.py`, `tests/test_builder.py` -- I/O matrix rows, expansion count/order, builder == YAML config, example file expands without error

**Acceptance Criteria:**
- Given a sweep with one invalid combination, when `csi-bench sweep` runs, then no run directory is created and the exit code is 2
- Given an interrupted sweep relaunched, when all earlier runs are `done`, then none re-executes (spy on `_execute`)
- Given `runs/` already holds the same configs, when `--force` is passed, then every run re-executes
- Given the builder, when it builds, then no registry lookup, dataset or splitter is touched

## Implementation Notes

- Baseline `window_len` starts at 20 (not 10): `paper_beamforming` needs >= 18 packets. `preprocessing: [ant2]` applies to all cells because `paper_beamforming` needs 2 antennas. MRMR `k: [2, 3]`.
- `build()` runs `validate_registry` (single construction path), so adapters must already be discovered; it touches no dataset or splitter.
- Dirty `done` runs are skipped when git is unavailable or the tree is still dirty.
- Verified: pytest 256 passed, ruff, ty and lint-imports clean.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check . && uv run ruff format --check .` -- expected: clean
- `uv run ty check` -- expected: clean
- `uv run lint-imports` -- expected: contracts kept

### Review Findings (2026-10-10)

Patch:

- [x] [Review][Patch] List-index dotted paths and missing-mapping creation in `_set_path` have zero test coverage — every exercised sweep key traverses only dict keys (`tests/test_sweep.py`); an `extractors.0.name` / `selection.rooms.0` path, an out-of-range index, and a missing-mapping creation (e.g. `feature_selection.mrmr` with base null) are all reachable via `sweep.py:24-38` and a regression would keep CI green [`src/csi_bench/config/sweep.py:24-38`, `tests/test_sweep.py`]
- [x] [Review][Patch] Sweep path with a non-ASCII "digit" part (e.g. `preprocessing.²` into a list) passes `isdigit()` but crashes `int()` with a raw `ValueError` traceback and exit 1 instead of the designed `ConfigError` + exit 2 [`src/csi_bench/config/sweep.py:41-44`]
- [x] [Review][Patch] Builder `features(('a',))` / malformed tuple raises a raw `ValueError`/`TypeError` from `name, params = entry` instead of a `ConfigError`, unlike every other construction-path error [`src/csi_bench/config/builder.py:12-15`]
- [x] [Review][Patch] `_sweep_action`'s git-unavailable fallback (`skip` when `_git` returns `None`) is never tested — an inversion would rerun dirty recorded runs in git-less environments without a failing test [`src/csi_bench/runner/run.py:520-521`]
- [x] [Review][Patch] Sweep summary counts skipped cells as ok: `f"{len(planned) - len(failed)} ... runs ok"` makes a no-op rerun print "N of N runs ok" with no executed-work distinction [`src/csi_bench/runner/run.py:563`]

Rejected:

- [Review][Patch→false] Builder AC "no registry lookup" is violated by `build()` running `validate_registry` — the fix edits the spec under review: the AC text contradicts the frozen single-construction-path boundary and the spec's own Implementation Notes; spec text must be reconciled by the author, not patched here.
- [Review][Patch→false] Task line records `window_len [10,30,40,50]` but the shipped example uses `[20,30,40,50]` — fix edits the spec under review; the deviation is correctly documented in Implementation Notes and covered by `test_example_sweep_expands`.
- [Review][Patch→false] `run_sweep` passes the sweep file path as `config_path` to `_execute`, so per-cell logs misattribute the loaded config — log-text-only cosmetic; the per-cell config YAML in the run directory remains correct provenance.
- [Review][Patch→false] A fully `done` run with unreadable provenance prints `done` instead of `skipped` — rare edge, run label cosmetic; the fallback to `run` for unverifiable provenance is the documented conservative behavior.
- [Review][Patch→false] Dedupe of equal `run_id` combos is silent — the summary already reveals the post-dedupe count before any run, and the spec demands no dedupe message.
- [Review][Patch→false] Data-dependent constraints (antenna count, known rooms) surface per-cell inside the loop, so an infeasible sweep costs full per-cell dataset loads — config-level preflight already rejects invalid combos; data-dependent checks cannot be prefilled, per-cell failure with a labeled line is the designed behavior.
- [Review][Patch→false] Two sweep paths touching the same subtree resolve silently (last wins) — requires contradictory keys on one combo; the spec leaves layering undefined and a guard would add untested complexity.
- [Review][Patch→false] `--force` on a partially failed run may overwrite a partial metrics set — identical to single-run `--force` semantics and covered by the spec's "previous seeds ignored" row.
- [Review][Patch→false] Forced rerun cells print `done` indistinguishable from plain runs — cosmetic wording; `skipped` lines already convey what did not run.
- [Review][Patch→false] Preset-name cross-reference drift in the Code Map — spec edit under review.
- [Review][Patch→false] `fresh_registry` not applied between `test_sweep` tests — false: it is an `autouse` fixture imported from `tests/test_run.py:41` (the `noqa: F401` re-export pattern); pytest applies imported autouse fixtures module-wide.
- [Review][Patch→false] `"run"` action restarts a previously failed run's whole seed loop rather than resuming seeds — identical to `_execute`'s standalone behavior and to the frozen boundary "behave as in `run`".
- [Review][Patch→false] `_sweep_action` can abort an OSError outside the loop's try — false: provenance read catches OSError at `run.py:516`, `_completed_seeds` catches OSError at `run.py:111`, and the loop's blanket `except Exception` at `run.py:557` bounds everything else.
- [Review][Patch→false] `bool('false')` from string-wrapped provenance flips `rerun` to `skip` — false: provenance is written by the toolkit itself as JSON bools; a corrupted value is caught by the same except as unreadable provenance (conservative `run`).
- [Review][Patch→false] Commit substitution under clean tree + dirty provenance — speculative without a demonstrated trigger; `rerun` on any clean-but-once-dirty record is the spec'd conservative rule.
- [Review][Patch→false] Example YAML's mRMR-k justification lacks a "capped by narrowest cell" caveat — comment-only polish on a shipped example.
