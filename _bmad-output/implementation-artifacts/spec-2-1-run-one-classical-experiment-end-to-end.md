---
title: 'Run one classical experiment end-to-end'
type: 'feature'
created: '2026-10-08'
status: 'done'
route: 'dispatch'
baseline_commit: '1b9c759ddc853393b9176343c91b80c3f2bd1e82'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-1-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The CLI can only `describe`; nothing turns a config into windows, features, a fitted classifier and stored metrics, so no result exists yet.

**Approach:** Add `csi-bench run <config.yaml>`: a pydantic `ExperimentConfig` is validated first, then a thin runner executes Windower → `amplitude_stats` extractor → `logreg` head → Evaluator for one seed and writes the AD-13 run directory. Everything is the narrowest slice that works (one extractor, one head, no preprocessing chain, cache or feature selection).

## Boundaries & Constraints

**Always:**
- `ExperimentConfig` (pydantic, `extra="forbid"`) in `config/`: `dataset`, `split` (split hash or null = the single committed plan), `selection.rooms`, `window_len`, `stride` (default 1), `extractors: [{name, params}]`, `head: {name, params}`, `seeds` (default `[0]`), `role ∈ {comparison, reference}` (required). Invalid fields raise `ConfigError` naming the field path, before any data is loaded. Adapter names and `Params` are validated via the registry.
- Ports added to `core/ports.py`: `FeatureExtractor`, `Head`. The Windower (`core/windows.py`, owns window boundaries, AD-7) and Evaluator (`core/metrics.py`) are pure core functions, not adapters. Adapters `adapters/extractors/amplitude_stats.py` and `adapters/heads/logreg.py` register with `Params` and int `VERSION`.
- Windower: windows lie fully inside one block of one split part, `window_id = "<recording_id>@<start>:<end>"`, ordered by `(recording_id, start)`, only selected-room recordings; `window_len > plan.gap_packets` → `ConfigError`.
- `amplitude_stats` (`granularity: per_window`, `requires: {phase: false, n_antennas: any}`): gets only the window's `[start, end)` packet slice; columns `amplitude_stats.mean_power`, `.std`, `.flatness` over |csi| (power = |csi|², flatness = geometric/arithmetic mean of power, std of amplitude), each pooled over antennas, packets and subcarriers; float32; a batch and a single window give identical values. Runner raises `ExtractorError` on non-finite values or constant columns.
- `logreg` head: `fit(train, val)` tunes `C` over a fixed grid on val macro-F1, refits on train ∪ val, standardises features inside the head, predicts test once. `seed` from config drives `random_state`.
- Evaluator: accuracy, macro-F1, per-class precision/recall/F1 keyed by class name, confusion matrix, for val and test.
- Run dir `runs/<run_id>/{config.yaml, status.json, metrics.json, seeds/<seed>/predictions.parquet, provenance.json, run.log}`; paths and schemas live only in `core/results_schema.py`. `run_id = short_hash` of the resolved config (excluding `seeds`) plus adapter `VERSION`s and excluding `affects_result=False` params. `status.json`: `pending → running → done | failed` (with error trace). Provenance: git commit, dirty flag (+ diff when dirty), package versions, `split_hash`, duration.
- `csi-bench run runs/<run_id>/config.yaml` yields the same `run_id`. Errors map to exit code 2 with a clean stderr message.

**Never:**
- No preprocessing chain, feature cache, feature selection, pooler, multiple seeds loops beyond the list, sweeps, `report`, or other classifiers/extractors (stories 2.2-2.8). No `random_window` splitter or `role: reference` behaviour. No changes to `data/`, `splits/` or split logic. `core/` imports nothing internal; only `runner` imports adapters (via `discover`). No `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | tor_vergata, rooms `[r1]`, W=40, `amplitude_stats`, `logreg`, comparison | run dir complete, `status=done`, metrics for val and test | N/A |
| Bad field | `window_len: "x"` or unknown key | nothing loaded | `ConfigError` naming the field, exit 2 |
| Window too long | `window_len` > `gap_packets` | no run dir `done` | `ConfigError`, exit 2 |
| Unknown adapter | `head: {name: nope}` | nothing run | `ConfigError` listing registered names |
| Window isolation | any window | start/end inside one block of one part | N/A |
| Rerun from saved config | `runs/<id>/config.yaml` | same `run_id` | N/A |
| Failure mid-run | extractor raises `ExtractorError` | `status=failed` with trace | exit 2 |

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/cli.py` -- argparse `main`, `describe`; add `run` subcommand, same error → exit 2 pattern
- `src/csi_bench/runner/discover.py` -- `discover()` before registry lookups
- `src/csi_bench/core/registry.py` -- `register/get/names`; `Params` + `VERSION` required
- `src/csi_bench/core/{contracts,ports,hashing,errors,splitplan}.py` -- add `WindowSet`, `FeatureMatrix`, `Predictions`/`Metrics`, new ports; reuse `short_hash`, `load_split_plan`
- `src/csi_bench/core/results_schema.py` -- add run-dir paths and parquet/JSON schemas next to the dataset table
- `src/csi_bench/adapters/datasets/tor_vergata.py` -- source (classes, `load()`); default root is CWD-relative, run from repo root; do not change
- `src/csi_bench/adapters/splitters/time_block.py`, `splits/tor_vergata/e31e1b7dd9b4.json` -- committed plan, `gap_packets=64`
- `tests/contracts/dataset_source.py`, `tests/contracts/splitter.py` -- pattern for new extractor/head contract tests
- `tests/test_describe.py` -- registry-reset fixture and synthetic recordings style to copy
- `pyproject.toml` -- import-linter contracts already cover the new packages; no new dependencies
- new: `config/experiment.py`, `core/windows.py`, `core/metrics.py`, `adapters/extractors/amplitude_stats.py`, `adapters/heads/logreg.py`, `runner/run.py`

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/contracts.py`, `ports.py` -- add `WindowSet`, `FeatureMatrix`, `Predictions`, `Metrics` and `FeatureExtractor`/`Head` ports
- [x] `src/csi_bench/core/windows.py`, `core/metrics.py` -- Windower function and Evaluator functions
- [x] `src/csi_bench/config/experiment.py` -- `ExperimentConfig`, YAML loader, `run_id` computation
- [x] `src/csi_bench/adapters/extractors/amplitude_stats.py`, `adapters/heads/logreg.py` -- the two adapters
- [x] `src/csi_bench/core/results_schema.py` -- run-dir layout and schemas
- [x] `src/csi_bench/runner/run.py`, `runner/cli.py` -- orchestration, provenance, status, `run` subcommand
- [x] `experiments/r1_amplitude_stats_logreg.yaml` -- the AC config
- [x] `tests/` -- config validation, windower, extractor batch-vs-single contract, head, metrics, end-to-end on synthetic recordings, `run_id` round trip, real-data run (skip if data absent), registered-extractor/head contract tests

**Acceptance Criteria:**
- Given a valid YAML config, when I run `uv run csi-bench run <config.yaml>`, then the run directory holds all six artefacts in the `results_schema` formats and `status.json` is `done`.
- Given the saved `runs/<run_id>/config.yaml`, when I run it again, then the `run_id` is identical (test).
- Given an invalid field, when I run, then `ConfigError` names that field within seconds, before any recording is loaded.
- `ruff check src tests`, `ruff format --check src tests`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Design Notes

Config sketch (the YAML shape the tests and the AC use):

```yaml
dataset: tor_vergata
selection: {rooms: [r1]}
window_len: 40
stride: 1
extractors: [{name: amplitude_stats}]
head: {name: logreg}
seeds: [0]
role: comparison
```

The Windower is a core function (not an adapter) because AD-7 gives window boundaries to one owner; `WindowSet` is the only thing extractors and heads see.

## Verification

**Commands:**
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench run experiments/r1_amplitude_stats_logreg.yaml` -- expected: `runs/<run_id>/` complete, exit 0

### Review Findings

- [x] [Review][Patch] Refit the head's scaler on train ∪ val for the final model (keep the train-only scaler for the C search) and pin it with a test that `_scaler.mean_` equals the train ∪ val mean — decided by user [src/csi_bench/adapters/heads/logreg.py:42-52]
- [x] [Review][Patch] Config validation is too weak and partly late: `window_len`/`stride` unbounded, `extractors`/`seeds`/`selection.rooms` may be empty or duplicated, `dataset` is not checked against the registry (failure leaves a `failed` run dir), adapter `Params` accept unknown keys (typo `C_grd` silently ignored), `C_grid` may be empty. Add `Field(ge=1/min_length)`, `extra="forbid"` on Params, dataset check in `validate_registry`; after load, raise `ConfigError` for unknown rooms / zero windows / an empty train, val or test part [src/csi_bench/config/experiment.py:35-45, src/csi_bench/runner/run.py:_execute]
- [x] [Review][Patch] `run_id` omits the dataset adapter `VERSION` and hashes `split: null` as null, so swapping the committed plan reuses the same `run_id` for different results; include dataset VERSION and the resolved split hash [src/csi_bench/config/experiment.py:resolved_body]
- [x] [Review][Patch] No tests for `metrics_document`/`_mean_std` (multi-seed), `collect_provenance` (commit, dirty, diff via monkeypatched `_git`), or that `load()` is never called for an invalid config [tests/test_run.py]
- [x] [Review][Patch] `status.json` `started_at` is rewritten with a fresh `_now()` at every transition (including `done`); `PARTS` duplicates `core.splitplan.PARTS`; `logger` unused; `_find_split` error tells `run` users to use `--split` which `run` lacks [src/csi_bench/runner/run.py]
- [x] [Review][Defer] Rerun semantics: a rerun overwrites a `done` run's status/metrics, leaves stale `seeds/<n>/`, has no lock, and `KeyboardInterrupt` leaves `status=running` — deferred: AD-14 resumability/`--force` belongs to story 2.7
- [x] [Review][Defer] `_validate_extractors` (`min_window_len`, `n_antennas`) branches are untested — deferred: unreachable until a second extractor lands

#### Rejected

- `low` — constant-column/finite check spans val and test (blind, auditor): spec/AD-8 mandate the check without scope; it only fails runs, never feeds the model.
- `low` — provenance `git status` counts untracked files, diff omits them, CWD-dependent, no adapter versions, only written on success (blind, edge): `runs/` and `.cache/` are git-ignored; remaining points are minor and the fix adds branches.
- `low` — `results_schema.py` lacks status/provenance schemas, `STATUS_STATES` unused (auditor): developer-only; moving payload builders is more than a direct correction.
- `low` — `Head.fit(train, y_train, val, y_val)` differs from `fit(train, val)` (auditor): labels must come from somewhere; reasonable.
- `low` — rooms order changes hash, population std, `assert` in `predict`, feature slices held in memory, duplicate feature names, flatness `-inf` on zero power, single-class train error text (blind, edge): negligible at current scope; later stories add batching.
- `false` — `plan.blocks[rec.recording_id]` can raise `KeyError` (edge): `load_split_plan` already requires blocks and recordings to match.
- `false` — `StopIteration` in `metrics_document` (edge): needs empty `seeds`, which patch 1 forbids.
- `false` — `provenance.dirty` always true because `runs/` is untracked (auditor): `runs/` is in `.gitignore`.
- Spec/sprint bookkeeping (`last_updated` moved backwards, `baseline_commit`, empty log sections, `window_len = 1` wording): fix would edit the spec or tracking files, not code.
