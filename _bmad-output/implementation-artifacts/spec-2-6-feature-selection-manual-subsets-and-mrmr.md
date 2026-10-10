---
title: 'Feature selection (manual subsets and MRMR)'
type: 'feature'
created: '2026-10-09'
status: 'done'
baseline_commit: '14644feb353930f0b3738f9c0256ec00b5932e9b'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every run feeds the head all feature columns, so the researcher cannot test which features matter or how stable an automatic choice is.

**Approach:** Add a `feature_selection` config field and a `FeatureSelector` port with adapters `manual` (column names or whole families) and `mrmr` (`mrmr-selection` 0.2.8, `mrmr_classif`). The runner applies the selector between the feature matrix and the head, per seed; MRMR is fitted on train and k is chosen on val together with the head's hyperparameters; the selected `feature_names` are recorded per seed in `metrics.json`.

## Boundaries & Constraints

**Always:** `feature_selection: null` (default) uses all columns and leaves existing run ids and cache keys unchanged. A selector only ever receives train and val rows; test columns are filtered with the fitted choice. Selection only picks columns: the feature cache is untouched. `feature_selection` enters the run id (name, version, params).

**Never:** No recomputation of features per subset. No wrapper selectors. No report/stability output (story 2.8). No change to head hyperparameter grids.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Manual names and families | `{manual: [amplitude_stats.mean_power, paper_beamforming]}` | head sees exactly those columns, in matrix column order | N/A |
| Unknown manual entry | entry matches no column or family | N/A | `ConfigError` naming the entry and the available columns |
| MRMR | `{mrmr: {k: [2, 3, 5]}}` | per k: rank on train, head tuned on train/val; k with best val macro-F1 wins (first on ties); test filtered with it | N/A |
| k too large | k > number of columns | N/A | `ConfigError` before any seed runs |
| Null | `feature_selection: null` or absent | all columns; `metrics.json` records all names | N/A |
| Bad shape | not exactly one of `manual`/`mrmr`, or bad params | N/A | `ConfigError` at load, before any run |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/ports.py` -- add `FeatureSelector` Protocol (`VERSION`, `fit(train, y_train, val, y_val)`, `transform(matrix)`, `selected_names`); add `val_score: float` to `Head`.
- `src/csi_bench/core/head_search.py` -- `tune_and_refit` returns the best val macro-F1 as a fourth value; 5 heads in `src/csi_bench/adapters/heads/*.py` store it as `self.val_score` (the existing `_` slot).
- `src/csi_bench/adapters/selectors/{__init__,manual,mrmr}.py` -- new, registered as stage `selector`; discovered by `runner/discover.py` automatically. `mrmr` fits `mrmr_classif(X=DataFrame(train), y=Series, K=max(k))`; the greedy order makes top-k a prefix, so one ranking serves all k. Reference use: `data/csi_proc_python/classification.py:69`.
- `src/csi_bench/config/experiment.py` -- `ExperimentConfig.feature_selection`; normalizes `{manual: [...]}` / `{mrmr: {...}}` / `{name, params}` (saved configs reload) into `AdapterSpec | None`; `validate_registry` checks it; `resolved_body` adds it (name, version, affecting params) only when set.
- `src/csi_bench/runner/run.py` -- `_run_seed` (l.~245) builds the selector, loops candidate subsets, fits the head, picks by `head.val_score`, records names; `_execute` resolves manual entries against `matrix.feature_names` before the seed loop; `metrics_document` call gets the names.
- `src/csi_bench/core/results_schema.py` -- `metrics_document` writes `seeds.<seed>.feature_names`; `_completed_seeds` resume path must keep it (stored seed records are re-emitted verbatim).
- `tests/contracts/selector.py` -- new contract helper; `tests/test_run.py` has `_make_project`/`_config_file` fixtures and the two-family test at l.~566 to copy.
- Do not change: `adapters/storage/feature_cache.py`, `feature_cache_key`.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/ports.py`, `core/head_search.py`, `adapters/heads/*.py` -- expose `val_score` from the head -- lets the runner pick k on the head's own val tuning, not on a refit that saw val
- [x] `src/csi_bench/adapters/selectors/` -- `manual` (Params `columns: list[str]`, min 1) and `mrmr` (Params `k: list[PositiveInt]`, min 1, unique) -- the two selector adapters
- [x] `src/csi_bench/config/experiment.py` -- field, shorthand normalization, registry validation, run-id body -- config is the single path
- [x] `src/csi_bench/runner/run.py`, `core/results_schema.py` -- apply selection per seed and record `feature_names` -- AD-18
- [x] `tests/contracts/selector.py`, `tests/test_selectors.py`, `tests/test_run.py`, `tests/test_experiment_config.py` -- contract + I/O matrix + run-id/cache tests

**Acceptance Criteria:**
- Given a spy selector registered in a test, when a run executes, then `fit` receives only train and val rows and `transform` is the only call that touches test rows (contract test).
- Given manual selection, when the run executes, then the head's train/val/test matrices have exactly the chosen columns and `metrics.json` lists them for every seed.
- Given `mrmr: {k: [2, 3]}`, when the run executes, then `feature_names` has the winning size and is a prefix of one train-only ranking; test labels never enter the ranking or the choice.
- Given the same config with and without `feature_selection`, then run ids differ; with `null` the id equals the id of a config that omits the field.
- Given a head-only or selector-only change, then the feature cache is hit (no re-extraction).

## Implementation Notes

## Spec Change Log

## Review Triage Log

### Review Findings

_Code review (2026-10-10) of the uncommitted story-2.6 code-only change vs baseline `14644fe` (spec's `baseline_commit`). Layers: Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor._

**Decision needed**
- [x] [Review][Decision] Per-candidate selector val scores are discarded — `_run_seed` compares `head.val_score` across candidates to pick `best_index`, but only the winner survives to `metrics.json`; the per-candidate k/val-macro-F1 mapping that decided the selection is never recorded, so the choice is less auditable than the head's own grid. Fixing it means extending the `metrics.json` seed schema (new public surface) or deferring to the story-2.8 report work. — resolved: record per-candidate `n_features`/`val_score` and `chosen` as `seeds.<seed>.selection` [src/csi_bench/runner/run.py]

**Patch**
- [x] [Review][Patch] `_expand_feature_selection` error path can raise `TypeError` instead of `ConfigError` for a mapping with mixed-type keys (`sorted(raw)` on `{manual: [...], 2: ...}`) [src/csi_bench/config/experiment.py:77]
- [x] [Review][Patch] Zero selector candidates reaches `assert best_head is not None` instead of a `ConfigError` (only reachable via a third-party selector) [src/csi_bench/runner/run.py:297-311]
- [x] [Review][Patch] A registered head without `val_score` (pre-port, third-party) crashes with `AttributeError` at the candidate comparison; no adapter check pins the port attribute [src/csi_bench/runner/run.py:309]
- [x] [Review][Patch] No end-to-end test exercises a head other than LogReg together with `feature_selection` (head×selector interaction covered only via LogReg spies) [tests/test_run.py:417]
- [x] [Review][Patch] `tune_and_refit` docstring sentence is mangled ("... and the chosen candidate index and its val macro-F1", missing comma) [src/csi_bench/core/head_search.py:28]

**Rejected**
- `false` — `transform` before `fit` silently returns a 0-column matrix (both selectors, `core/columns.py`): no in-tree caller can reach it — the runner always `fit`s (which `validate`s) before `transform`; guarding it would add complexity nobody's path needs.
- `false` — `take_columns` raises raw `KeyError`/passes duplicates for user-supplied names: the boundary users hit is `Manual.validate`, which raises a friendly `ConfigError` naming the entry and available columns; only a third-party selector bypassing `validate` could reach `take_columns` directly (`core/columns.py`).
- `false` — Resumed seeds' `feature_names` trusted without cross-checking current columns (`run.py` `seed_names`): the same `run_id` already pins extractor config/version/vendor and selector params, so matrix columns cannot differ for a legitimate resume; reconcile would duplicate the run-id guarantee.
- `false` — Port text vs `Mrmr` docstring disagree on val usage: the port says val "may be used to validate" (not that it must), `Mrmr` documents train-only ranking in its module docstring, and `test_mrmr_ranking_ignores_val_rows` pins the behavior.
- `false` — Explicit `{name, ...}` form surfaces generic failures (`{"name": "manual", "junk": 1}` / `{"name": "mrmr"}`): `AdapterSpec` forbids extras (pydantic error at load) and `_check_adapter` → `_validate_params` raises a path-prefixed `ConfigError` (`feature_selection.params: ...`), symmetric with the shorthand errors.
- `false` — `mrmr_classif` returning fewer names than `max(k)` goes silent: `validate` enforces `max(k) ≤ len(feature_names)` before `fit`, and `mrmr_classif` with `K ≤ n_columns` returns exactly `K` features.
- `false` — Single-class `y_train` crashes `mrmr_classif` unguarded: no demonstrated path reaches a single-class train split while every other stage (splits, heads, macro-F1) still works; a degenerate-data failure is the harness-level behavior, not a selector defect.
- `false` — `choose` with an out-of-range/negative index misbehaves (`mrmr.py`): the runner only passes indices from `range(len(candidates))`; out-of-range fails loudly; guarding adds complexity for a state no in-tree caller reaches.
- `false` — Selector `transform(val)` recomputed identically inside the loop ("quadratic rework", `run.py`): each candidate has a different subset, so per-candidate transforms are required, not redundant; the only true duplicate is one final re-transform of the winning val matrix.
- `false` — Diff file ordering looked inconsistent: an artifact of how the code-only diff was assembled for this review, not a repo property.
- `false` — Unrelated test changes bundled silently into the 2.6 diff: they are the already-agreed 2.5 second-pass patches (mean/std assertion, pinned regex, duplicate-names negative test), checked off in spec-2-5's review findings.
- `low` — `Manual.validate` mutates `_names` as a side effect (port implies pure check): harmless in-tree (the probe is discarded and `fit` revalidates); the redesign is not warranted by any caller that meets the wart.
- `low` — Pre-2.6 finished runs are silently re-executed because `_completed_seeds` now requires `feature_names`: no stored runs exist in the repo, the recompute is idempotent and one-time, and the schema addition genuinely requires it.
- `low` — The `FeatureSelector` port's val-capable `fit` and the port surface (`candidates`/`choose`/`validate`) are untested-by-absence / beyond the spec's Code Map: wording about the port text, not a code defect; changing the wording edits the spec.

### Review Findings (second pass, 2026-10-10)

_Re-review of the uncommitted code change vs `14644fe`. Layers: Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor. The earlier Decision (per-candidate val scores not recorded) still stands and was re-raised by two layers._

**Patch**
- [x] [Review][Patch] Head `val_score` wiring is not verified: `check_head` only asserts `0.0 <= val_score <= 1.0` and every head initialises it to 0.0, so a head that forgets to assign it (or unpacks the wrong tuple slot) passes, and MRMR `k` selection silently becomes "always the first k". The MRMR test overwrites `LogReg.val_score` with a fake, so it cannot see this. Add a head-level check on a separable fixture (score must be well above chance), or run a real head under multi-k MRMR [tests/contracts/head.py:18]
- [x] [Review][Patch] `_expand_feature_selection` tests `raw is None` twice; collapse to `if raw is None: return None` then a single mapping check [src/csi_bench/config/experiment.py:68]
- [x] [Review][Patch] Test-strength gaps: `test_selector_sees_only_train_and_val_rows` checks row counts only, and there is no head-only-change cache-hit test (AC5) beside `test_selector_only_change_hits_the_feature_cache` [tests/test_run.py]

**Rejected (second pass)**
- `false` - NaN `head.val_score` poisons the comparison: `val_score` comes from `f1_score(..., zero_division=0)`, which returns a finite float.
- `false` - `feature_names` beside split keys breaks consumers that iterate a seed's items: the only reader is `run.py`, which indexes `SPLIT_NAMES` explicitly.
- `false` - `Manual` entry `''` or `'fam.'` misleads: neither matches any column, so `validate` raises the `ConfigError` naming the entry and the available columns.
- `false` - Fewer-than-K ranking, non-finite or constant train columns, and single-class train: repeats of first-pass rejections (`validate` enforces `max(k) <= ncols`; no reachable path demonstrated).
- `false` - Out-of-range or negative `choose`, `transform` before `fit`, bare `KeyError` in `take_columns`, zero-candidate selector, head without `val_score`: repeats of first-pass items (no in-tree caller reaches them, or already listed as patches above).
- `low` - Reported val metrics are optimistic because val picks both k and the head hyperparameters: this is already how the head grid works, and nothing in the story promises unbiased val.
- `low` - Equivalent manual lists (duplicates, different order) hash to different run ids: harmless extra run, and normalising adds branches for a rare case.
- `low` - Selector refit per seed and a separate probe instance: the cost is small and removing it needs caching or hoisting.
- `low` - Failure after earlier seeds are written leaves partial state; resume of pre-2.6 `metrics.json` has no test: pre-existing behaviour and no stored runs exist.
- `low` - Test helper brittleness (`ALL_NAMES`, unused tuple element, call-count coupling, untested k=5 tie): cosmetic.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check . && uv run ruff format --check .` -- expected: clean
- `uv run ty check` -- expected: clean
- `uv run lint-imports` -- expected: contracts kept
