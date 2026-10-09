---
title: 'Full classifier family with seeds'
type: 'feature'
created: '2026-10-08'
status: 'done'
baseline_commit: '5269e4feb127cf9cc1d03ee909a69408f6eefbaa'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-2-1-run-one-classical-experiment-end-to-end.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Only `logreg` exists, a finished run cannot be extended with more seeds, and the runner sets no CPU/thread/determinism state, so classifiers cannot be compared fairly with mean ± std.

**Approach:** Add `svm`, `knn`, `random_forest` and `mlp` heads that share one tune-on-val / refit-on-train∪val routine with `logreg`. Let a finished run execute only its missing seeds, and have the runner own device, torch threads and determinism flags and record them in provenance.

## Boundaries & Constraints

**Always:**
- Every head is an adapter in `adapters/heads/`, registered as `head/<name>` with `Params` (`extra="forbid"`, non-empty grids) and `VERSION`. Grids: `svm` `C_grid` + `kernel: rbf|linear`; `knn` `k_grid`; `random_forest` `n_estimators_grid`, `max_depth_grid` (int or null); `mlp` `hidden_layer_sizes` (default `(64,)`), `alpha_grid`, `max_iter`; `logreg` unchanged. Head constructor stays `(params, seed)`; `seed` drives `random_state` where the model has one.
- Shared routine in `core/head_search.py` (core imports nothing internal): for each grid candidate fit a fresh scaler + model on train, score macro-F1 on val, pick the best (first wins ties); then refit scaler and model on train ∪ val. Heads call it; `logreg` is refactored onto it with identical results. Heads never see test labels; `predict` is called on test once per seed.
- Seeds: `seeds/<seed>/predictions.parquet` per seed; `metrics.json` holds per-seed scores plus mean ± std (existing schema). A run whose `status.json` is `done` with some seeds already in `metrics.json` and their `predictions.parquet` on disk executes only the missing seeds, keeps the old per-seed metrics (loaded via a new `metrics_from_dict` in `core/results_schema.py`), and rewrites `metrics.json` with all seeds. With no missing seed nothing is executed or rewritten and the command still prints the `run_id`, exit 0.
- The runner (never adapters) sets before the seeds loop: CPU device, `torch.set_num_threads(TORCH_THREADS)`, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic`/`benchmark` flags, and per seed `random.seed`, `numpy.random.seed`, `torch.manual_seed`. `provenance.json` gains `runtime: {device, torch_threads, deterministic, torch_version}`.

**Never:**
- No cache, feature selection, sweeps, `--force`, locking or other rerun semantics (stories 2.4–2.7); no extra classifiers, no torch-based MLP (use scikit-learn `MLPClassifier`); adapters never touch torch threads, device or determinism; no changes to `data/`, `splits/`, the Windower or preprocessing. No `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Each head | `head: {name: svm\|knn\|random_forest\|mlp\|logreg}` | run `done`, val and test metrics | N/A |
| Multi seed | `seeds: [0, 1, 2]` | three `seeds/<n>/predictions.parquet`; per-seed metrics and mean ± std | N/A |
| Add a seed | finished run, config seeds `[0,1,2]` → `[0,1,2,3]` | only seed 3 executes; seeds 0-2 files and metrics untouched | N/A |
| Same seeds again | finished run, same config | nothing executed, `run_id` printed | N/A |
| Missing seed output | `done` run whose `seeds/1/` was deleted | seed 1 reruns | N/A |
| Bad grid | `head.params.k_grid: []` or unknown key | nothing loaded | `ConfigError` naming the field, exit 2 |
| Test-label isolation | test labels changed | fitted head and predictions unchanged | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/heads/logreg.py` -- pattern and logic to move into `core/head_search.py`; keep its tests green (`tests/test_logreg.py`)
- `src/csi_bench/core/ports.py` -- `Head` port (`fit(train, y_train, val, y_val)`, `predict`), unchanged
- `src/csi_bench/core/results_schema.py` -- `metrics_as_dict`, `metrics_document`; add `metrics_from_dict`
- `src/csi_bench/runner/run.py` -- `_execute` seeds loop (`per_seed`), `_run_seed`, `collect_provenance`, `run`; add seed resume and runtime setup
- `src/csi_bench/config/experiment.py` -- head params already validated through `Params`; no change expected
- `tests/contracts/head.py`, `tests/test_logreg.py`, `tests/test_run.py` -- patterns to copy; synthetic recordings fixture in `test_run.py`
- new: `core/head_search.py`, `adapters/heads/{svm,knn,random_forest,mlp}.py`, `tests/test_heads.py`

## Tasks & Acceptance

**Execution:**
- [x] `core/head_search.py`, `adapters/heads/logreg.py` -- shared tune/refit routine; refactor logreg onto it
- [x] `adapters/heads/{svm,knn,random_forest,mlp}.py` -- four heads with grid params
- [x] `core/results_schema.py` -- `metrics_from_dict` (round-trips `metrics_as_dict`)
- [x] `runner/run.py` -- runtime setup and provenance; run only missing seeds of a `done` run
- [x] `tests/contracts/head.py`, `tests/test_heads.py` -- contract run on every registered head via `discover()` (synthetic data, accuracy sanity, seed determinism, scaler inside head, grid selected on val only, test labels never reach `fit`); param rejection per head
- [x] `tests/test_run.py` -- multi-seed run, add-a-seed resume (spy: only new seed fitted, old files untouched), same-seeds no-op, deleted seed reruns, provenance `runtime`, each head end-to-end

**Acceptance Criteria:**
- Given `seeds: [0, 1, 2]`, when I run, then `metrics.json` has per-seed scores and mean ± std and three prediction files exist.
- `ruff check src tests`, `ruff format --check src tests`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

### Review Findings

- [x] [Review][Patch] Set the CPU device in `_setup_runtime` — spec-2.3 "Always" says the runner sets, before the seeds loop, `CPU device`, `torch.set_num_threads(TORCH_THREADS)`, `torch.use_deterministic_algorithms(True)` and the cudnn flags. `_setup_runtime` set the threads and cudnn/determinism flags but made no device call; added `torch.set_default_device(DEVICE)` so the device is owned by the runner, not only recorded. [src/csi_bench/runner/run.py:75-81]
- [x] [Review][Patch] `tune_and_refit` docstring overstated the routine: it said "for every candidate a fresh scaler and model are fitted", but one scaler is fitted on train and reused for every candidate (result-equivalent: the scaler is deterministic from train). Reworded the docstring to match. [src/csi_bench/core/head_search.py:22-24]
- [x] [Review][Patch] `test_test_labels_never_reach_fit` did not exercise its name: it fit twice on identical train/val and compared predictions, re-testing determinism. Replaced with a `fit` recorder that asserts `fit` is handed exactly the train and val labels and no extra (test) array. [tests/test_heads.py:91-110]
- [x] [Review][Patch] Added a `run`-route test for the `DataError` arm: a valid config with no committed split plan (or several with `split: null`) exits 2 with a clean message and no run dir; only `describe` covered `find_split` errors before. [src/csi_bench/runner/run.py:396]
- [x] [Review][Patch] Added a test for the empty-split guard: a plan whose train/val/test block is non-empty but shorter than `window_len` (and `<= gap_packets`) raises `ConfigError` naming the empty part. [src/csi_bench/runner/run.py:320-325]
- [x] [Review][Defer] Resume after a mid-run failure redoes every seed — `_completed_seeds` requires `status == "done"` and `metrics.json` is written only after the whole seed loop, so predictions for finished seeds are discarded. Deferred: AD-14 resumability/`--force` belongs to story 2.7. [src/csi_bench/runner/run.py:98-117]
- [x] [Review][Defer] `started_at`/`duration_seconds` on an add-a-seed resume reflect only the resume invocation — `started_at` is re-stamped `_now()` and duration measures only the missing-seed loop. Deferred: run-timing semantics on re-execution belong to story 2.7. [src/csi_bench/runner/run.py:284]
- [x] [Review][Defer] Reducing `config.seeds` leaves stale seed metrics — `run_id` excludes seeds, so `[0, 1, 2] → [0]` finds no missing seed and returns without pruning `metrics.json`/`seeds/`. Deferred: rerun/prune semantics belong to story 2.7. [src/csi_bench/runner/run.py:277-280]

#### Rejected

- `false` — `_setup_runtime` below extraction escapes determinism/thread pins (blind): spec-2.3 mandates setup "before the seeds loop" (the code's position), and extraction/preprocessing use no torch or RNG, so no nondeterminism occurs.
- `false` — hyperparameter selection is not re-validated under the final union scaler (blind): spec-2.3 prescribes exactly this (search in train-scaled space, then refit scaler and model on train ∪ val).
- `false` — `collect_provenance`'s `zip(..., strict=True)` can raise a bare `ValueError` (blind): `run_chain` appends one state per step, so the lengths always match.
- `low` — `_completed_seeds` swallows or crashes on a malformed `metrics.json` (blind, edge): the program never writes a non-object/short document, so it is unreachable in normal use and the guards would add branches.
- `low` — `metrics_from_dict` duplicates the `Metrics` field list and the round-trip test "skips the JSON path" (blind): the test passes exactly the JSON-shaped structures (`json.loads` yields the same dict/list/str-key shapes), so the path is covered; the duplication names no harm.
- `low` — round-trip fixture is internally inconsistent yet passes (blind): `Metrics` is a plain dataclass with no cross-field validation; harmless test data.
- `low` — `check_seed_determinism` cannot catch a head that ignores `seed` (blind): the contract's stated purpose is determinism, not seed differentiation; kNN is legitimately seed-free.
- `low` — preprocess-chain test omits the provenance `version` field (blind): story-2.2 coverage, not this change.
- `low` — `runtime_info` nests `device` while provenance also has a top-level `device` (blind): cosmetic duplication with no drift source in the code.
- `low` — `svm` passes an ignored `random_state`, `knn` stores an unused `seed` (blind): the spec requires the shared `(params, seed)` constructor; no behavioral effect.
- `low` — a `k_grid` value larger than the training set makes sklearn raise (edge): unreached with the real dataset (thousands of windows) and fails loudly; a clamp would add guard complexity for an input not demonstrated in normal use.
- `low` — `logreg`'s refactor is not "identical results": its final scaler moves from train-only to train ∪ val (auditor): this implements the user-decided patch recorded in the loaded spec-2.1 review log and is pinned by `test_final_scaler_is_fitted_on_train_and_val`; reconciling the wording would edit the spec, not the code.
