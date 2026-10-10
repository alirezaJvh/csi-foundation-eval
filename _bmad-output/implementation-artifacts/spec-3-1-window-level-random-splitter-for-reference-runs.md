---
title: 'Window-level random splitter for reference runs'
type: 'feature'
created: '2026-10-10'
status: 'done'
route: 'dispatch'
baseline_commit: '4abd6d237f1d90ecf09a69f161b198e4b52820d6'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The paper assigned individual windows to train/val/test at random. No splitter reproduces that, and nothing stops such a leaky split from entering the comparison.

**Approach:** Add a registered `random_window` splitter whose committed plan carries `unit: window`, seed and fractions. The Windower derives the per-window assignment from the plan. Config validation and the runner reject any `comparison` run on a window-unit plan.

## Boundaries & Constraints

**Always:** The assignment is a pure function of `(seed, recording_id, window_len, stride, n_packets)`, so the same seed gives the same windows. Fractions are exact per recording (`floor` for val and test, the rest train). The plan is independent of rooms and models. Its `split_hash` differs from every time-block plan. Existing time-block plans keep their hash and still load.

**Never:** Change `time_block` behaviour or its `split_hash`. Modify the committed `splits/tor_vergata/*.json`. Add a split-generation CLI or the paper reproduction sweep (Story 3.2). Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Reference run | `random_window` plan, `role: reference` | windows randomly split, plan `unit == "window"` | N/A |
| Same seed | split twice, same params | identical plan and assignment | N/A |
| Other seed | seed 0 vs 1 | different `split_hash` and different assignment | N/A |
| Comparison on window plan | `role: comparison` and `split` names a window-unit plan | N/A | `ConfigError` at validation (and again in `run`) |
| Missing plan at validation | `split` not found | validation passes; `run` raises its usual `DataError` | N/A |
| Old plan | committed time-block JSON without `unit` | loads as `unit: block`, hash unchanged | N/A |
| Tiny recording | a part gets 0 windows for a class | N/A | `DataError` naming the class and part |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/contracts.py` -- `SplitPlan`: add `unit: str = "block"`.
- `src/csi_bench/core/splitplan.py` -- `compute_split_hash` drops `unit` when `"block"` (old hashes stay valid); `load_split_plan` reads `raw.get("unit", "block")`; add `read_plan_unit(path)` and `check_split_role(unit, role)` (raises `ConfigError`).
- `src/csi_bench/adapters/splitters/random_window.py` -- new, pattern `time_block.py`: `Params(seed: int = 0, fractions = (0.6, 0.2, 0.2))`, `VERSION = 1`. Plan: `unit="window"`, `gap_packets=0`, `blocks[rid]` = `{train,val,test: (0, n_packets)}` (full-recording extents, used for per-block extraction and the cache).
- `src/csi_bench/core/windows.py` -- `build_windows`: for `plan.unit == "window"`, per recording enumerate all starts (step `stride`), permute with `np.random.default_rng([seed, crc32(recording_id)])`, cut by `plan.params["fractions"]`; skip the `gap_packets` check.
- `src/csi_bench/config/experiment.py` -- `validate_registry`: if the plan resolves via `find_split` (swallow `DataError`), call `check_split_role(read_plan_unit(...), config.role)`. Uncommitted edits exist here; keep them.
- `src/csi_bench/runner/run.py` -- after `load_split_plan`, call `check_split_role(plan.unit, config.role)`. Uncommitted edits exist here; keep them. `run_chain` and `_extract` need no change (blocks are full recordings).
- `tests/test_random_window.py` -- new; `tests/contracts/splitter.py` -- reuse where blocks apply (full-extent blocks overlap, so assert separately).

## Tasks & Acceptance

**Execution:**
- [x] `core/contracts.py`, `core/splitplan.py` -- add `unit`, hash back-compat, role check helpers -- one place owns the rule
- [x] `adapters/splitters/random_window.py` -- registered splitter; `ConfigError` for bad fractions; `DataError` if a class has no windows in a part (estimated from `n_packets` with `window_len` 1)
- [x] `core/windows.py` -- window-unit assignment -- the Windower owns window boundaries
- [x] `config/experiment.py`, `runner/run.py` -- role gate at validation and at run
- [x] `tests/test_random_window.py` -- cover every I/O row, fractions exact per recording, windows of a part never overlap other parts' ids, and the committed time-block plan still loads with the same hash; one `run` smoke test on a synthetic dataset with `role: reference`

**Acceptance Criteria:**
- Given `random_window` with `seed` and fractions, when a reference run uses it, then windows are assigned randomly, the plan has `unit: window`, and its `split_hash` differs from the time-block plan's.
- Given the same seed, when split and windowed twice, then the assignment is identical.
- Given `role: comparison` and a window-unit plan, when the config is validated, then `ConfigError`.

### Review Findings

- [x] [Review][Patch] `describe` table misreports window-unit plans (decided: show train/val/test as n/a for `unit == "window"`) [src/csi_bench/runner/cli.py:41]
- [x] [Review][Patch] No `DataError` when a class gets 0 windows in a part (I/O row "Tiny recording"; Tasks: estimate from `n_packets` with `window_len` 1) [src/csi_bench/adapters/splitters/random_window.py:28]
- [x] [Review][Patch] `check_splitter` contract assumes ordered, gap-separated blocks and rejects `RandomWindow`; make it unit-aware and call it from `tests/test_random_window.py` [tests/contracts/splitter.py:7]
- [x] [Review][Patch] The run-side role gate is never the failing check (validation fires first); add a test that reaches `_execute` with a comparison config on a window plan [src/csi_bench/runner/run.py:596]
- [x] [Review][Patch] No test that each start lands in exactly one part and the parts cover all starts (`_assignment` dict hides duplicates) [tests/test_random_window.py:43]
- [x] [Review][Patch] `test_committed_block_plans_keep_hash_and_load` never loads a plan or asserts `unit == "block"`; call `load_split_plan` and check `unit` and `compute_split_hash` [tests/test_random_window.py:136]
- [x] [Review][Patch] `test_config_validation_rejects_comparison_on_window_plan` has dead setup (extra write and unlink) and rebuilds a frozen dataclass through `__dict__`; build the plan for `tor_vergata` directly with `dataclasses.replace` [tests/test_random_window.py:106]
- [x] [Review][Patch] (Story 2.9 code, fixed on request: one live thread per allowlist entry; mutation-checked) `test_fork_is_safe_ignores_loky_housekeeping_threads` starts housekeeping threads with no target; they are dead before `_fork_is_safe` runs (verified: `is_alive()` is False right after `start()`), so the test passes even with the allowlist removed, and `TMonitor`/`tqdm_monitor` are never tested [tests/test_run.py:950]

**Rejected:**
- false: the name allowlist matching a non-loky `multiprocessing.Queue` feeder (`run.py:86`) could re-open the fork deadlock. `multiprocessing.Queue` registers `_after_fork`, which resets its lock and feeder state in the child, and no project code keeps a queue alive during extraction. loky's feeder is a plain `threading.Thread`, so a module check could not tell them apart anyway.
- false: "context extractors compute the full-recording context three times". All three parts share the block `(0, n)`, so `context_cache_key` is the same; the context is computed once and the other two parts hit the cache.
- by design: preprocessing and context are fit on the whole recording under window plans. The spec's Design Notes accept this because the plans are reference-only; changing it would mean editing the spec.
- low: plan params are not re-validated at load, and unknown `unit` strings are not rejected. Only a hand-edited plan triggers this, and the fix adds guards.
- low: float error in `int(f * n)` could put a count off by one. This matches the spec's "floor", and the default 0.2 is unaffected in practice.
- spec-mandated: `validate_registry` swallows every `DataError`, including an ambiguous plan. The spec says to swallow `DataError`.
- low: `write_split_plan` now writes `"unit": "block"` into new block plans. The hash is unchanged and nothing regenerates the committed plans (no generation CLI).
- pre-existing: the missing-plan test doesn't assert the `DataError` from `run`. That behaviour predates this story.
- other story: `weights_revision` in `context_cache_key` invalidates the cache once, which is intended; the per-block INFO cache logging is low.
- out of scope: no CLI or docs for generating a `random_window` plan. The spec defers that to Story 3.2.

## Implementation Notes

Implemented as specified. Role gate lives in `core/splitplan.py` (`check_split_role`, `read_plan_unit`). `uv run pytest` 320 passed; ruff and import-linter clean. `ty` reports one pre-existing error at `tests/test_run.py:995`, unrelated.

## Spec Change Log

## Review Triage Log

## Design Notes

A per-window plan file would be large and tied to one window length. Deriving the assignment from the seed keeps one small committed plan valid for every window length and room set. Because `blocks` span whole recordings, train-fitted preprocessing sees the whole recording on window-unit plans. That matches the paper and is allowed only because these plans are reference-only.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ty check` -- expected: clean
