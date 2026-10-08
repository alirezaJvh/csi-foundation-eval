---
title: 'Generate the committed time-block train/val/test split'
type: 'feature'
created: '2026-10-08'
status: 'done'
baseline_commit: 'd0d5520aec8dad657c40b00a9dbe7d8724f1fc33'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-1-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every method must see the same leak-free split, but no `SplitPlan` contract, splitter or committed split file exists yet.

**Approach:** Add the frozen `SplitPlan` contract with write/load helpers to `core/`, and a registered `time_block` splitter that gives each recording contiguous train | val | test packet blocks separated by `gap_packets`.

## Boundaries & Constraints

**Always:**
- Layout per recording, in order: `train | gap | val | gap | test`. Indices are 0-based, end-exclusive, on the packet axis as delivered.
- `Params`: `gap_packets: int = 64`, `fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)` (train, val, test; must sum to 1, each > 0, else `ConfigError`). Registered `@register("splitter", "time_block")`, `VERSION = 1`.
- Sizing: `usable = n_packets - 2*gap`; `val = floor(f_val*usable)`, `test = floor(f_test*usable)`, `train = usable - val - test`. A part of size 0 is a `DataError` naming the class and part (`val`/`test`) (or recording, if `usable <= 0`). Class check is per class, summed over its recordings.
- `SplitPlan` is a frozen dataclass: `dataset`, `splitter`, `params` (dict), `gap_packets`, per-`recording_id` blocks `{train,val,test: (start, end)}`, per-`recording_id` `source_sha256`, `split_hash`. `split_hash = short_hash` of everything but itself, so same inputs and params give identical plans.
- The plan is written as canonical-style JSON to `splits/<dataset>/<split_hash>.json`. `load_split_plan(path, recordings)` raises `DataError` if any `source_sha256` differs from the recording's `meta` or the recording sets differ.
- The split covers all recordings of the dataset once; it never sees windows, models or room selection.

**Never:**
- No `describe`, CLI, config or windower code (stories 1.4+); no `random_window` splitter; do not modify `data/`; no `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Defaults | n=1000, gap 64 | usable 872: train 0:524, val 588:762, test 826:1000 | N/A |
| Real data | the 19 recordings | all blocks non-empty, gaps >= 64, no overlap | N/A |
| Repeat | same inputs and params twice | identical plan and `split_hash` | N/A |
| Too short | recording with n <= 2*gap, or part sizing to 0 | — | `DataError` naming class and part |
| Bad fractions | `(0.5, 0.2, 0.2)` or a 0 | — | `ConfigError` |
| Changed file | split loaded after a recording's sha differs | — | `DataError` naming the recording |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/contracts.py` -- add `SplitPlan` beside `Recording`
- `src/csi_bench/core/ports.py` -- add `Splitter` Protocol: `split(recordings, dataset) -> SplitPlan`
- `src/csi_bench/core/hashing.py` -- reuse `short_hash`, `canonical_json`; `core/errors.py` -- `DataError`, `ConfigError`
- new `src/csi_bench/core/splitplan.py` -- `write_split_plan(plan, root)`, `load_split_plan(path, recordings)`
- new `src/csi_bench/adapters/splitters/{__init__,time_block}.py` -- pattern: `adapters/datasets/tor_vergata.py`
- `src/csi_bench/runner/discover.py` -- finds the new subpackage; tests use the registry-reset fixture style of `tests/test_tor_vergata.py`
- `tests/contracts/` -- add a reusable `check_splitter` contract (blocks in range, ordered, gaps, no overlap, determinism)
- `splits/tor_vergata/` -- commit the generated plan for the 19 real recordings (not git-ignored)

## Tasks & Acceptance

**Execution:**
- [x] `core/contracts.py`, `core/ports.py`, `core/splitplan.py` -- `SplitPlan`, `Splitter`, write/load with sha check
- [x] `adapters/splitters/time_block.py` -- registered splitter, sizing and class checks
- [x] `tests/contracts/splitter.py` -- reusable `check_splitter`
- [x] `tests/test_time_block.py` -- matrix cases on synthetic `Recording`s, write/load round trip, changed-sha failure, real-data plan (skip if data absent)
- [x] `splits/tor_vergata/<split_hash>.json` -- generate and add the committed plan; a test asserts it loads against the real data

**Acceptance Criteria:**
- Given the 19 recordings and default params, when split, then each has one contiguous block per part, gaps >= `gap_packets`, no overlap, and the contract passes.
- Given the same inputs twice, when split, then plan and `split_hash` are identical and the JSON lands at `splits/tor_vergata/<split_hash>.json` with each `source_sha256`.
- `ruff check`, `ruff format --check`, `ty check`, `lint-imports`, `pytest` all pass.

### Review Findings

- [x] [Review][Patch] `load_split_plan` error path untested and incomplete: no test for missing file, invalid JSON, missing key, short block; a wrong-shaped `blocks`/`source_sha256` raises uncaught `AttributeError` instead of `DataError` — add `AttributeError` to the except tuple and a parametrized `test_unreadable_plan` [src/csi_bench/core/splitplan.py:176-191, tests/test_time_block.py]
- [x] [Review][Patch] `split_hash` defined in two places (hand-built `body` vs `compute_split_hash`) — build the `SplitPlan` first, then set the hash via `compute_split_hash` [src/csi_bench/adapters/splitters/time_block.py:~60-70]
- [x] [Review][Patch] Committed plan only checked when the dataset is present — add a data-free test that loads `splits/tor_vergata/*.json` and checks `split_hash` and block invariants [tests/test_time_block.py:~368]

**Rejected (14):**
- Empty (s,s) val/test span for a short recording when class total is positive — low; spec fixes the class check as summed per class, and fix would contradict it.
- Float flooring off by one (`0.29*100`) — false for the defaults (0.2/0.6); deterministic IEEE result, not reachable with committed params.
- `VERSION` not in hash — low; spec lists the hashed fields, future-only concern.
- Loader does not validate block bounds/order when the hash is recomputed — low; hash is an integrity check, hand-edit with recompute is unlikely, fix adds guards.
- Loader does not check dataset/splitter against caller — low; signature is fixed by spec.
- Empty recordings / duplicate `recording_id` — low; datasets return unique sorted ids, guard adds complexity.
- `params` dict / unknown keys / strict types — false; matches `TorVergata.Params` convention.
- Path traversal via `plan.dataset`, non-atomic write, silent overwrite — low; dataset name is developer config, file name is the content hash so overwrite is idempotent.
- Mutable dicts in frozen `SplitPlan` — low; no caller mutates, same convention as `Recording`.
- No CLI to regenerate plan — low; not in spec scope.
- Train absorbs remainder / first fraction discarded — false; specified sizing.
- No class-coverage check — false; not specified.
- Swapped "missing/unplanned" labels — false; labels are correct.
- Boundary test `n=2*gap+1`, assertionless `test_class_check_sums_over_recordings` — low; behavior covered elsewhere.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
