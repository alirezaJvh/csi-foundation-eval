---
title: 'Share paper_beamforming per-position metrics across the sweep'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: 'e6cc8805f93e52c2b4d11c7d4cda4b23bf2ba70c'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
  - '{project-root}/_bmad-output/planning-artifacts/sprint-change-proposal-2026-10-10-story-2-11.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `paper_beamforming` computes the six per-position arrays once per block per run (Story 2.10), but they do not depend on `window_len`, `stride` or the room set, so the 12 extractions of `experiments/classical_baseline.yaml` (4 window lengths × 3 room sets) repeat the same work. About 1.6 h of extraction today.

**Approach:** The runner computes each block's per-position arrays once (extractor offers `compute_context(block)`), stores them in a runner-owned per-block cache keyed without `window_len`, `stride` or room set, and passes them to `extract_blocks(..., context=...)`. r1+r2 reuses the r1 and r2 entries.

## Boundaries & Constraints

**Always:** Feature matrix equal to the Story 2.10 block path and the slice path exactly (same values, same row order), for `fix_distance_loop` true and false, any `window_len` and `stride`, and equal to a matrix cached by the old path. Per-block entry key: dataset and version, recording id and source hash, block range, preprocessing chain with fitted-state hash, extractor name, result-affecting params, `VERSION`, vendor hash. Only the runner reads and writes the entries; writes are atomic; a corrupt or missing entry is recomputed; deleting `.cache` never changes a result. Arrays cover the whole block. Peak RSS stays within about 4 GB.

**Never:** Change the feature-matrix cache key, `VERSION` or run ids. Change the extractor maths or the paper-parity oracle. Let an extractor touch the cache. Add sweep-level parallelism or run locks. Share entries across different fitted preprocessing state.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Second run, other window_len | Block entry exists | No recompute; rows equal slice path | N/A |
| Room set r1+r2 | r1 and r2 entries exist | Built from both; none recomputed | N/A |
| Stride > 1 | Offsets not contiguous | Same rows as slice path | N/A |
| Different fitted state | e.g. `global_rescale` with other room set | Different key, fresh arrays | N/A |
| Corrupt entry | Unreadable npz | Recomputed and rewritten | Logged warning |
| Matrix cache hit | Feature matrix already cached | Block cache not touched | N/A |
| No `compute_context` | `amplitude_stats` | Unchanged path | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/extractors/paper_beamforming.py` -- `_position_metrics(ch1, ch2, fix, first, count)` returns `(6, count)`; add `compute_context(block)` (all `B - 16` positions) and `context=` on `extract_blocks`; `_reduce` slices it. Math untouched.
- `src/csi_bench/core/ports.py` -- document optional `compute_context`, `context=`; add `ContextCache` port (get/put one ndarray by key), next to `FeatureCache`.
- `src/csi_bench/adapters/storage/feature_cache.py` -- model for atomic npz writes; add `NpzContextCache` (own module `context_cache.py`).
- `src/csi_bench/config/experiment.py` -- `feature_cache_key` (l.248) is the model; add `context_cache_key(...)`, no `rooms`, `window_len`, `stride`.
- `src/csi_bench/runner/run.py` -- `_extract_blocks` (l.117) groups by `(recording_id, split)`; `_extract` (l.295) builds keys; `_execute` (~l.538) makes `NpzFeatureCache(cache_root)`. Pass `plan.blocks`, a context cache under `<cache_root>/context`, config and fitted state; use full block bounds from `plan.blocks`, not min/max of the windows.
- `tests/contracts/extractor.py`, `tests/test_paper_beamforming.py`, `tests/test_run.py`, `tests/test_feature_cache.py` (if present) -- extend; existing 2.10 tests stay.
- Not to touch: `feature_cache_key`, `_extract_parallel`, `tests/oracles/`.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/adapters/extractors/paper_beamforming.py` -- `compute_context`, `context=` in `extract_blocks` -- the shareable part
- [x] `src/csi_bench/core/ports.py`, `src/csi_bench/adapters/storage/context_cache.py` -- `ContextCache` port and npz adapter -- runner-owned store
- [x] `src/csi_bench/config/experiment.py` -- `context_cache_key` -- complete key without window/stride/rooms
- [x] `src/csi_bench/runner/run.py` -- get-or-compute per block, pass context, log hits/misses -- the sweep speed-up
- [x] tests -- exact equality (synthetic and real windows, both flags, stride 1 and >1, several window lengths against one context); second run recomputes nothing; r1+r2 from r1 and r2 entries; different fitted state, different key; corrupt entry recomputed; feature-matrix key unchanged -- ACs
- [x] this spec, Implementation Notes -- extraction time and peak RSS before and after for the baseline's `paper_beamforming` runs

### Review Findings

- [x] [Review][Decision] Working tree bundles Story 2.9 work into the 2-11 diff -- resolved: review narrowed to this story's files (extractor, ports, context cache, config keys, runner block-context path, its tests); Story 2.9 code (parallel extraction, head tuning, `svm` cache) stays out of this blessing, its findings ride with 2.9.
- [x] [Review][Decision] Loky threads disable parallel extraction from run 2 of a workers>1 sweep -- resolved: applied option 1; `_fork_is_safe` now ignores benign outlives-everything housekeeping monitors (loky executor manager/queue feeder, tqdm `TMonitor`), with a two-run regression test (`test_loky_head_tuning_does_not_disable_parallel_extraction`) proving the second run still extracts in parallel. During the fix, the tqdm monitor was found to be the real-world polluter (`mrmr` imports tqdm), explaining the original "multi-core gave no speed-up" measurement.
- [x] [Review][Patch] Context-cache hits are never logged [src/csi_bench/runner/run.py:175] -- fixed: the dead `counts` dict was removed and both paths log ("computed"/"cached" per block); covered by `test_block_context_hits_and_misses_are_logged`.
- [x] [Review][Patch] `context_cache_key` omits `WEIGHTS_REVISION` that `feature_cache_key` carries [src/csi_bench/config/experiment.py:327] -- fixed: `"weights_revision": getattr(ext_cls, "WEIGHTS_REVISION", None)` added next to `vendor_hash`.
- [x] [Review][Patch] Untracked `context_cache.py` was excluded from the review diff [src/csi_bench/adapters/storage/context_cache.py] -- resolved: `git add`ed with the story change.
- [x] [Review][Defer] No automated regression guard for equality against a pre-change cached matrix -- deferred: AC-2's "against a matrix cached by the old path" clause was satisfied by a recorded manual check (L=20 r1, bit-for-bit per Implementation Notes); an automated guard needs a stored fixture of pre-change output, which the suite cannot regenerate.
- [x] [Review][Defer] BLAS/OpenMP thread pools unbounded in forked extraction workers -- deferred: only torch threads are capped (`TORCH_THREADS = 1`); belongs to Story 2.9's stalled-speedup diagnosis (956 s vs 989 s).
- [x] [Review][Defer] `svm` kernel cache 500 MB per instance under parallel head fitting -- deferred: 4 workers could hold ~2 GB, but measured peak RSS is 1.76 GB, well inside the ~4 GB budget; reconciliation of the "kept modest" comment belongs to Story 2.9 (`src/csi_bench/adapters/heads/svm.py:16`).

**Rejected (false or not worth fixing):**
- `_extract_blocks` nested-Mapping shape assumption -- false: `plan.blocks` is built as `{recording: {split: (start, end)}}` (`core/splitplan.py:60-62`), matching the consumption.
- Mixed window lengths in one (recording, split) group silently use `members[0]`'s length -- false: `build_windows` takes one `window_len` per run (`core/windows.py:13`), so a run's windows are uniform by construction.
- No timeout on extraction futures -- low: speculative hang; a timeout parameter adds machinery for a scenario never observed.
- Guard `extract_blocks` returned row count -- low: a wrong shape fails loudly in `values[indices]`; reachable only with a buggy extractor.
- Contract `_check_block_path` indexes `windows[1]` unconditionally -- low: only context-capable extractors reach it (paper_beamforming passes 21/21); loud test-helper failure, not silent.
- `uses_context` without `extract_blocks` gives raw `AttributeError` -- low: the port contract documents the requirement (`core/ports.py:39`) and the contract suite exercises every registered extractor.
- Nested parallelism oversubscription (head workers x estimator workers) -- false: no head sets an estimator `n_jobs` (all defaults, single-threaded), and joblib runs nested `Parallel` sequentially anyway.
- Readable-but-wrong-shaped context entry aborts the run instead of recompute -- false vs the AC: the frozen row covers *unreadable* npz (recomputed + warning, implemented in `NpzContextCache.get`); the key binds block range + fitted state, so a mismatched readable entry requires a key collision.
- r1+r2 entry reuse only holds for fitless preprocessing chains -- rejected: the fix would edit the frozen spec; keying by fitted state is correct per the fitted-state AC, and the r1+r2 row describes the baseline's fitless `ant2` chain.
- Extraction-parallelism default 4 workers contradicts deferred "default to 1" decision -- rejected as an entry here: it is Story 2.9 code (see Scope decision), and the deferred item explicitly awaits the 2.9 review.
- sprint-status.yaml mixes date formats -- low cosmetic in an agent-tracking file.
- `compute_context` missing ndim guard for non-3D blocks -- low: blocks are always slices of the 3-D `csi` array; unreachable.

**Acceptance Criteria:**
- Given two runs differing only in `window_len`, `stride` or rooms, when the second extracts, then it reads the block arrays from the cache and recomputes none that exist.
- Given any window length, stride or room set and both flags, when extracted through shared arrays, then the matrix equals the Story 2.10 matrix exactly, including against an old cached matrix, and the paper-parity test passes.
- Given a preprocessing chain with fitted state, then entries are keyed by that state; the matrix cache key, `VERSION` and run ids are unchanged; deleting `.cache` changes no result.
- Given the baseline's `paper_beamforming` runs, then total extraction time and peak RSS before and after are recorded here, RSS about 4 GB or less.

## Implementation Notes

Sweep of the 12 `paper_beamforming` extractions of `classical_baseline.yaml` (window_len 20/30/40/50 x rooms r1, r2, r1+r2; svm head; fresh `.cache`; `/usr/bin/time -v`, workers default):

| | extraction, 12 runs | whole sweep | peak RSS |
|---|---|---|---|
| before (Story 2.10, no sharing) | not re-run; 365 s measured for the first run (L=20, r1), so >= 12 x 365 s = 1.2 h, 1.6 h estimated | not re-run | 1.55 GB |
| after (shared per-block arrays) | 585 s (9.8 min) | 14:00 | 1.76 GB |

After, per run: L=20 r1 365.8 s (33 blocks computed), L=20 r2 207.4 s (24 blocks computed), L=20 r1+r2 1.8 s (0 computed), all other 9 runs 0.6-1.8 s (0 computed). The remaining cost is the single pass over each block; the r1 and r2 entries are reused by every later window length and by r1+r2 (57 entries in `.cache/features/context`).

Exactness: the L=20 r1 feature matrix (key e7a4d911...) from the shared path equals the one cached before the change bit for bit (values and window ids). Unit and runner tests check equality with the slice path for both `fix_distance_loop` values, several window lengths and strides. `VERSION` stays 1; the matrix cache key and run ids are unchanged. Full suite: 306 passed; ruff and ty clean.

Decisions: entries live under `<cache_root>/context` (one npz per block, `NpzContextCache`, runner-only). The block entry covers the whole plan block, so `_extract_blocks` takes block bounds from `plan.blocks` when a context source is given. The key omits rooms, `window_len`, `stride` and `split_hash` (the block range and recording source hash identify the data) and includes the fitted-state hash. No subagent was used; implemented directly. No code review or git, as requested.

## Spec Change Log

## Review Triage Log

## Design Notes

Position `m` of the arrays reads block packets `START+m .. START+m+size-1`, so it is a property of the block alone. A window at offset `o` uses positions `o .. o+n_metric-1`; the largest position any length needs is `B-17`, so one array of `B-16` positions per block serves every window length. Because 2.10 already makes chunking value-neutral, computing from position 0 instead of the first offset must stay bit-identical; a test guards it.

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass
- `uv run ruff check src tests && uv run ty check` -- expected: clean
- `/usr/bin/time -v uv run csi-bench sweep <12-run paper_beamforming sweep>` with fresh `.cache` -- expected: times and max RSS recorded
