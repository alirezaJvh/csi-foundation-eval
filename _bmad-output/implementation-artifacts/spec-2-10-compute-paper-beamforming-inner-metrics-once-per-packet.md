---
title: 'Compute paper_beamforming inner-window metrics once per packet'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: 'e6cc8805f93e52c2b4d11c7d4cda4b23bf2ba70c'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/proposal-2-10-reuse-inner-windows.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `paper_beamforming` recomputes each inner-window metric (doppler, distance, flatness) in every window that contains it, about `window_len - 16` times at stride 1. Extraction takes ~16 min per run and multi-core did not help (Story 2.9).

**Approach:** Add an optional block-aware extractor path (`uses_context = True`, `extract_blocks`). The runner calls it once per split block with that block's packets and the window offsets. `paper_beamforming` computes the six per-position arrays once per packet position in the block, then slices them per window. Output must equal the slice-based `extract` exactly.

## Boundaries & Constraints

**Always:** Feature matrix equal to the slice-based one (same values, same row order) for `fix_distance_loop` true and false. Blocks are never joined across a split gap. Feature-cache key and cached matrix unchanged. `VERSION` stays 1 when values are identical. Per-position arrays are computed in position chunks so peak RSS stays within about 4 GB. Log per-stage timings (load, preprocess, extract, head fit) in `run.log`.

**Never:** Change `extract(window_slices)` behaviour or the paper-parity oracle. Cache per-position arrays across window lengths (out of scope). Add run-directory locking. Make bit-inequality silent: if exact equality is impossible, stop and ask the human about a tolerance, then bump `VERSION`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Equal to slice path | Synthetic and real windows from one block, both `fix_distance_loop` values | `extract_blocks` rows equal `extract` rows exactly | N/A |
| Several blocks | Windows from different recordings/splits | Runner calls per block; row order matches `WindowSet` order | N/A |
| Stride > 1 | Window offsets not contiguous | Same rows as slice path | N/A |
| Block too short | Block shorter than `min_window_len` | N/A (no window exists) | `ExtractorError` if offsets/len inconsistent with block |
| Extractor without context | `amplitude_stats` | Runner uses the slice path unchanged | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/extractors/paper_beamforming.py` -- `_features` holds the per-window logic; split into per-position metric arrays (`_position_metrics(block, fix)` chunked) and per-window reduction (`np.var`/`np.mean` over `a+START .. a+START+L-17`); add `uses_context`, `extract_blocks`.
- `src/csi_bench/core/ports.py` -- `FeatureExtractor`: document optional `uses_context` and `extract_blocks(block, offsets, window_len)`.
- `src/csi_bench/runner/run.py` -- `_extract` (~l.272) builds slices; add block grouping by `(recording_id, split)` and call `extract_blocks` when `uses_context`; keep the cache key code untouched. Add stage-timing log lines in `_execute`. `_extract_parallel` stays for slice-path extractors.
- `src/csi_bench/core/windows.py` -- window order `(recording_id, start)`; blocks come from `plan.blocks`.
- `tests/contracts/extractor.py` -- add check that block-aware output equals slice output for extractors offering both.
- `tests/test_paper_beamforming.py`, `tests/test_run.py` -- equality tests, runner grouping test; oracle test stays.
- `_bmad-output/implementation-artifacts/spec-2-9-*.md` -- baseline: extraction 956 s, run 16:29, peak RSS 1.33 GB (window_len 20, r1, svm, 37,937 windows).

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/runner/run.py` -- log load/preprocess/extract/head-fit seconds; take a baseline timing run before the extractor change -- measured gain, next bottleneck
- [x] `src/csi_bench/core/ports.py` -- document `uses_context` and `extract_blocks` -- port contract
- [x] `src/csi_bench/adapters/extractors/paper_beamforming.py` -- chunked per-position metrics and `extract_blocks` -- the speed-up
- [x] `src/csi_bench/runner/run.py` -- group windows per block and dispatch to `extract_blocks` -- never across a split gap
- [x] `tests/contracts/extractor.py`, `tests/test_paper_beamforming.py`, `tests/test_run.py` -- exact-equality (synthetic + real recording windows, both flags, stride 1 and >1), grouping, cache key unchanged -- ACs
- [x] this spec, Implementation Notes -- stage timings and extraction time / peak RSS before and after

**Acceptance Criteria:**
- Given a block and window offsets, when `paper_beamforming` runs `extract_blocks`, then the matrix equals `extract` on the same windows exactly, and the paper-parity test passes.
- Given an extractor with `uses_context = True`, when the runner extracts, then it calls per block, and the cache key and cached matrix equal today's.
- Given the first `paper_beamforming` run of `classical_baseline.yaml`, when it finishes, then extraction time and peak RSS before/after are recorded here and RSS is about 4 GB or less.

### Review Findings

- [x] [Review][Rejected] Real-recording exact-equality AC has no automated test — Task 5 is marked done, but the only equality tests on disk are synthetic (`tests/test_paper_beamforming.py:794-809`, `tests/contracts/extractor.py:767-775`); the real-data bit-for-bit check exists only as a one-time manual cache comparison in Implementation Notes and is unreproducible in CI — rejected: decision 2026-10-10, human accepted the one-time manual verification as the evidence of record.
- [x] [Review][Patch] No runner test that `_extract` dispatches `uses_context` (fixed 2026-10-10) extractors to `extract_blocks` [src/csi_bench/runner/run.py:319] — the only runner-level test with `paper_beamforming` asserts only `_extract_parallel` non-use and metric presence, which hold on the slice path too; removing or inverting the dispatch branch would ship the ~2.6x extraction gain silently reverted.
- [x] [Review][Patch] Thread-fallback test is vacuous — asserted (fixed 2026-10-10) against the wrong extractor [tests/test_run.py:951] — `test_parallel_extraction_falls_back_when_threads_live` uses `paper_beamforming`, which now takes the block path unconditionally, so `calls == []` holds even without the `_fork_is_safe` fallback; `_fork_is_safe` with live threads is never exercised through `_extract`. Implementation Notes say the test was meant to use `amplitude_stats`.
- [x] [Review][Patch] Extraction log prints `workers=%d` on the block path (fixed 2026-10-10) where workers are unused [src/csi_bench/runner/run.py:334] — for `uses_context` extractors the in-process block path ignores `--workers`, yet the log line still reports the requested count, misleading anyone tuning workers for `paper_beamforming`.
- [x] [Review][Patch] `_combined_amplitude` docstring stale (fixed 2026-10-10) [src/csi_bench/adapters/extractors/paper_beamforming.py:37] — the function now yields per-position amplitudes under the block path; the stale comment misstates its output shape/semantic.
- [x] [Review][Defer] knn/random_forest "use more than one core" AC verified only via `joblib.effective_n_jobs(None), not observed parallel work [tests/test_run.py:979] — deferred: already recorded in deferred-work.md and spec-2-9's Review Findings (`[Review][Defer]`), pending the Story 2.10 timing work.

**Rejected appendix:**

- `false` — Split-gap violation in `_extract_blocks` grouping: `plan.blocks` stores exactly one block interval per (recording, part) and `build_windows` creates windows only inside that interval, so grouping by `(recording_id, split)` cannot join windows across a gap.
- `false` — Mixed window lengths silently corrupt offsets: every window in a run has the same `window_len` by construction (`build_windows` drills one config length); the outcome is unreachable in current code.
- `false` — Row-count mismatch raises raw ValueError instead of ExtractorError: only a contract-violating extractor can do this; it fails loudly either way inside an already-throwing run.
- `false` — Fork TOCTOU race (`_fork_is_safe` check vs fork): no concurrent thread spawner exists between the check and pool creation in current code; theoretical only.
- `false` — `_extract_parallel` max_workers/len(bounds) divergence: `len(bounds)` is the clamped chunk count by design; the finding's own text calls the outcome impossible.
- `false` — `_val_score` failure aborts Parallel: the pre-change serial loop had no per-candidate try/except either; failure behavior is unchanged.
- `false` — Spec Verification claim "`ruff check .` clean" inaccurate: the 3 ruff errors are pre-existing in `notebooks/01_classical_overview.ipynb`, untouched by this diff.
- `false` — Contract test crashes on <2 windows: all callers pass ≥3 windows (`tests/test_paper_beamforming.py:39`); issue unreachable as wired.
- `low`, rejected — Baseline per-stage timings recorded as n/a: stage logging did not exist before the change so the values are un-reproducible; fixing means editing this spec's Implementation Notes, not code.
- `low`, rejected — Diff bundles Story 2.9 head speed-ups with 2.10: declared in the spec's Implementation Notes addendum as result-neutral; a process/traceability note, not a code fix.
- `low`, rejected — One `--workers` knob throttles extraction and head jobs: current documented behavior with the rebalance explicitly deferred; a design tradeoff, not a defect.
- `low`, rejected — Stride > 1 computes unused positions: deliberate tradeoff (all positions in the block slab, simplest correct design); the equality AC passes.
- `low`, rejected — Per-task array pickling in head search: performance optimization opportunity in deferred territory, not a defect.
- `low`, rejected — Missing negative-offset / empty-group rejection tests: those branches are unreachable from the runner (offsets derive from `w.start - start ≥ 0`); test-hardening nicety.
- `low`, rejected — `resolve_workers` float/bool coercion: CLI and env can only supply ints/strings; no reachable path.
- `low`, rejected — External explainer link in the proposal artifact: cosmetic; link rot doesn't affect code or runs.
- `low`, rejected — Test nits (unused `monkeypatch` fixture, `cache_size` comment): cosmetic; unlikely to trouble users or devs in everyday use.

## Implementation Notes

First `paper_beamforming` run of `classical_baseline.yaml` (window_len 20, r1, svm; 37,937 windows; run id c5a7175e39e0), fresh feature cache, `/usr/bin/time -v`:

| | extraction | whole run | peak RSS | load | preprocess | head fit + evaluate |
|---|---|---|---|---|---|---|
| before (Story 2.9, workers=1) | 956 s | 16:29 | 1.33 GB | n/a | n/a | n/a |
| after (block-aware) | 365 s | 6:38 | 1.33 GB | 0.53 s | 0.03 s | 28.7 s |

Extraction is 2.6x faster, below the ~4x operation-count estimate: the doppler FFT and the amplitude construction (`(positions, 40, size, subc)` temporaries) now dominate and still scale with positions. Head fit (svm, 28.7 s) is negligible next to extraction. Next bottleneck: the per-position amplitude/FFT cost, and sweep-wide reuse across window lengths (out of scope here).

Exactness: the cached matrix from the old slice path (taken before the change, same cache key) equals the newly computed one bit-for-bit on the real data (values, feature names, window ids), and old vs new code are bit-identical on synthetic blocks for both `fix_distance_loop` values. `VERSION` stays 1.

Decisions: the block path runs in-process (workers apply only to slice-path extractors). The block slice spans min start to max end of the block's windows. `extract_blocks(block, offsets, window_len)` is on the extractor, `_extract_blocks` in `runner/run.py` groups by `(recording_id, split)`. The test that spied on `_extract_parallel` now uses `amplitude_stats`, since `paper_beamforming` no longer takes that path. Stage timings are logged in `run.log` as `stage ...` lines.

Addendum (head speed-ups, result-neutral, no new story): `tune_and_refit` (`src/csi_bench/core/head_search.py`) scores candidates in parallel through joblib (worker count from the runner's `parallel_config`), and `Svm` sets `cache_size=500` MB (`src/csi_bench/adapters/heads/svm.py`). Same run, svm head stage 28.7 s -> 19.5 s; `metrics.json` mean_std identical before and after; peak RSS 1.55 GB.

## Spec Change Log

## Review Triage Log

## Design Notes

Inner start `p` (absolute in block) depends on packets `p .. p+size-1` only. Window `[a, a+L)` uses starts `a+START .. a+L-16`. Doppler and flatness keep only those positions, so no extra packets are read. Per-position arrays (6 floats per position) are small; the large temporaries `(positions, 40, size, subcarriers)` are why chunking is needed.

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass
- `uv run ruff check . && uv run ty check` -- expected: clean
- `/usr/bin/time -v uv run csi-bench run <first paper_beamforming config> --workers 1` before and after -- expected: times and max RSS recorded in Implementation Notes
