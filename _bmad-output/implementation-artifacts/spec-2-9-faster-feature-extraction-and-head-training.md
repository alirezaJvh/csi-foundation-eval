---
title: 'Faster feature extraction and head training'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: 'e6cc8805f93e52c2b4d11c7d4cda4b23bf2ba70c'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `paper_beamforming` extraction and the `random_forest`/`knn` heads run on one core, so the 216-run baseline sweep takes days on a 16-thread machine.

**Approach:** The runner owns one worker count. It splits a cache-missing extractor's window slices into contiguous chunks run in a process pool (rows concatenated in order), and fits heads under a joblib context of the same size. Log extraction time and peak memory per run.

## Boundaries & Constraints

**Always:** Worker count is runner-owned (not in the config, so not in `run_id` or the cache key): `--workers N` on `run` and `sweep`, else env `CSI_BENCH_WORKERS`, else `min(4, os.cpu_count())`; `1` keeps today's in-process path with no pool. Results for any worker count equal the `1` result (same values, same row order; same metrics for a fixed seed). Chunks are contiguous and a pool is started per extractor call, not per run. Extractor code is unchanged (purity per input already guarantees chunking is safe).

**Never:** Running several sweep processes (needs a run-directory lock), reusing inner windows across windows (needs extractor-interface change), changing extractor/head `Params`, `VERSION`, or any result value, loading all window slices into worker payloads by copy (use fork so recordings are shared).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Parallel extraction | `paper_beamforming`, N workers | Matrix equals `workers=1` exactly | N/A |
| Few windows | fewer windows than workers | Uses fewer chunks; still ordered | N/A |
| Bad worker count | `--workers 0` or non-integer env | Rejected before any run | exit 2, message names the source |
| Head fit | `random_forest`/`knn`, N workers | Metrics for a fixed seed equal `workers=1` | N/A |
| Interrupted sweep | run left `status: running` | Rerun on the next `sweep` | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/run.py` -- `_extract` (the single extractor call, line ~190), `_run_seed` (head fit), `_execute`/`run`/`run_sweep` signatures, `_sweep_action` (`running` status already yields `run`, since `_completed_seeds` needs `done`); add worker resolution, chunked pool extraction, joblib context, timing and peak RSS in `run.log`.
- `src/csi_bench/runner/cli.py` -- add `--workers` to `run` and `sweep`.
- `src/csi_bench/adapters/heads/random_forest.py`, `knn.py` -- sklearn estimators pick the joblib context up when `n_jobs=None`; do not add params.
- `src/csi_bench/adapters/selectors/mrmr.py` -- `n_jobs=1`; leave as is.
- `tests/test_paper_beamforming.py`, `tests/oracles/paper_processing.py` -- parity with the paper port; must still pass.
- `tests/test_run.py`, `tests/test_sweep.py` -- `_make_project`, `main([...])` fixtures to reuse.
- `experiments/classical_baseline.yaml` -- source of the first `paper_beamforming` run to measure.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/runner/run.py` -- `resolve_workers`, chunked fork-pool extraction, `joblib.parallel_config` around head fits, log extraction seconds and peak RSS -- the speed-up
- [x] `src/csi_bench/runner/cli.py` -- `--workers` on `run`/`sweep` -- command surface
- [x] `tests/test_run.py` -- workers=1 vs N equal feature matrix and metrics (extractor and `random_forest`/`knn`); few-windows case; bad worker count; heads see N jobs -- the I/O matrix
- [x] `tests/test_sweep.py` -- a `running` run is rerun on resume -- spec AC
- [x] this spec, Implementation Notes -- measured extraction time and peak memory before (workers=1) and after for the first `paper_beamforming` run of `classical_baseline.yaml` -- spec AC

**Acceptance Criteria:**
- Given `paper_beamforming` and a set of windows, when extracted with several workers, then the matrix is identical to the single-process result and the paper-parity test passes.
- Given the default worker count, when the baseline's first `paper_beamforming` run finishes, then its peak memory is about 4 GB or less.
- Given `random_forest` or `knn` and a fixed seed, when fitted with several workers, then they use more than one core and the metrics are unchanged.

### Review Findings

- [x] [Review][Patch] Guard fork against a multi-threaded parent — before `_extract_parallel` forks, detect live threads (e.g. `threading.active_count() > 1`) and warn and/or fall back to `workers=1`. [src/csi_bench/runner/run.py:107]
- [x] [Review][Patch] Relabel the children peak-RSS log metric — rename `children_peak_rss_mb` to `max_child_peak_rss_mb` so it reads as the largest single child and note it excludes live workers. [src/csi_bench/runner/run.py:72-76]
- [x] [Review][Patch] Parallel extraction path is never proven to run — `test_parallel_extraction_matches_single_process` spies on `_extract`, not `_extract_parallel`, so the equality assertions still pass if the pool is silently bypassed; the `feature_names`-mismatch branch (`run.py:113-114`) is untested. [tests/test_run.py:875-883]
- [x] [Review][Patch] Sweep worker forwarding and bad-count rejection untested — no sweep test passes `--workers` or asserts the resolved count reaches `_execute`, and `test_bad_worker_count_rejected_before_any_run` drives only `run`. [src/csi_bench/runner/run.py:634,669]
- [x] [Review][Patch] `joblib` is imported directly but not declared in `dependencies` — resolves only transitively through scikit-learn; add it to `pyproject.toml`. [pyproject.toml:6]
- [x] [Review][Patch] `--workers` help misstates the default — help says "default: $CSI_BENCH_WORKERS or 4" while `resolve_workers` returns `min(4, os.cpu_count())`. [src/csi_bench/runner/cli.py:117,121]
- [x] [Review][Defer] knn "more than one core" AC is only checked via `joblib.effective_n_jobs` [tests/test_run.py:931] — deferred: unverified; settle by checking the chosen estimator's algorithm/parallel dispatch or measuring CPU time.
- [x] [Review][Defer] No wall-clock speed-up from multi-worker extraction [spec Implementation Notes] — deferred: acknowledged in Implementation Notes and tracked for Story 2.10 in `deferred-work.md`.

**Rejected**

- `false` — Extraction time/RSS logged only on cache-miss: per-run peak RSS is logged unconditionally at `_execute` (`run.py:551-557`); extraction time is only meaningful on a miss.
- `false` — `_peak_rss_mb` divides `ru_maxrss` by 1024 (macOS is bytes): target is Linux-only (`resource` is POSIX; `ru_maxrss` is KiB on Linux), so no harm in the supported environment.
- `false` — `assert _FORK_STATE is not None` stripped under `-O`: the forked child always inherits `_FORK_STATE` set before the pool starts; the state is unreachable, so only the error text changes.
- `false` — `--workers`/`CSI_BENCH_WORKERS` undocumented in README/docs: the repo has no README or `docs/`; argparse help is the documented surface.
- `false` — Implementation-Notes deliverable and proposal file absent: the spec's Implementation Notes are populated and `proposal-2-10-reuse-inner-windows.md` exists on disk (untracked, so absent only from `git diff`, not from the tree).
- `low` — `--workers` value not capped at CPU count: the spec intentionally lets an explicit N be honored; capping would contradict "runner-owned worker count".
- `low` — Extraction log reports requested `workers` even on the single-window fallback: only when `workers>1` with ≤1 window; negligible.
- `low` — `_extract_parallel` lacks a docstring and uses `Any` types: style, no named harm.
- `low` — `resolve_workers` try/except duplicated in `run`/`run_sweep` (and `from None` inconsistency): 6-line duplication, no demonstrated drift harm.
- `low` — `test_running_status_is_rerun_on_resume` uses a raw `str.replace('"done"', '"running"')`: status.json has a single `"done"` token; test-only.
- `low` — `test_chunk_bounds_contiguous_and_capped` pairs bounds with `strict=False`: correctly pairs adjacent bounds; production `strict=True` is a separate edge-length invariant.
- `low` — `deferred-work.md` proposes Story 2.10 not present in `sprint-status.yaml`: planning-artifact nit, not a code defect.
- `low` — No row-alignment check across chunks in `_extract_parallel`: a wrong row count breaks downstream length/indexing loudly; extractors are pure per input.

## Implementation Notes

Measured on the first `paper_beamforming` run of `classical_baseline.yaml` (sweep index 18: window_len 20, r1, svm, no selection; 37937 windows; run id c5a7175e39e0), each in a fresh cache, via `/usr/bin/time -v`:

| workers | extraction | whole run | peak RSS (`time -v`) | logged peak RSS (self / children) |
|---|---|---|---|---|
| 1 | 956 s | 16:29 | 1.33 GB | 1131 MB / 0 |
| 4 (default) | 989 s | 17:03 | 1.32 GB | 1084 MB / 985 MB |

Peak memory is well under the 4 GB target (workers share recordings via fork).

No wall-clock speed-up on this machine: `paper_beamforming` is memory-bandwidth bound (234 subcarriers, large per-window temporaries). A microbenchmark here shows 4 concurrent processes each run about 3.7x slower on it, and a plain numpy streaming benchmark about 4.7x slower, while cache-resident numpy and pure-Python loops scale linearly. So chunking is correct (bit-identical output, tested) but the gain depends on the host's memory bandwidth; reducing the extractor's temporaries would need an extractor change, which this spec excludes.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass
- `uv run ruff check . && uv run ty check` -- expected: clean
- `/usr/bin/time -v uv run csi-bench run <first paper_beamforming config> --workers 1` and default -- expected: times and max RSS recorded in Implementation Notes
