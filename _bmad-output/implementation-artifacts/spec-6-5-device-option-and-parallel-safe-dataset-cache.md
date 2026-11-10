---
title: 'Device option and parallel-safe dataset cache'
type: 'feature'
created: '2026-11-10'
status: 'done'
baseline_commit: 'c75bee6839bd73eeec8c3a5695530faa316a941c'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** LWM embedding is pinned to CPU (`DEVICE = "cpu"`), so Story 6.7's Padova (30×) and CSI-Bench (128×) passes would take days. Parallel CSI-Bench sweeps on the Mac failed with `cache file unreadable`.

**Approach:** Add a runner-owned `--device cpu|mps|cuda` / `CSI_BENCH_DEVICE` option that LWM follows, keyed into the packet cache, plus an optional CUDA torch source. Fix the cache failure at its cause: each memmapped recording holds one open file descriptor (438 per CSI-Bench load), so concurrent sweeps with forked workers exhaust the fd limit.

**Cause confirmed (2026-11-10):** `CsiBenchHp().load()` holds 442 fds (4 baseline) and frees them on release; with `RLIMIT_NOFILE` ≤ 442 it raises exactly `DataError: …: cache file unreadable: …/.cache/v1/<sha>.npy` (≥ 443 loads). Writes are already atomic (`mkstemp` + `os.replace`), so no write race exists. The failed Mac run died 0.1 s after start, in `source.load()`.

## Boundaries & Constraints

**Always:**
- Device resolution like `resolve_workers`: `--device` > `CSI_BENCH_DEVICE` > `cpu`. Unknown value or unavailable backend (`torch.backends.mps.is_available()`, `torch.cuda.is_available()`) → `ConfigError`, exit 2, before any run starts. `run`, `sweep` and `rescore-runs` all resolve it.
- Device is runtime-only: not in `ExperimentConfig`, never in `run_id`. Recorded in `provenance.json` (`device`, `runtime.device`).
- Runner sets `torch.set_default_device(device)`; for `cuda` also `CUBLAS_WORKSPACE_CONFIG=:4096:8` (setdefault) so deterministic mode works. Determinism flags unchanged.
- LWM follows `torch.get_default_device()`: model and token batches on it, embeddings back to CPU numpy float32. Adapters never choose a device.
- Packet cache key gains the device **only when it is not `cpu`**, so existing CPU entries (all `tor_vergata` LWM caches) stay valid and CPU/GPU embeddings never mix.
- Torch sources: plain `uv sync` keeps today's CPU wheels on Linux and macOS (macOS wheels already include MPS; the lock's existing torch entries must not change). An opt-in uv extra/group `cuda` on Linux selects the `cu126` index (widest driver support of the indexes that have 2.14.1: cu126, cu130, cu132).
- Cache fix: cached arrays are mapped read-only without keeping a file descriptor open (one fd only during open). Same in `padova` and `csi_bench_hp`; cache layout, keys and `VERSION` unchanged, so no reconversion. fd/mapping exhaustion (`EMFILE`, `ENFILE`, `ENOMEM`) raises a `DataError` naming the cause instead of reconverting and reporting "unreadable".

**Never:**
- No device-dependent code in heads, poolers, classical extractors or datasets.
- No change to CPU results or CPU cache keys; no raise of fd limits as the fix.
- No new sweeps, plans or experiment files (Stories 6.6, 6.7).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Default | no flag, no env | device `cpu`, keys and results as before | N/A |
| Env | `CSI_BENCH_DEVICE=mps`, flag `--device cpu` | flag wins: `cpu` | N/A |
| Bad value | `--device gpu` | — | `ConfigError`, exit 2 |
| Unavailable | `--device cuda`, no CUDA | — | `ConfigError`, exit 2 |
| Low fd limit | `RLIMIT_NOFILE` 64, 100 cached recordings | loads; holds ~0 extra fds | N/A |
| Two processes | 2 concurrent cold loads of one root, low fd limit | both succeed, identical CSI | N/A |
| Truly exhausted | no fd free at open | — | `DataError` naming fd/mapping exhaustion |

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/run.py` -- `DEVICE` const (l.63), `resolve_workers` (pattern to copy), `_setup_runtime` (l.340), `runtime_info`, `collect_provenance` (`"device": DEVICE`), `_execute` (l.757), `_extract`/`_pool_packets` (packet key call), `run`, `run_sweep`. Thread `device` through these.
- `src/csi_bench/runner/rescore.py:152` -- calls `_execute`; resolve device like workers.
- `src/csi_bench/runner/cli.py` -- add `--device` to `run`, `sweep`, `rescore-runs` (mirror `--workers` help).
- `src/csi_bench/config/experiment.py:412` `packet_cache_key` -- add `device: str = "cpu"`; add `"device"` to the hashed dict only when not cpu.
- `src/csi_bench/adapters/extractors/lwm/adapter.py` -- `_get_model` (`.to(device)`), `_embed` (`from_numpy(...).to(...)`, `pooled.cpu().numpy()`). `load_model` loads to CPU via meta+assign; keep it.
- `src/csi_bench/adapters/datasets/csi_bench_hp.py`, `padova.py` -- `_open_cache` uses `np.load(mmap_mode="r")` (holds a dup'd fd per array, Python 3.12 `mmap` has no `trackfd`). Replace with a shared fd-free mapper.
- New `src/csi_bench/core/npy_map.py` -- `map_npy(path) -> np.ndarray`: parse the `.npy` header (`np.lib.format`), `mmap` the whole file via `ctypes` libc (`PROT_READ`, `MAP_SHARED`), close the fd, `np.frombuffer` at the header offset, read-only, `munmap` via `weakref.finalize` on the buffer owner. Reject Fortran order / object dtype with `ValueError`. Core is the only shared place adapters may import (import-linter `independence`).
- `pyproject.toml` / `uv.lock` -- `[tool.uv.sources]` torch/torchvision, `[[tool.uv.index]]`; add `pytorch-cu126` (`explicit = true`) and the opt-in mechanism (extra or dependency group with `conflicts`), then `uv lock`.
- Tests: `tests/test_run.py:129,351` (provenance device), `tests/test_packet_cache.py:52` (key helper), `tests/test_lwm.py`, `tests/test_csi_bench_hp.py`, `tests/test_padova.py` (synthetic `.mat` writers to reuse).

## Tasks & Acceptance

**Execution:**
- [x] `tests/test_dataset_fds.py` -- first, write the reproduction: subprocesses with `RLIMIT_NOFILE` lowered (64) loading 100 synthetic cached recordings for `csi_bench_hp` and `padova`, and two concurrent cold loads of one root; confirm it fails before the fix.
- [x] `src/csi_bench/core/npy_map.py` + `tests/test_npy_map.py` -- fd-free mapper; tests: values equal `np.load`, read-only, fd count unchanged after mapping, unmapped after release, bad header → `ValueError`.
- [x] `src/csi_bench/adapters/datasets/csi_bench_hp.py`, `padova.py` -- use `map_npy` in `_open_cache`; exhaustion errnos → `DataError`.
- [x] `src/csi_bench/runner/run.py`, `rescore.py`, `cli.py` -- `resolve_device`, thread device, provenance, cuBLAS env; tests in `tests/test_run.py` for each matrix device row, `run_id` independence, provenance.
- [x] `src/csi_bench/config/experiment.py` -- device in packet key; tests: cpu key equals today's key (pinned literal), mps/cuda keys differ.
- [x] `src/csi_bench/adapters/extractors/lwm/adapter.py` -- follow default device; test: CPU output unchanged; GPU-vs-CPU test (`allclose`, `atol` 1e-4) skipped unless mps/cuda is available.
- [x] `pyproject.toml`, `uv.lock` -- CUDA opt-in; check `uv lock` keeps the CPU torch entries and `uv sync --dry-run` with the cuda option resolves `+cu126` on Linux.
- [x] `--device` CLI help + `resolve_device` docstring (no README documents run options) -- `CSI_BENCH_DEVICE`, the CUDA sync command, reproducibility rule (same device → same metrics; across devices only `run_id` is equal).

**Acceptance Criteria:**
- Given the same config run on `cpu` then `mps`, when both finish, then `run_id` is equal, the packet cache holds separate entries, and each `provenance.json` names its device.
- Given an LWM CPU embedding pass on `tor_vergata`, when timed before and after, then the CPU time is recorded in Implementation Notes for 6.7 planning (GPU timing is a manual check on the Mac / a CUDA box).
- Given the finished story, when the full check suite runs, then everything passes and no CPU result or existing cache entry changes.

### Review Findings

Code review 2026-11-10 (Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor; diff vs `c75bee6`, `uv.lock` excluded).

- [x] [Review][Patch] Finished run is reused regardless of device: keep skipping, but warn [src/csi_bench/runner/run.py:817] — resolved 2026-11-10: GPU is only for new LWM runs (Padova, CSI-Bench, maybe Epic 7); existing runs must never be rerun. When a finished run's recorded device differs from the requested one, write a warning to stderr naming both devices (results kept from the recorded device; use another `--runs` root or `sweep --force` to rerun) and skip as today. Applies to `run` and `sweep`.
- [x] [Review][Patch] `rescore-runs` uses each run's recorded device [src/csi_bench/runner/rescore.py:117,160] — resolved 2026-11-10 (option 1): without `--device`, each run is rescored on the device in its `provenance.json` (`cpu` when absent); an explicit `--device` overrides. A recorded device that is unavailable here is a per-run error.
- [x] [Review][Patch] No CPU-runnable test that LWM moves to a non-cpu default device; GPU test does not run under the runner's determinism flags [tests/test_lwm.py:256,273] — reverting `_get_model` to ignore the device passes every non-skipped test; `test_model_follows_the_default_device` only proves the cpu no-move path. Add a `meta`-device test on a freshly loaded model, and enable `use_deterministic_algorithms(True)` (+ `CUBLAS_WORKSPACE_CONFIG` for cuda) in `test_gpu_matches_cpu`.
- [x] [Review][Patch] `CUBLAS_WORKSPACE_CONFIG` leaks out of `test_cuda_sets_the_cublas_workspace` [tests/test_run.py:1811] — `delenv(raising=False)` on an absent var registers no undo, so `_setup_runtime`'s `setdefault` stays for the session.
- [x] [Review][Patch] Exhaustion message blames fds/mappings for any ENOMEM [src/csi_bench/adapters/datasets/csi_bench_hp.py:134, padova.py:147] — the `_load` wrapper catches ENOMEM from `read_bytes`/conversion too; word it as fd, mapping or memory exhaustion.
- [x] [Review][Patch] Dead `"exhaust"` branch in the subprocess script [tests/test_dataset_fds.py:39] — no test passes it.
- [x] [Review][Patch] `_DEVICE_HELP` string split mid-sentence across literals [src/csi_bench/runner/cli.py:23].
- [x] [Review][Patch] Mention `UV_NO_SYNC=1` alongside `uv run --no-sync` for the cuda setup [src/csi_bench/runner/cli.py:23, pyproject.toml] — a plain `uv run` after the cuda sync silently reinstalls CPU torch (fails later as ConfigError).
- [x] [Review][Defer] Fork pools after CUDA/MPS init in the same process [src/csi_bench/runner/run.py:313,329] — deferred: unverified, would be medium. A sweep that runs an LWM config on mps/cuda then a forked (`--workers > 1`) classical/block-context config forks after the accelerator runtime started its native threads (invisible to `_fork_is_safe`); macOS Metal/ObjC is fork-unsafe. Settle with such a sweep on the Mac (`--device mps --workers 4`).

**Rejected:**
- `false` — npy_map negative `count` from int64 overflow: `frombuffer` with a negative count reads the whole buffer and the following `reshape` raises `ValueError`, which `_open_cache` treats as unreadable → reconvert.
- `false` — npy_map breaks Windows imports: `run.py` already imports `resource` unconditionally; the project is POSIX-only.
- `false` — `cuda` vs `cuda:0` re-moves the model every call: `torch.get_default_device()` returns the indexed device, and `Module.to` is a no-op anyway.
- `false` — user-preset non-deterministic `CUBLAS_WORKSPACE_CONFIG`: fails loudly on an explicit user misconfiguration.
- `false` — seeds from two devices mixed on resume: `_completed_seeds` reuses nothing unless status is `done`, so a partly finished run reruns every seed.
- `low` — `glob` swallows EMFILE (verified: returns `[]`) giving "no recordings" instead of the exhaustion message: the loader now holds no fds, so this needs outside exhaustion; fix adds a probe guard.
- `low` — `test_gpu_matches_cpu` leaves a cpu device context installed: harmless.
- `low` — report/index do not show the device: one dir per `run_id`, covered by the first decision.
- `low` — provenance lacks GPU name / CUDA version; cache key does not separate GPUs: the feature cache is per machine; adds fields.
- `low` — exhaustion wrapper duplicated in both adapters: refactor only, no current divergence.
- `low` — munmap not verified on macOS (`/proc/self/maps` skip): mechanism is platform-independent.
- `low` — device names exact (`CUDA`, `cuda:1` rejected): matches the spec's `cpu|mps|cuda`.
- `low` — device enters the key for any `per_packet` extractor, not only torch ones: only LWM is per_packet.
- `low` — `uv sync --no-default-groups` / `pip install .` fall back to PyPI torch: unusual paths; plain `uv sync` unchanged.
- spec edit — lock entries' markers changed despite "must not change" (versions/wheels/hashes kept, noted in Implementation Notes); spec status/triage-log bookkeeping.

## Spec Change Log

## Design Notes

Why not a pack file (all recordings in one `.npy`): it doubles disk (7 GB CSI-Bench, 14 GB Padova) and Padova variants load different subsets of one cache. Why not raising `RLIMIT_NOFILE`: macOS also has a system-wide `kern.maxfiles`, and parallel sweeps × forked workers × 438 fds hit it regardless. The `ctypes` mapper keeps the cache untouched and makes a load cost O(0) persistent fds.

## Verification

**Commands:**
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass (pre-existing notebook/`ty` diagnostics from 6.4 excepted)
- `uv run python -c "…CsiBenchHp().load()…"` with `RLIMIT_NOFILE` 64 -- expected: 438 recordings, fd count ≈ baseline

**Manual checks:**
- Mac: `uv run csi-bench run <lwm config> --device mps`, compare time and macro-F1 to `cpu`. CUDA box: `uv sync` with the cuda option, then `--device cuda`.

## Implementation Notes

- **Reproduction (before the fix):** `tests/test_dataset_fds.py` failed for both adapters with exactly `DataError: …rec_059.mat: cache file unreadable: …/.cache/v1/<sha>.npy` at `RLIMIT_NOFILE` 64. With `map_npy` it passes; real CSI-Bench HP loads 438 recordings at `RLIMIT_NOFILE` 64 with 3 fds before and after (Padova: 105, 3 fds). The "truly exhausted" row is tested in-process (patched `map_npy` raising EMFILE/ENFILE/ENOMEM): at a fixed low limit the fixed loader never runs out, and lowering the limit to zero free fds breaks `glob` first (it swallows EMFILE).
- **LWM CPU timing (tor_vergata, 19 recordings, 63 272 packets, all 4 antennas, global_rescale, 1 torch thread, deterministic, run concurrently on a 16-core Linux x86 box):** before 656.0 s (96 packets/s), after 658.9 s (96 packets/s), so no change. For 6.7: about 10.4 ms per packet per thread on CPU. GPU timing is still a manual check.
- **CUDA opt-in mechanism:** dependency groups, not an extra. uv cannot express "CPU unless an extra is on" for a base dependency, so `cpu` (torch/torchvision from the CPU index) is a default group and `cuda` (cu126 index, Linux only) conflicts with it: `uv sync --group cuda --no-group cpu`, then `uv run --no-sync …` (a plain `uv run` re-syncs the default `cpu` group). Plain `uv sync` is unchanged (dry run only rebuilds the project). The existing CPU torch/torchvision lock entries keep their versions, sources, wheels and hashes. Their dependency markers gained group conditions, and the lock adds `+cu126` entries plus a PyPI torch fallback, used only when neither group is enabled.
- **Cache-hit tests:** `test_padova.py`/`test_csi_bench_hp.py` used to assert `np.memmap`. They now assert `not flags.owndata`, because `map_npy` returns a plain read-only ndarray.

## Review Triage Log

- 2026-11-10: review skipped at the user's request (no code review); no findings recorded.
- 2026-11-10: code review (4 layers): 2 decisions resolved (finished run on another device is kept with a warning; rescore uses the recorded device), 8 patches applied, 1 deferred (fork after accelerator init), 16 rejected. Checks: ruff, format, lint-imports pass; pytest 661 passed, 1 skipped; ty 12 pre-existing diagnostics, unchanged.
