---
title: 'Runner-owned feature cache'
type: 'feature'
created: '2026-10-08'
status: 'done'
baseline_commit: '5269e4feb127cf9cc1d03ee909a69408f6eefbaa'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-2-3-full-classifier-family-with-seeds.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every run re-extracts features even when only the head changed, which makes classifier sweeps slow and wasteful.

**Approach:** The runner caches each extractor's `FeatureMatrix` under `.cache/features/` with a complete content key (AD-10) and loads it instead of calling the extractor on a hit. Extractors and heads never see the cache.

## Boundaries & Constraints

**Always:**
- Port `FeatureCache` in `core/ports.py`: `get(key) -> FeatureMatrix | None`, `put(key, matrix, window_ids)`. Implementation `adapters/storage/feature_cache.py` (npz holding `values`, `feature_names`, `window_ids`), constructed and called **only** by `runner/run.py`. Default root `.cache/features`, overridable through a `cache_root` argument of `run()` for tests.
- Cache unit is one extractor's matrix over all windows. Key = `sha256_hex` of `{dataset, dataset VERSION, split hash, selection.rooms, preprocessing chain (name, version, affecting params) + hash of the fitted states, extractor name + affecting params + VERSION + class attrs VENDOR_HASH/WEIGHTS_REVISION (null if absent)}`, plus `window_len` and `stride` when the extractor's `granularity` is `per_window`. Built by `feature_cache_key` in `config/experiment.py`. Head, seeds, role never enter the key.
- Atomic writes: temp file in the cache dir, then `os.replace`; a failed write leaves no entry and no `.tmp` file. A hit whose `window_ids` differ from the current `WindowSet`, or an unreadable/corrupt entry, counts as a miss (logged) and is recomputed and overwritten.
- On a hit the extractor is not instantiated or called. The existing finite/non-constant checks still run on the concatenated matrix. `provenance.json` gains `feature_cache: [{extractor, key, hit}]`.
- Deleting `.cache/features` never changes metrics.
- Import contract in `pyproject.toml` forbids every adapter package except `storage` (datasets, splitters, preprocess, extractors, heads) from importing `csi_bench.adapters.storage`, checked by `lint-imports`.

**Never:**
- No cache eviction, locking, size limits, per-seed or head caching, CLI flags, or sweeps (2.7). No changes to extractor/head adapters, windower, splits or `data/`. Extractors never import the port's implementation. No `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Head-only change | run A, then same config with another head/params | second run loads features, extractor spy not called, `hit: true` | N/A |
| Key part changes | other dataset VERSION, split hash, rooms, preprocessing step/order/fitted value, extractor params or `VERSION`, `window_len` or `stride` | key differs, miss | N/A |
| Cold cache | `.cache/features` deleted | features recomputed, metrics identical to the cached run | N/A |
| Corrupt entry | truncated `.npz` | treated as miss, recomputed, entry replaced | N/A |
| Stale windows | entry with different `window_ids` | miss | N/A |
| Failed write | `os.replace` raises | no entry and no temp file left | error propagates, run `failed` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/runner/run.py` -- `_extract` (builds slices, calls extractors, concatenates, checks), `_execute` (fitted states from `run_chain`), `collect_provenance`, `run`; cache lookup/store goes here
- `src/csi_bench/config/experiment.py` -- `resolved_body` shows how versions and `registry.affecting_params` are collected; add `feature_cache_key` beside it
- `src/csi_bench/core/hashing.py` -- `sha256_hex`, `canonical_json`
- `src/csi_bench/core/ports.py` -- add `FeatureCache`
- `src/csi_bench/runner/discover.py` -- walks `adapters` so `adapters/storage` is imported; the cache class is not registered (runner constructs it directly)
- `pyproject.toml` -- `[tool.importlinter]` contracts; add the storage-forbidden contract
- `tests/test_layering.py`, `tests/test_run.py` -- patterns (registry-reset fixture, synthetic recordings, extractor spy via monkeypatch)
- new: `adapters/storage/{__init__,feature_cache}.py`, `tests/test_feature_cache.py`

## Tasks & Acceptance

**Execution:**
- [x] `core/ports.py`, `adapters/storage/feature_cache.py` -- `FeatureCache` port and npz implementation with atomic write, corrupt/stale handling
- [x] `config/experiment.py` -- `feature_cache_key`
- [x] `runner/run.py` -- per-extractor lookup/store in `_extract`, `cache_root` argument, provenance `feature_cache`
- [x] `pyproject.toml` -- import contract forbidding other adapters from importing `adapters.storage`
- [x] `tests/test_feature_cache.py`, `tests/test_run.py`, `tests/test_layering.py` -- every matrix row (spy for hit; key-difference table per key part; cold-cache metric equality; corrupt, stale and failed-write cases), contract check passes

**Acceptance Criteria:**
- Given two runs differing only in head or head params, when the second runs, then its features load from the cache without the extractor being called (spy).
- `ruff check src tests`, `ruff format --check src tests`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

### Review Findings

#### Patch

- [x] [Review][Patch] `_setup_runtime()` runs after preprocessing and feature extraction [src/csi_bench/runner/run.py:341] [src/csi_bench/runner/run.py:370] — determinism, thread count and device are pinned only at line 370, but `run_chain` (341) and `_extract` (355-364) already ran; `provenance.runtime` (395) then advertises `deterministic=True`/`torch_threads=1` for a run whose features were computed unpinned. With the cache now persisting those features, a torch-based extractor's output could be nondeterministic and sticky. Moved `_setup_runtime()` before the first extraction/preprocessing call. **Fixed.**
- [x] [Review][Patch] `FeatureCache`/`PreprocessStep` ports are not the runner's real contract [src/csi_bench/core/ports.py:59] [src/csi_bench/core/ports.py:71] — the runner depended on the concrete `NpzFeatureCache` (`runner/run.py:190`) and `run_chain` took `Sequence[Any]` (`core/preprocess.py:21`). Typed the cache parameter as `FeatureCache` and `steps` as `Sequence[PreprocessStep]`. **Fixed.**
- [x] [Review][Patch] Real-data end-to-end test reads/writes the repo-level default cache [tests/test_run.py:197] — `run(config, tmp_path)` omitted `cache_root`, so after `chdir(ROOT)` it used `ROOT/.cache/features`; a pre-populated entry made the run a cache hit, so the test passed without exercising extraction and leaked state outside `tmp_path`. Now passes `cache_root=tmp_path / "cache"`. **Fixed.**
- [x] [Review][Patch] Cache shape-consistency guard had no test [src/csi_bench/adapters/storage/feature_cache.py:35] — no test stored a readable `.npz` whose `values` dimensions disagree with `feature_names`/`window_ids`; the guard was unexecuted and deleting it broke nothing. Added `test_inconsistent_shapes_is_a_miss`. **Fixed.**

#### Defer

- [x] [Review][Defer] Dataset params are unreachable and absent from run id / cache key [src/csi_bench/config/experiment.py:69] — deferred: pre-existing design from story 2.1 (`dataset: str`; `Params()` constructed with no overrides); not caused by the cache change.

#### Rejected

- `blind-hunter` key omits package versions/runtime flags — `low`: the key matches the spec's AD-10 composition exactly; adding library versions is a spec change and complexity the design deliberately avoids (adapters bump `VERSION`).
- `blind-hunter` `granularity == "per_window"` geometry rule mis-keys other extractors — `false`: `per_recording` is not a valid granularity (contract is `per_packet | per_window`), production always calls `get(key, window_ids)`, and the behavior matches the spec.
- `blind-hunter` `whole_recording` guard reads raw params — `false`: every registered step defaults `fit_scope="train"` and explicit values are caught; no adapter defaults to `whole_recording`.
- `blind-hunter` provenance omits rooms/window_len/stride/extractors/head — `false`: the full resolved config is saved as `config.yaml` in the run directory; provenance need not duplicate it.
- `blind-hunter` `PreprocessStep` omits `fit_scope` — `false`: `fit_scope` is an optional attribute (`getattr(step, "fit_scope", "train")`), and only `recording_mean_norm` defines it; declaring it required on the protocol would wrongly reject `antennas`/`global_rescale`.
- `blind-hunter` broad `except`/no unlink in `get` — `low`: the entry is overwritten on the next `put`, and the guarded block is a 3-line read; narrowing the set adds complexity for no everyday harm.
- `blind-hunter` no cache schema version — `low`: forward-compat concern only; no everyday defect.
- `blind-hunter+edge-case-hunter` `_completed_seeds` assumes JSON objects — `low`: the tool writes those files itself as objects, a non-object needs external tampering, and the loud failure still maps to exit 2.
- `blind-hunter` duplicated preprocessing-step key derivation — `low`: developer-only duplication; the two builders differ intentionally (run id excludes fitted state) and no divergence is demonstrated.
- `blind-hunter` import contract enumerates adapter packages — `low`: the spec itself enumerates exactly datasets/splitters/preprocess/extractors/heads.
- `blind-hunter` run id uses plan filename stem — `low`: `write_split_plan` guarantees filename == `split_hash` and `load_split_plan` validates content; only manual tampering diverges.
- `blind-hunter+edge-case-hunter` unwrapped `sha256_hex` on non-JSON fitted state — `false`: the error is caught by `_execute`'s boundary and recorded as a failed run (`run.py:408-419`); it does not escape, and a loud failure on an unreachable state is correct.
- `acceptance-auditor` `get(key, window_ids)` deviates from the spec's port signature — rejected: the added parameter is required to satisfy the spec's own stale-window AC; the only fix is to edit the spec.
- `acceptance-auditor` `put` before finite/constant validation — `low`: checks still run on every hit on the concatenated matrix; a bad matrix yields the same `ExtractorError`, and reordering needs per-extractor validation duplication.
- `acceptance-auditor` `FakeRec` uses invalid `granularity="per_recording"` — `low`: the `!= "per_window"` branch is exercised identically to a real `per_packet` extractor; no per-packet extractor exists yet.
- `edge-case-hunter` plan resolved twice (run id then `_execute`) — `low`: negligible TOCTOU race with no user impact.
- `edge-case-hunter` duplicate `selection.rooms` allowed — `low`: duplicates have no behavioral effect (`set()` is used downstream); only a cosmetic extra run id.
