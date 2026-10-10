---
title: 'Paper beamforming feature family'
type: 'feature'
created: '2026-10-08'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Classical experiments only have `amplitude_stats`; the paper's beamforming features (Doppler integral, CSI distance, flatness) cannot be used alone or combined with it.

**Approach:** Port `data/csi_proc_python/processing.py` as the per-window extractor `paper_beamforming` (channels 0 and 1, 40 weights, inner windows 8/16/1, param `fix_distance_loop`), verified against a test-only oracle that reproduces the script's per-window computation, with two documented deviations (metrics window spans all inner windows; `fix_distance_loop` pair loop). `processing.py` stays unchanged. Several families in `extractors` are already concatenated by the runner; add a uniqueness check on the names.

</frozen-after-approval>

## Implementation Notes

Decisions (not user-visible gaps):
- One output row per outer window of `window_len` L packets. The script's "metrics window" W2 is all inner windows of the 16-packet method: W2 = L − 16 (the script starts inner windows at packet 1, kept verbatim). So the paper's W2 = 40 is `window_len` 56, within `gap_packets` 64. Methods 1 and 3 use their first W2 inner windows, as the script does.
- Columns: `paper_beamforming.{doppler_theta_var, doppler_integral_mean, distance_theta_var, distance_mean, flatness_theta_var, flatness_mean}` (script `feature1..6`).
- `min_window_len = 18` (16 lookback + 1 start offset + W2 ≥ 2, so variances are not constant). `requires = {phase: True, n_antennas: 2}`.
- `fix_distance_loop: true` averages distances over unique pairs `next > sample`; `false` keeps the script's double loop.
- Computation in float64/complex128 so single-window and batched output agree exactly.
- `min_window_len` is checked in `validate_registry` (config validation) and removed from the runner.
- `tests/oracles/paper_processing.py` holds the verbatim per-window loop; test compares both distance modes.

## Review Triage Log

- Oracle is a hand transcription, not the real script: low, the story prescribes a verbatim copy since `processing.py` is not importable; deviations are in the oracle docstring. Rejected.
- Default `fix_distance_loop=false` undocumented: low, patched with a comment on the param.
- `requires.phase` not enforced: low, the runner does not enforce it for any extractor. Rejected.
- Memory use unbounded: maybe-false, deferred.
- `fit_scope` check on raw params: pre-existing, deferred.
- `min_window_len` AttributeError / removed runner guard: false, the port requires the attribute and `load_config` calls `validate_registry`.
- Matrix cached before checks: pre-existing, deferred.
- Missing boundary test at `min_window_len` and complex64 input: medium, patched.
- Cache key vs stride/rooms test gap: pre-existing from 2.4, rejected.

### Review Findings

_Code review (2026-10-08) of the cumulative uncommitted change (stories 2.2–2.5) at HEAD `5269e4f`; story 2.5 verified against this spec. Layers: Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor._

**Decision needed**
- [ ] [Review][Decision] Oracle and feature aggregation deviate from the paper's fixed W2=10 sliding metrics window — The frozen Intent says the oracle "copies the script's computation verbatim", but `tests/oracles/paper_processing.py:6-8` documents two deliberate deviations (the metrics window spans all `len(theta_star_out2)` inner windows instead of sliding a fixed 10; `fix_distance_loop` swaps the script's pair loop), and `paper_beamforming._features` collapses all method-2 inner windows into one row per outer window (W2 = window_len − 16). `data/csi_proc_python/processing.py:221,235-251` slides a fixed `window_size_2 = 10` and emits many rows per recording. Either the Intent's "verbatim" claim is stale (accept the redesign) or a verbatim oracle/aggregation is required. Human call on paper-reproduction fidelity.

**Patch**
- [ ] [Review][Patch] Cross-extractor duplicate feature-name rejection is never exercised [src/csi_bench/runner/run.py:212]

**Deferred**
- [x] [Review][Defer] `requires` metadata (`phase`) is declared but never enforced [src/csi_bench/adapters/extractors/paper_beamforming.py:101] — deferred: pre-existing pattern; `Recording` has no phase notion.
- [x] [Review][Defer] `validate_registry` dereferences `cls.min_window_len` and treats an unknown `granularity` as per-packet without validation [src/csi_bench/config/experiment.py:137] — deferred: pre-existing extension-point gap.
- [x] [Review][Defer] `train_view` leaves `meta.n_packets` describing the untrimmed recording [src/csi_bench/core/preprocess.py:17] — deferred: pre-existing (story 2.2); no step reads it yet.
- [x] [Review][Defer] Preprocessing step-name uniqueness is not validated (only extractor names are) [src/csi_bench/config/experiment.py:85] — deferred: pre-existing (story 2.2).
- [x] [Review][Defer] `compute_run_id` now depends on the CWD-relative `splits/` directory [src/csi_bench/config/experiment.py:168] — deferred: pre-existing design; the run id is no longer a pure function.
- [x] [Review][Defer] `seeds` accepts values outside numpy's 0..2**32-1 range; failure surfaces at execution, not validation [src/csi_bench/runner/run.py:88] — deferred: pre-existing (story 2.3).
- [x] [Review][Defer] No test exercises an explicit `split:` hash through `find_split`'s success branch [src/csi_bench/config/experiment.py:168] — deferred: pre-existing; test gap.
- [x] [Review][Defer] Presets `ant4`, `ant4_rescale`, `paper_norm` are unverified [src/csi_bench/config/presets.py:7] — deferred: pre-existing (story 2.2); only `ant2`/`ant2_rescale` are tested.
- [x] [Review][Defer] run-id/cache-key body derivation is duplicated and they handle `window_len`/`stride` differently [src/csi_bench/config/experiment.py:161] — deferred: maintainability; the divergence is intentional today (documented in `feature_cache_key`).
- [x] [Review][Defer] Cache has no on-disk schema/version stamp [src/csi_bench/adapters/storage/feature_cache.py:45] — deferred: pre-existing (story 2.4).

**Rejected**
- `false` — Missing `pyproject.toml`/`uv.lock` in the review diff: the storage import-linter contract and dependency bumps are present in `pyproject.toml:103-113`; they were excluded from the code-only diff by review scoping, not absent from the change.
- `low` — Cache written before the finite/duplicate/constant checks (`run.py:207`): already recorded as deferred from 2.4 in `deferred-work.md`; the run still fails on the checks, so it is not re-listed.
- `low` — `fit_scope` reference-role guard reads raw params (`experiment.py:131`): already recorded as deferred from 2.5 in `deferred-work.md`.
- `low` — `window_len = 1` accepted for a `per_window` extractor (`experiment.py:137`): `amplitude_stats` declares that combination; no harm demonstrated.
- `low` — `_weights()` cumsum vs the script's repeated addition is not bit-identical (`paper_beamforming.py:31`): only affects pathological argmax/argmin ties.
- `low` — `runtime_info()["deterministic"]` reflects torch only (`run.py:94`): the pipeline is sklearn/numpy; the key name is imprecise, not a defect.
- `low` — Resume/corrupt-artifact robustness (`_completed_seeds`, `metrics_from_dict`, per-seed merge; `run.py:103,386`, `results_schema.py:86`): artifacts are self-produced; non-object/missing-key inputs are not reachable in normal use.
- `low` — Test-hygiene duplicate/local imports (`test_results_schema.py`, `test_logreg.py`): cosmetic.

### Review Findings (second pass, 2026-10-09)

_Code review of `5269e4f..14644fe` narrowed to 2.5-related files; layers: Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor._

**Decision needed**
- [x] [Review][Decision] RESOLVED 2026-10-09 (accept redesign; Intent reworded): Oracle/aggregation deviates from the script's fixed W2=10 sliding window while the frozen Intent says "verbatim" — still open from the first pass; flagged again by Acceptance Auditor and Edge Case Hunter. Accept the redesign (and reword the Intent) or make oracle + aggregation verbatim. [tests/oracles/paper_processing.py:6-8]

**Patch**
- [x] [Review][Patch] Cross-extractor duplicate feature-name rejection has no negative test (config-level check fires first) [src/csi_bench/runner/run.py:213]
- [x] [Review][Patch] `test_add_a_seed_runs_only_the_new_seed` never asserts `mean_std` after the rerun (should equal mean/std of the four per-seed values) [tests/test_run.py]
- [x] [Review][Patch] `test_part_shorter_than_window_len_is_config_error` asserts only `"val" in err`; pin `no val windows .* window_len` [tests/test_run.py]

**Deferred**
- [x] [Review][Defer] Failed add-seed rerun overwrites `status.json` with `failed`, so `_completed_seeds` discards all finished seeds next run [src/csi_bench/runner/run.py:111,317] — deferred: pre-existing (story 2.3), not 2.5.
- [x] [Review][Defer] Resumed run overwrites provenance/duration for the incremental pass only [src/csi_bench/runner/run.py] — deferred: pre-existing (story 2.3).
- [x] [Review][Defer] Resumed seed is never compared against a fresh all-seeds run (`_seed_everything` unpinned by tests) [src/csi_bench/runner/run.py:88] — deferred: only observable with global-RNG heads (mlp); cheap test to add later.
- [x] [Review][Defer] No run-directory lock for concurrent same-`run_id` runs [src/csi_bench/runner/run.py:302] — deferred: pre-existing (story 2.3).
- [x] [Review][Defer] `_setup_runtime` sets torch determinism/default device process-globally and never restores [src/csi_bench/runner/run.py:79] — deferred: pre-existing; leaks across tests/library callers.
- [x] [Review][Defer] Feature cache key omits dataset content hash and uses unsorted `rooms` [src/csi_bench/config/experiment.py] — deferred: pre-existing (story 2.4).

**Rejected**
- `low` — cache written before finite/duplicate checks (`run.py:207`): already deferred from 2.4.
- `low` — `requires.phase` unenforced, missing `VENDOR_HASH`: already deferred / not required by spec.
- `low` — `paper_beamforming` lacks ndim/NaN guards: the runner's finite check catches NaN; no reachable ndim≠3 input.
- `low` — unbounded memory in `paper_beamforming`: already deferred; unverified.
- `low` — duplicate `rooms`, `fit_scope` raw-param check, preset missing `params`, `ids` shorthand, uniform exit code 2, silent no-op resume, `_completed_seeds` on non-dict JSON, `_seed_everything` ordering vs preprocessing, redundant slice in `_combined_amplitude`, global `os.replace` patch in a test, layering-test coverage: negligible or self-produced inputs; fixes add complexity.
