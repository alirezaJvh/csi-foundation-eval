---
title: 'Pool and score LWM in the comparison'
type: 'feature'
created: '2026-10-17'
status: 'done'
baseline_commit: '0b15d752a8ce1572174d977cfcabc29ce0e3f686'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-5-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `lwm` extractor (5.1) and its preprocessing presets (5.2) exist, but config validation still rejects `per_packet` extractors. LWM can't be windowed, scored or shown in `report`.

**Approach:** Add a top-level `pooler` config field and `mean`/`mean_std` pooler adapters. Add a runner path that gets per-packet embeddings from a per-recording packet cache (no window length in its key) and pools them per window, then feeds the shared selector/heads/metrics/intervals. LWM input stats go into provenance. Add `experiments/lwm.yaml` covering `time_block` and both `room_holdout` plans.

## Boundaries & Constraints

**Always:**
- `pooler: AdapterSpec | None = None`, also accepting a bare name (`pooler: mean`). It must be set iff every extractor is `per_packet`. A per-window extractor with a pooler, a per-packet one without, or a mix of the two → `ConfigError` naming the field. `window_len: 1` with a per-window extractor → `ConfigError` (AD-7), even if its `min_window_len` is 1. Poolers declare `min_window_len` (`mean` 1, `mean_std` 2: std over one packet is constant).
- `run_id`: `pooler` is dropped from the resolved body when null, so every existing run_id stays the same. When set it enters as `{name, version, params}`.
- Poolers are adapters in `adapters/poolers/` (registry stage `pooler`). They have no fitted state and are pure: `pool(packets: PacketFeatures, starts, window_len) -> FeatureMatrix`, with names `<packet name>.mean` / `.std`. Output equals naive per-window numpy within atol 1e-5.
- Packet cache: one entry per (recording, extractor), stored atomically by a runner-only storage adapter. The key covers dataset + version, recording_id, source_sha256, the preprocessing steps, the fitted hash, and the extractor (name, affecting params, version, vendor_hash, weights_revision). It has **no** window_len, stride, rooms, pooler, head or seeds. The entry stores values, feature_names and input_stats, so a cache hit still gives provenance stats. Only selected recordings are embedded. One extractor instance per run (the model loads once).
- Per-packet extractors skip the window-level feature cache. After pooling, the runner's existing finite/non-constant check applies unchanged.
- `provenance.json` gets `extractor_input_stats: {<extractor>: {<recording_id>: stats}}` for per-packet runs. Each packet-cache lookup goes into `feature_cache` as `{extractor, recording_id, key, hit, kind: "packets"}`.
- **Decision (Jj, 2026-10-17): `experiments/lwm.yaml` is the full epic grid (5000 runs).** `head.name` is the outermost (slowest) sweep axis, in the order `[logreg, knn, mlp, random_forest, svm]`, so the cheap heads cover every cell first. The header comment states the run counts and the estimate (about 3 days at 4 workers, about 1.5–2 days with `--workers 12`).
- `report`: the pooler is a variant axis (it goes into `variant` when set), not part of the method. Row/cell selection is unchanged (best val macro-F1, test score with interval).

**Never:** Fine-tune, or add trainable poolers or attention. Change the LWM adapter output, existing steps/presets, or any existing `run_id`/variant string. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| LWM run | `extractors: [lwm]`, `pooler: mean_std`, w=10 | features `(n_windows, 256)`, run `done`, stats in provenance | — |
| window change | same config, `window_len` 10 → 40 | packet cache all hits, `extract_packets` never called | — |
| w=1 | lwm + `mean`, `window_len: 1` | valid run | — |
| w=1 per-window | `amplitude_stats`, `window_len: 1` | — | `ConfigError` `window_len` |
| mean_std w=1 | lwm + `mean_std`, w=1 | — | `ConfigError` `pooler` |
| missing / extra pooler | lwm with no pooler; classical with `pooler: mean`; lwm + amplitude_stats | — | `ConfigError` |
| selection default | lwm config without `feature_selection` | `feature_selection` is null | — |

</frozen-after-approval>

## Code Map

- `src/csi_bench/config/experiment.py` -- `ExperimentConfig` (:96): add `pooler` with a bare-name validator. `validate_registry` (:185): replace the per_packet guard (:218) with the pooler/granularity/window rules. `resolved_body` (:249): pop `pooler` when null. Add `packet_cache_key(config, spec, fitted, recording_id, source_sha256)` modelled on `context_cache_key` (:316).
- `src/csi_bench/core/ports.py` -- add the `Pooler` and `PacketCache` protocols, and update the per_packet comment on `FeatureExtractor`.
- `src/csi_bench/adapters/poolers/{__init__,mean,mean_std}.py` (new) -- copy the shape of `adapters/preprocess/global_rescale.py`. Use float64 cumsum over the recording's rows for window means; std = sqrt(max(E[x²]−E[x]², 0)). Cast the output to float32.
- `src/csi_bench/adapters/storage/packet_cache.py` (new) -- `NpzPacketCache` modelled on `feature_cache.py`, with input_stats as a JSON string array. A corrupt entry is a miss.
- `src/csi_bench/runner/run.py` -- in `_extract` (:382), branch on `granularity == "per_packet"`: per recording, get the packet features from the cache or call `extract_packets(rec.csi)` and put them, then pool that recording's windows (grouped by recording_id, rows in window order). Collect stats. `_execute` (:581) builds the pooler and passes stats to `collect_provenance` (:330, new optional arg). The heads/selector/intervals code is unchanged.
- `src/csi_bench/config/builder.py` -- add a `pooler(name, **params)` method.
- `src/csi_bench/runner/report.py` `_variant` (:33) -- add `pooler` only when set.
- `experiments/lwm.yaml` (new) -- one sweep mapping (so head-major order covers both protocols). Base `preprocessing: [ant2, lwm_rescale]`. Axes in order: `head.name`, then `preprocessing.0`/`preprocessing.1` (list-index paths already work in `_set_path`), `extractors.0.params.embedding`, `pooler.name`, `window_len`, and a paired axis `@protocol` (the `@` form in `sweep.py`) with 5 entries `{split, selection.rooms}`: `e31e1b7dd9b4` × [r1] / [r2] / [r1, r2] (`time_block`), and `1e8688472a1f` / `c4faaf799ad5` × [r1, r2] (`room_holdout`). The header comment gives the run counts.
- Tests: `tests/test_run.py` fixtures (`_make_project`, 4×800×3 CSI, `_spy_extract` :517) as the pattern. Register a fake per-packet extractor in the test (fast, call-counting) for the runner tests, plus one end-to-end run with the real `lwm` on the tiny fixture. `tests/test_lwm.py:294` asserts the old guard; replace it. Add a pooler contract check to `tests/contracts/`.

## Tasks & Acceptance

**Execution:**
- [x] `core/ports.py`, `config/experiment.py`, `config/builder.py` -- pooler field, validation, run_id, packet key -- the config surface
- [x] `adapters/poolers/*`, `adapters/storage/packet_cache.py` -- poolers and the cache -- pure pooling, reusable embeddings
- [x] `runner/run.py`, `runner/report.py` -- per-packet path, provenance stats, variant -- LWM rows in the table
- [x] `experiments/lwm.yaml` -- the full-grid sweep -- the epic's success criterion
- [x] `tests/test_poolers.py`, `tests/test_packet_cache.py`, `tests/test_run.py`, `tests/test_experiment_config.py`, `tests/test_lwm.py`, `tests/test_sweep.py`, `tests/test_report.py` -- the matrix rows, the pooler contract, cache atomicity/corruption, existing run_ids unchanged, `lwm.yaml` expands and validates (the run count matches its comment), report variant shows the pooler -- coverage

**Acceptance Criteria:**
- Given a finished LWM run, when only `window_len`, `pooler` or `head` changes, then no packet is re-embedded and the metrics of an unchanged config are identical on a cold vs warm cache.
- Given `lwm.yaml` runs (or a subset of them), when `csi-bench report` runs, then LWM rows appear in the `time_block` section and in the `room_holdout` section with intervals.
- Given the whole suite, when `pytest`, `ruff`, `ruff format --check` and `lint-imports` run, then they pass, and `ty` reports no new errors in the touched files.

## Implementation Notes

- 2026-10-17: implemented directly in the planning session (not by a subagent), because the context was already loaded.
- **Run count is 4500, not 5000.** The frozen decision says "full epic grid (5000 runs)". But the same spec's matrix makes `mean_std` × `window_len: 1` a `ConfigError`, so that pair cannot exist. `lwm.yaml` therefore uses a paired `@pooling` axis with 9 valid (pooler, window) pairs: 5 × 2 × 5 × 2 × 9 × 5 = 4500 runs (2700 `time_block`, 1800 `room_holdout`). The estimate drops to about 2.5–3 days at 4 workers.
- One sweep mapping with a paired `@protocol` axis (instead of two list items) keeps `head.name` outermost across both protocols. The first 900 runs are logreg, the last 900 svm (tested).
- `mean_std` centres each column on its recording mean before the prefix sums, so E[x²]−E[x]² doesn't cancel on long recordings with large offsets (tested with offset 50, 6000 packets).
- The packet cache lives under `<cache_root>/packets/`. A cached entry whose row count differs from the recording's packet count raises `ExtractorError`, on a hit as well as a miss.
- Per-window configs with `window_len: 1` are now rejected (AD-7), even for `amplitude_stats` (`min_window_len` 1). No committed experiment used w1.
- `report` `_variant` shows only the pooler name. Configs without a pooler keep their exact variant string.
- Benchmarks (synthetic features of the real shape, 4 joblib workers): logreg 3 s, knn 1–10 s, mlp 6 s, random_forest 131–291 s, svm 41–203 s per fit, at 19k×128 to 38k×256 train windows.
- `pytest` 506 passed; `ruff`, `ruff format --check` and `lint-imports` are clean. `ty` reports no errors in the touched files. The remaining errors were there before (`runner/intervals.py`, `tests/test_run.py:1055`, `tests/test_sweep.py:329`).
- **Real-data slice** (10 runs: logreg, `[ant2, lwm_rescale]`, cls, mean w10 / mean_std w40 × 5 protocols): all `done`. The w10 runs embedded every recording, taking 209–548 s of extraction per protocol. Every w40 run had all packet-cache hits (19/19, 11/11, 8/8) with 0.1–0.4 s of extraction, so LWM is never re-run when the window changes. Peak RSS was 2.6 GB. Post-rescale median |CSI| was 0.71–1.15, with no amplitude warning. `report` shows LWM rows in both sections with intervals: `time_block` F1 97.7 (r1) / 88.5 (r2) / 91.8 (mixed); `room_holdout` 26.5 (r1→r2) / 19.7 (r2→r1).
- Embedding under the runner is about 2.6× slower than the 5.1 timing (532 s vs 203 s for 63k packets), because the runner pins `torch.set_num_threads(1)`. The full sweep needs about 68 embedding passes, so about 8 h of embedding, not the planned 2–3 h. The `lwm.yaml` comment is updated (about 3 days at 4 workers).
- logreg emits `lbfgs failed to converge` warnings on the LWM features (max_iter 1000). The runs still complete.
- Code review skipped at the user's request; nothing staged or committed.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run lint-imports` -- expected: clean
- `uv run ty check` -- expected: no new errors in touched files
- `uv run csi-bench sweep experiments/lwm.yaml` on a small slice (e.g. one preset, w 10/40, logreg) then `csi-bench report` -- expected: LWM rows in both sections
