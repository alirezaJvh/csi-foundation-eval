## Deferred from: code review of spec-1-2-load-the-own-recordings-as-labelled-recordings (2026-10-07)

- `TorVergata.Params.root` default (`data/csi_proc_python/saved_csi`) is relative to CWD; running outside the repo root fails with "dataset root not found". Spec mandates the default; revisit path resolution with the CLI/config stories.

## Deferred from: code review of spec-1-4-inspect-the-dataset-with-csi-bench-describe (2026-10-08)

- Repo-wide `ruff check` / `ruff format --check` fail (181 errors, all under untracked `.agents/skills/**`), so story 1.4's AC3 verification command does not pass in this workspace. Environmental/pre-existing, not caused by the change — fix by excluding the BMad skills tree from the ruff config.

## Deferred from: code review of spec-2-1-run-one-classical-experiment-end-to-end (2026-10-08)

- Rerun semantics: rerunning an existing `run_id` overwrites a `done` run's status/metrics, leaves stale `seeds/<n>/`, has no lock, and `KeyboardInterrupt` leaves `status=running`. Belongs to AD-14 (story 2.7, skip-done / `--force`).
- `_validate_extractors` (`min_window_len`, `n_antennas`) branches have no test; unreachable until a second extractor is added.

## Deferred from: code review of spec-2-3-full-classifier-family-with-seeds (2026-10-08)

- Resume after a mid-run failure redoes every seed: `_completed_seeds` requires `status == "done"` and `metrics.json` is written only after the whole seed loop, so predictions already written for finished seeds are discarded. AD-14 resumability/`--force` belongs to story 2.7.
- `started_at`/`duration_seconds` on an add-a-seed resume reflect only the resume invocation (`started_at` re-stamped, duration measures the missing-seed loop). Run-timing semantics on re-execution belong to story 2.7.
- Reducing `config.seeds` leaves stale seed metrics: `run_id` excludes seeds, so `[0, 1, 2] → [0]` finds no missing seed and returns without pruning `metrics.json`/`seeds/`. Rerun/prune semantics belong to story 2.7.

## Deferred from: code review of spec-2-4-runner-owned-feature-cache (2026-10-08)

- Dataset params are unreachable and absent from run id / cache key: `ExperimentConfig.dataset` is a bare `str` with no params field, so `TorVergata.Params(root=...)` cannot be set from config, and `resolved_body`/`feature_cache_key` record only the dataset name + `VERSION` (`config/experiment.py:69`). Pre-existing design from story 2.1; not caused by the cache change.

- source_spec: `_bmad-output/implementation-artifacts/spec-2-5-paper-beamforming-feature-family.md`
  summary: `_extract` writes a matrix to the feature cache before the finite/duplicate-name checks run.
  evidence: a non-finite matrix is cached and served on later runs (the checks re-run on a hit, so the run still fails, but the bad entry stays); pre-existing from story 2.4.
- source_spec: `_bmad-output/implementation-artifacts/spec-2-5-paper-beamforming-feature-family.md`
  summary: `validate_registry` tests `fit_scope` on raw params of any step instead of validated `Params`.
  evidence: a step whose default `fit_scope` is `whole_recording` would bypass the reference-role rule; pre-existing, no such step exists yet.
- source_spec: `_bmad-output/implementation-artifacts/spec-2-5-paper-beamforming-feature-family.md`
  summary: `paper_beamforming` builds a `(windows, 40, size, subcarriers)` complex128 array per method with no chunking.
  evidence: fine at window_len 56 (tens of MB); would grow with much longer windows or many subcarriers (unverified).

## Deferred from: code review of spec-2-5-paper-beamforming-feature-family (2026-10-08)

- `requires` metadata (`phase`) is declared by extractors but never enforced: `Recording` has no phase notion and `_validate_extractors` reads only `n_antennas`. Pre-existing pattern; enforce or drop the key.
- `validate_registry` (`config/experiment.py:137`) dereferences `cls.min_window_len` and `feature_cache_key` treats an unknown `granularity` as per-packet, so a future extractor that omits or mistypes the attribute fails with `AttributeError` / silently changes cache-key behavior. Pre-existing extension-point gap.
- `train_view` (`core/preprocess.py:17`) rewrites `csi` but leaves `meta.n_packets` describing the untrimmed recording, so a step that consults `meta` sees a length contradicting `csi.shape[1]`. Pre-existing (story 2.2); no step reads it yet.
- Preprocessing step-name uniqueness is not validated, though extractor-name uniqueness is (`config/experiment.py:85`); duplicate steps are accepted silently and change the run id. Pre-existing (story 2.2).
- `compute_run_id` now resolves the committed split via `find_split` (`config/experiment.py:168`), so it depends on the CWD-relative `splits/` directory and is no longer a pure function of the config. Pre-existing design.
- `seeds` accepts values outside numpy's `0..2**32-1` range; the failure surfaces in `_seed_everything` at execution (run marked failed) instead of at config validation. Pre-existing (story 2.3).
- No test runs an explicit `split:` hash through `find_split`'s success branch. Pre-existing test gap.
- Presets `ant4`, `ant4_rescale` and `paper_norm` (`config/presets.py:7`) are never exercised; a wrong ids/steps value would ship with no CI signal. Pre-existing (story 2.2).
- The run-id body (`resolved_body`) and cache-key body (`feature_cache_key`) are hand-built twice and treat `window_len`/`stride` differently; the divergence is intentional today but the duplication can drift. Maintainability.
- `NpzFeatureCache` has no on-disk schema/version stamp, so a future layout change would serve entries written by an old implementation rather than treating them as a miss. Pre-existing (story 2.4).
