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

## Deferred from: code review of spec-2-5-paper-beamforming-feature-family (2026-10-09)

- A failed add-seed rerun writes `status.json = failed` while `metrics.json` keeps the earlier seeds; `_completed_seeds` requires `done`, so the next run recomputes every seed. Pre-existing (story 2.3).
- Resumed runs rewrite `provenance.json`, `started_at` and `duration_seconds` for the incremental pass only; reused seeds lose their commit/env record. Pre-existing (story 2.3).
- No test compares a resumed seed with the same seed from an all-seeds run, so `_seed_everything` placement is unpinned; only observable with global-RNG heads (mlp).
- No run-directory lock: two processes with the same `run_id` can race on metrics/predictions. Pre-existing (story 2.3).
- `_setup_runtime` sets `torch.use_deterministic_algorithms`/default device process-globally and never restores them.
- Feature cache key omits a dataset content hash and uses unsorted `rooms`. Pre-existing (story 2.4).
- Compare `paper_beamforming` against the real `data/csi_proc_python/processing.py` once on a sample recording (oracle is a hand transcription with documented deviations).

## Deferred from: code review of spec-2-8-comparison-report-and-notebook-results-api (2026-10-10)

- Sweep `--force`/dirty-rerun tests (`tests/test_sweep.py:87-121`) only assert the `force` kwarg; spy on `run_module._extract` to prove seeds are recomputed. Story 2.7.
- Sweep robustness (story 2.7): duplicate `run_id`s in one sweep run twice; `_sweep_action` can raise outside the per-run try; a still-dirty finished run is skipped even if the edits changed; `--force` leaves stale seed artifacts.
- Notebook `01_classical_overview.ipynb` is never executed by a test; add a test applying its pandas expressions to the `runs` fixture index.

## Deferred from: Story 2.9 measurements (2026-10-10)

- source_spec: `_bmad-output/implementation-artifacts/spec-2-9-faster-feature-extraction-and-head-training.md`
  summary: Compute `paper_beamforming` inner-window metrics once per packet and reuse them across overlapping windows (Story 2.10, via the block-aware `extract_blocks` path). Sharing across window lengths needs a runner-owned cache entry and stays deferred until extraction is still a bottleneck after 2.10.
  evidence: Multi-core extraction gave no speed-up (956 s vs 989 s); see `proposal-2-10-reuse-inner-windows.md`.

## Deferred from: code review of spec-2-9-faster-feature-extraction-and-head-training (2026-10-10)

- knn "more than one core" AC is only checked via `joblib.effective_n_jobs(None)` inside `fit` (`tests/test_run.py:931`), not by observing actual parallel work; `KNeighborsClassifier` may run its tree algorithms serially. Settle by checking the estimator's chosen algorithm/dispatch or measuring CPU time.
- No wall-clock speed-up from multi-worker extraction (spec Implementation Notes: 956 s vs 989 s); already acknowledged and tracked for the proposed Story 2.10.
- Default extraction workers to 1: 4 workers gave no speed-up (956 s vs 989 s) and cost fork overhead and ~5 min test time. Keep head jobs at `min(4, cpus)`. Decide during the Story 2.9 code review, since it changes that story's AC.
- Time each stage (load, preprocess, extract, head fit) on a few sweep cells; `svm` on 40–60k rows may dominate once extraction is fast. First AC of Story 2.10.
- The "memory-bandwidth bound" diagnosis of the extractor comes from microbenchmarks only; Story 2.10's timing step should re-check it.

## Deferred from: code review of spec-2-11-share-paper-beamforming-per-position-metrics-across-the-sweep.md (2026-10-10)

- No automated regression guard for equality against a pre-change cached matrix (AC-2's "matrix cached by the old path" clause): the bit-for-bit check against the pre-change matrix (key `e7a4d911...`, L=20 r1) is recorded in the spec's Implementation Notes but was manual; an automated guard requires committing a stored fixture of pre-change output the suite cannot regenerate.
- BLAS/OpenMP thread pools are unbounded in forked extraction workers (only torch threads are capped); consistent with the measured lack of parallel-extraction speed-up (956 s vs 989 s) — belongs to Story 2.9's performance diagnosis.
- `svm` `KERNEL_CACHE_MB = 500` per fitted instance: 4 parallel head workers could hold ~2 GB of libsvm cache; measured peak RSS (1.76 GB) is within the ~4 GB budget, so only the "kept modest" comment wording is off — belongs to Story 2.9.

## Deferred from: code review of spec-2-10-compute-paper-beamforming-inner-metrics-once-per-packet.md (2026-10-10)

- knn/random_forest "use more than one core" AC is verified only via `joblib.effective_n_jobs(None)` — the test observes the ambient joblib config, not actual parallel work dispatched by the estimator; already tracked from spec-2-9's review.

## Deferred from: correct-course for Story 2.11 (2026-10-10)

- source_spec: none
  summary: Run several sweep processes at once (claim lock per run, lock per feature-cache key, safe index updates).
  evidence: out of scope for Story 2.11; estimated 2-3x on the head stage; do after 2.11 so workers do not duplicate extraction.
- source_spec: none
  summary: Rewrite the `paper_beamforming` maths to compute each packet's amplitude and pair distances once, keeping the sum order so results stay identical.
  evidence: out of scope for Story 2.11; only if extraction is still a bottleneck after sharing.

## Deferred from: code review of spec-3-2-reproduce-the-paper-and-measure-leakage.md (2026-10-11)

- source_spec: spec-3-2-reproduce-the-paper-and-measure-leakage.md
  summary: Test order dependence through shared registry state — `test_config_validation_rejects_comparison_on_window_plan` and `test_missing_plan_passes_validation` fail with `unknown dataset 'tor_vergata'; registered: none` when other modules run first.
  evidence: pre-existing; reproduced with the 3.2 changes stashed; full suite in default order passes.
