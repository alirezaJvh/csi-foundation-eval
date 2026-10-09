---
title: 'Preprocessing chain with antenna selection and global rescale'
type: 'feature'
created: '2026-10-08'
status: 'done'
baseline_commit: '5269e4feb127cf9cc1d03ee909a69408f6eefbaa'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-2-1-run-one-classical-experiment-end-to-end.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** A run feeds raw 4-antenna CSI to the extractors; there is no way to select antennas or rescale amplitudes, and any such statistic would risk learning from val/test data.

**Approach:** Add an ordered `preprocessing:` chain of fitted `PreprocessStep` adapters (`antennas`, `global_rescale`, `recording_mean_norm`) plus named presets. The runner fits each step on the selected **train** blocks only, applies it to whole recordings, and records fitted state in provenance. Order is part of the `run_id`.

## Boundaries & Constraints

**Always:**
- Port `PreprocessStep` in `core/ports.py`: `VERSION`, `fit(recordings) -> dict` (JSON-able state) and pure `transform(recording, state) -> Recording` (returns a new `Recording`, never mutates input). Adapters live in `adapters/preprocess/`, register as stage `preprocess`, with `Params` and `VERSION`.
- Config: `preprocessing` list (default `[]`). Entry forms: shorthand `{antennas: [0, 1]}` (list value → params `{ids: [...]}`), `{global_rescale: {}}` (mapping value → params), explicit `{name, params}`, or a preset-name string. Presets expand in place at load, so the saved `config.yaml` holds explicit steps. Presets in `config/presets.py`: `ant2` (`[0, 1]`), `ant4` (`[0, 1, 2, 3]`), `ant2_rescale`, `ant4_rescale`, `paper_norm` (`recording_mean_norm`). Unknown preset/step or bad params → `ConfigError` naming `preprocessing.<i>…`, before any data loads.
- Chain runs in listed order on selected-room recordings. Step *i* is fitted on step *i−1*'s output restricted to each recording's **train** block (`core/preprocess.py`: `train_view`, `run_chain`; core imports nothing internal), then applied to the whole recording. Windower, extractor validation (`n_antennas`) and extraction use the processed recordings.
- `antennas` keeps the listed original ids in listed order and sets `Recording.antenna_ids` to them; an id not present → `ConfigError`. `global_rescale`: state `{mean_amplitude}` = mean |csi| over all selected train blocks; divides csi by it. `recording_mean_norm`: state = per recording, per antenna mean |csi| over that recording's train block; divides each antenna by its own mean. Output stays complex64; non-finite or non-positive means → `DataError`.
- `recording_mean_norm` param `fit_scope: train | whole_recording` (default `train`); `whole_recording` fits on the full recording and is accepted only with `role: reference`, else `ConfigError` naming `preprocessing.<i>.params.fit_scope`.
- `run_id` hashes the chain as an ordered list of `{name, version, affecting params}`; `provenance.json` gains `preprocessing: [{name, version, params, fitted}]`.

**Never:**
- No cache, phase steps, per-packet temporal steps, other splitters, feature selection, or changes to `data/`, `splits/`, the Windower or split logic. Steps never see val/test packets (except the explicit `whole_recording` reference scope). `core/` imports nothing internal; only `runner` imports adapters. No `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy chain | `[{antennas: [0,1]}, {global_rescale: {}}]` | `antenna_ids == (0, 1)`; mean fitted on train blocks; value in provenance | N/A |
| Reordered chain | same steps, other order | different `run_id` | N/A |
| Preset | `preprocessing: [ant2_rescale]` | same steps and `run_id` as written out | unknown name → `ConfigError` |
| Leak guard | val/test packets perturbed | fitted state unchanged | N/A |
| Reference norm | `recording_mean_norm` + `whole_recording` | allowed with `reference` | `comparison` → `ConfigError` |
| Missing antenna | `antennas: [0, 7]` | run fails | `ConfigError`, exit 2 |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/ports.py` -- add `PreprocessStep`
- `src/csi_bench/core/contracts.py` -- `Recording` (frozen; use `dataclasses.replace`)
- `src/csi_bench/core/splitplan.py` -- `PARTS`; `plan.blocks[rid]["train"]` is `(start, end)`
- `src/csi_bench/core/registry.py` -- `register`, `affecting_params`; add stage name `preprocess`
- `src/csi_bench/config/experiment.py` -- `ExperimentConfig`, `validate_registry`, `resolved_body`; add `preprocessing`
- `src/csi_bench/runner/run.py` -- `_execute` (processed recordings feed `_validate_extractors`, `build_windows`, `_extract`), `collect_provenance`
- `src/csi_bench/runner/discover.py` -- already walks adapter packages; check `preprocess` is found
- `tests/contracts/extractor.py`, `tests/test_run.py`, `tests/test_experiment_config.py` -- patterns to copy (registry-reset fixture, synthetic recordings)
- new: `core/preprocess.py`, `config/presets.py`, `adapters/preprocess/{__init__,antennas,global_rescale,recording_mean_norm}.py`, `tests/contracts/preprocess_step.py`, `tests/test_preprocess.py`

## Tasks & Acceptance

**Execution:**
- [x] `core/ports.py`, `core/preprocess.py` -- `PreprocessStep` port; `train_view`, `run_chain` returning processed recordings and per-step fitted state
- [x] `adapters/preprocess/*.py` -- the three steps
- [x] `config/presets.py`, `config/experiment.py` -- preset expansion, step shorthand, `preprocessing` field, registry and `fit_scope`/role validation, `run_id` body
- [x] `runner/run.py` -- build steps, run the chain before validation and windowing, add fitted state to provenance
- [x] `tests/contracts/preprocess_step.py` -- `transform` pure (input unchanged, repeatable); `fit` state unchanged when val/test packets are perturbed; called for every registered step
- [x] `tests/test_preprocess.py`, `test_experiment_config.py`, `test_run.py` -- I/O matrix rows, chain order, train-only mean value, end-to-end run with `[0,1]` + rescale (provenance holds the fitted mean)

**Acceptance Criteria:**
- Given the config in the I/O matrix, when I run it, then `status.json` is `done`, the extractors saw 2 antennas, and `provenance.json` lists both steps with fitted values.
- `ruff check src tests`, `ruff format --check src tests`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

### Review Findings

- [x] [Review][Patch] Migrate `describe` to `core.splitplan.find_split` and delete the duplicated `cli._find_split`, whose multi-plan hint has already diverged from the shared rule [src/csi_bench/runner/cli.py:25-39]
- [x] [Review][Patch] Drop the `fitted: Sequence[dict] = ()` default on `collect_provenance`; with `zip(..., strict=True)` it turns any config with preprocessing into a bare `ValueError` [src/csi_bench/runner/run.py:82]
- [x] [Review][Patch] Delete the dead `STAGES` constant (no reader, and it omits the new `preprocess` stage) [src/csi_bench/config/experiment.py:20]
- [x] [Review][Patch] Deep-copy preset params in `_expand_preprocessing`; the loaded `ids` list currently aliases `PREPROCESS_PRESETS`, so mutating a config corrupts the module-level preset [src/csi_bench/config/experiment.py:48]
- [x] [Review][Patch] Pin `started_at` stability in a test; the current `started_at <= finished_at` assertion passes even with the pre-fix per-transition `_now()` [tests/test_run.py:123]
- [x] [Review][Patch] Add an unknown-extractor-param case (`extractors.0.params`) to `test_invalid_fields_name_the_field`; `extra="forbid"` on extractor `Params` is untested [tests/test_experiment_config.py:118]
- [x] [Review][Patch] Add a preprocessing case (bad preset/params) to `test_invalid_config_never_loads_data` so the "before any data loads" guarantee covers the new field [tests/test_run.py:211]

#### Rejected

- `low` — `resolved_body` keys `split` on the plan file's stem rather than its content hash (blind): a runnable plan is content-verified by `load_split_plan` and `write_split_plan` names files by `split_hash`, so the id cannot diverge in normal use; a check would add filesystem reads [src/csi_bench/config/experiment.py:164]
- `low` — `resolved_body`/`compute_run_id` now do filesystem I/O, so the id is not a pure function of the config (blind): intended by spec-2.1 ("include the resolved split hash"); the resolution is consistent within a run [src/csi_bench/config/experiment.py:199]
- `low` — `save_config` writes `split: null` and never persists the resolved hash (blind): spec-2.1 defines null as "the single committed plan"; rerun semantics belong to story 2.7 [src/csi_bench/config/experiment.py:204]
- `low` — no save→load round-trip test exercises a preprocessing chain (blind): `test_saved_config_holds_explicit_steps` and `test_run_id_round_trip` cover the two halves [tests/test_experiment_config.py]
- `low` — `run_chain`/`train_view` never assert a transformed `Recording` keeps its id, label, room and packet axis (blind): no current step violates this; guards are speculative [src/csi_bench/core/preprocess.py]
- `low` — the `PreprocessStep` port does not declare the `fit_scope` attribute `run_chain` reads via `getattr` (blind): only `recording_mean_norm` uses it; declaring it adds protocol surface [src/csi_bench/core/ports.py:57]
- `low` — `find_split` has no direct unit tests for its error paths (blind): it is exercised through the run-id tests and `describe` runs the duplicate copy [src/csi_bench/core/splitplan.py:20]
- `low` — `paper_norm` expands `recording_mean_norm` with the default `fit_scope: train` though the name suggests whole-recording (blind): spec-2.2 defines `paper_norm` as plain `recording_mean_norm`; the paper's whole-recording norm belongs to epic 3 [src/csi_bench/config/presets.py:14]
- `low` — list-valued shorthand maps to `{"ids": ...}` for any step, not just `antennas` (blind): no other list-valued step exists; a guard is speculative [src/csi_bench/config/experiment.py:53]
- `low` — `tests/test_preprocess.py` lacks the `fresh_registry` fixture and `test_registered_steps_all_covered` leaves the registry and `sys.modules` mutated (blind): every other module resets its own registry, and no failure is observed [tests/test_preprocess.py:34]
- `low` — provenance stores raw config `params` while the run id stores resolved/affecting params (auditor): spec-2.2 says provenance lists `params` (fitted state is recorded alongside), so raw is spec-literal [src/csi_bench/runner/run.py:102]
- `false` — the explicit `{"name": ...}` branch appends the caller's dict by reference (blind): that dict comes from `yaml.safe_load` and is discarded; pydantic builds the stored `AdapterSpec`, so no live alias persists [src/csi_bench/config/experiment.py:49]
- `false` — the preprocess contract test is not "called for every registered step" (auditor): `test_registered_steps_all_covered` asserts `registry.names("preprocess")` equals exactly the three steps, so a new step fails CI and forces the parametrize list to grow [tests/test_preprocess.py:47]
- `false` — `tests/test_results_schema.py` adds coverage unrelated to this change (verification-gap): extra passing coverage is not a defect
