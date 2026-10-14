---
title: 'Cross-room classical baseline'
type: 'feature'
created: '2026-10-12'
status: 'done'
baseline_commit: '7b9f128a2f1b53a8a5b1086681e07bb2b5827137'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-4-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every score in the store is within-room (98.5–99.9%), so methods cannot be told apart. There are no out-of-distribution numbers for LWM (Epic 5) to be compared against.

**Approach:** Add `experiments/classical_cross_room.yaml`. It runs the classical grid of `classical_baseline.yaml` and the paper method under both committed `room_holdout` plans (r1→r2 `1e8688472a1f`, r2→r1 `c4faaf799ad5`). One sweep file can't express "grid ∪ one extra point" today, so `sweep:` also accepts a list of sweep mappings, expanded as a union. Then run `sweep` and `report` on the real store.

## Boundaries & Constraints

**Always:** Every config has `selection.rooms: [r1, r2]`, `role: comparison` and `seeds: [0]`, and its `split` is one of the two holdout hashes. Grid = the four `classical_baseline` axes unchanged (window 20/30/40/50 × 3 extractor sets × {null, mrmr k [2,3]} × svm/knn/random_forest, `ant2`) × 2 directions = 144 runs. Paper method = `paper_beamforming` + `random_forest`, `window_len: 56`, `stride: 1`, all six features (`feature_selection: null`, as in the paper's single-room setting, since training is single-room), `preprocessing: [ant2]` only (decided 2026-10-13: `recording_mean_norm` cannot be fitted on test-room recordings under strict OOD, so no normalisation; no new preprocessing code) × 2 directions = 2 runs. Total 146 distinct `run_id`s. List-form `sweep:`: each item is an ordinary sweep mapping (plain and `@` paired axes) expanded against the same base. Items are concatenated in file order and deduplicated by `run_id` across items. Validation errors name the item index (`sweep.<i>.<path>`). An empty list or a non-mapping item → `ConfigError`. A mapping-form `sweep:` behaves exactly as today. Strict OOD: nothing is fitted on, or selected with, test-room packets.

**Never:** Change `classical_baseline.yaml`, `paper_reproduction.yaml`, the committed plans, or any existing `run_id`. Delete or rewrite existing runs. Add an oracle-selection row or tune on the test room. Edit `report`/`results_api` unless the cross-room section fails to render. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Expand experiment | `classical_cross_room.yaml` | 146 valid configs: 73 per split hash, all `rooms [r1, r2]` | — |
| List sweep | two items | union of both products, item 0 first | — |
| Overlapping items | the same config produced by two items | kept once | — |
| Bad item | `sweep: [{...}, 5]` | rejected, naming `sweep.1` | `ConfigError` |
| Empty list | `sweep: []` | rejected | `ConfigError` |
| Mapping sweep | existing experiment files | same configs and order as before | — |
| Sweep + report | real store, 222 runs | 368 runs; `tor_vergata / room_holdout` section with `r1→r2` and `r2→r1` columns, intervals in every cell, `empty` flagged under r2→r1 | — |

</frozen-after-approval>

## Code Map

- `experiments/classical_baseline.yaml` -- copy its grid axes; base needs `split` and `selection.rooms: [r1, r2]`. `paper_reproduction.yaml` -- paper method params (w=56, RF, paper_beamforming).
- `src/csi_bench/config/sweep.py:64` `expand_sweep` -- today `sweep` must be a dict and the body is one product. Move the per-mapping logic (axis validation + product + config build) into a helper; with a list, call it per item with source suffix `sweep.<i>`, and share one `seen` set so dedup spans items. Keep error text for the mapping form unchanged (tests match on it): thread a `prefix` (`"sweep"` or `"sweep.<i>"`) into the helper and `_tokens` instead of hard-coding `sweep.`. Top-level check: dict → one item with prefix `sweep`; non-empty list of dicts → items; anything else → today's `ConfigError`. `tests/test_sweep.py:60` already expects `sweep: [1, 2]` to fail — it must still fail (now as a non-mapping item).
- `src/csi_bench/runner/run.py:808` `run_sweep` -- only calls `expand_sweep`; needs no change. `config/builder.py` doesn't use sweeps.
- `src/csi_bench/core/splitplan.py` `check_split_rooms` / `config/experiment.py` `validate_registry` -- already enforce `rooms == train ∪ test` for holdout plans; the yaml must satisfy it.
- `src/csi_bench/runner/report.py` -- 4.1–4.3 already render `room_holdout` sections with direction columns, intervals and footnotes. Expected: no change.
- `tests/test_sweep.py:146` -- pattern for an experiment-file expansion test (`monkeypatch.chdir(ROOT); discover()`). `tests/test_paper_fixture.py:135` -- pattern for asserting per-config fields.
- `tests/test_paper_fixture.py:138` and `runner/run.py:834` are the only other `expand_sweep` callers; signature and return type stay the same.
- Store facts (checked 2026-10-13): `runs/index.parquet` has 222 rows, all `done`, all with `test_*_lo/hi`. Plans: `1e8688472a1f` = train r1 / test r2, `c4faaf799ad5` = train r2 / test r1, both `fractions [0.75, 0.25]`, `gap_packets 64`. Runs take about 1 s each with the feature cache. `runs/` is gitignored.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/config/sweep.py` -- accept a list of sweep mappings (union, cross-item dedup, indexed errors) -- one file can hold the grid plus the paper method
- [x] `experiments/classical_cross_room.yaml` (new) -- item 0: grid × `split`; item 1: paper method × `split` -- the story's experiment
- [x] `tests/test_sweep.py` -- list-form union/order/dedup/bad item/empty list; `classical_cross_room.yaml` expands to 146 configs with the field checks in the Always rules -- coverage
- [x] run `uv run csi-bench sweep experiments/classical_cross_room.yaml` then `report` on the real store -- produce the baseline and confirm the AC

**Acceptance Criteria:**
- Given the real store, when `sweep experiments/classical_cross_room.yaml` and then `report --runs runs` run, then all 146 runs are `done`, the report shows a `tor_vergata / room_holdout` section next to `time_block` with `r1→r2` and `r2→r1` columns, and every cell has a 95% interval.
- Given the sweep finished, when it is run again, then no run is re-executed (all skipped as done).
- Given the 222 pre-existing runs, when the sweep finishes, then their `run_id`s and metrics are unchanged.

## Implementation Notes

- 2026-10-14: sweep ran on an uncommitted tree, so all 146 runs carry `dirty: true` and `report --runs runs` hides them; the `room_holdout` section appears with `--include-dirty`. After the code is committed, re-running the sweep re-executes the dirty runs once (dirty-rerun rule) and they then show without the flag.
- The paper-method runs (w56) share the method key `paper_beamforming | none | random_forest` with the grid, so the report shows them as variants of that row (best-validation wins; w20 was picked in both directions). Paper-method test macro-F1: r1→r2 37.2 [36.1, 38.3], r2→r1 38.8 [23.0, 52.0].
- Cold `paper_beamforming` block-context cache cost about 11 min per direction; the full sweep took about 50 min.

- Code review skipped at the user's request (2026-10-14); nothing staged or committed.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests` -- expected: clean
- `uv run csi-bench sweep experiments/classical_cross_room.yaml` -- expected: exit 0, 146 runs done
- `uv run csi-bench report --runs runs` -- expected: room_holdout section with both directions, intervals in every cell
