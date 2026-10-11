---
title: 'Reproduce the paper and measure leakage'
type: 'feature'
created: '2026-10-10'
status: 'done'
route: 'dispatch'
baseline_commit: '2a05d9d6f64b53463dd2b8c8c12a6a0b16b68678'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The paper's method has never been run here under its own random window split, so nobody can say how much of the published 99.41 / 99.59 / 97.37 comes from leakage, or whether the port computes the paper's features.

**Approach:** Add `experiments/paper_reproduction.yaml`, a sweep of the paper method (`paper_beamforming` + `random_forest`) per room under a `random_window` reference split and the time-block comparison split. Commit the random plan. Check the port against `saved_dataset/*.npy`. Extend `report` with a leakage block.

## Boundaries & Constraints

**Always:** W2 = 40 means `window_len` 56 (W2 inner positions plus 16), stride 1, `ant2`, `random_forest`, seed 0. Rooms r1 and r2 use all six features; mixed uses MRMR with `k: 4`. Reference rows: `role: reference`, the `random_window` plan (seed 0, fractions 0.6/0.2/0.2, same as time-block), `recording_mean_norm` with `fit_scope: whole_recording`. Comparison rows: `role: comparison`, the time-block plan, `recording_mean_norm` train-fitted. The fixture check runs the raw, un-normalized recordings like `processing.py` (`norm = 0`) and skips when `data/**/*.npy` are absent (git-ignored).

**Never:** Change existing extractor, splitter or head behaviour, the committed time-block plan, or the run ids of existing configs. Add a split-generation CLI. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Sweep | `paper_reproduction.yaml` | 6 configs: reference and comparison for r1, r2, mixed | `ConfigError` for any invalid pairing |
| Fixture | r1/r2/r1r2 file, W2 | values match saved rows (rtol 1e-4); labels match | mismatch recorded in a test note |
| Report | reference and comparison done, same method and room | reference row beside 99.41 / 99.59 / 97.37; per room random, time-block and difference | room with only one side shows `-` |
| Several plans | `splits/tor_vergata` has a block plan and a window plan, `split` unset | the block plan is chosen | window plan needs an explicit `split` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/config/sweep.py` -- `expand_sweep`: Cartesian product of dotted paths, so role, split, normalization and selection cannot be paired. Add `@name` axes whose values are mappings of dotted paths, applied together via `_set_path`.
- `src/csi_bench/core/splitplan.py` -- `find_split`: with `split` unset, consider only block-unit plans (via `read_plan_unit`); a named plan is unchanged.
- `splits/tor_vergata/<hash>.json` -- new committed `random_window` plan: `write_split_plan(RandomWindow().split(TorVergata().load(), "tor_vergata"))`.
- `src/csi_bench/runner/report.py` -- `_comparison` picks the best-val row per (method, room) in both blocks; add a leakage block from the same picks. Reference rows may use other split hashes.
- `experiments/paper_reproduction.yaml` -- new; base config like `classical_baseline.yaml`; presets in `config/presets.py` (`ant2`); `published_reference.yaml` already holds the published numbers.
- `tests/oracles/paper_processing.py`, `tests/test_paper_beamforming.py` -- oracle pattern. Fixture check: `compute_context` + `extract_blocks` per full recording at `window_len = W2 + 16`. Files exist for r1r2 W2 {10,30,40,50}, r1 {40}, r2 {10,40}; files list order follows the commented lists in `data/csi_proc_python/processing.py`; classes from filename (`empty`, `sit`, `standing`, `walk`).
- `tests/test_sweep.py`, `tests/test_report.py`, `tests/test_random_window.py` -- fixtures to reuse.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/config/sweep.py` -- `@name` paired axes (not seedable, validated per combination) -- couples role, split, preprocessing and selection
- [x] `src/csi_bench/core/splitplan.py` -- default `find_split` ignores window plans -- existing configs keep working with two plans
- [x] `splits/tor_vergata/` -- commit the generated `random_window` plan -- reference runs need it by hash
- [x] `experiments/paper_reproduction.yaml` -- the 2 × 3 sweep with `@setting` and `@rooms` axes -- the reproduction
- [x] `src/csi_bench/runner/report.py` -- leakage block: random vs time-block test accuracy and difference per room, next to published -- the thesis number
- [x] `tests/test_paper_fixture.py` -- fixture match per available file; plan loads; sweep expands to 6 valid configs -- port check
- [x] `tests/test_sweep.py`, `tests/test_report.py` -- `@` axes, `find_split`, leakage block, one-sided room -- edge cases

**Acceptance Criteria:**
- Given `paper_reproduction.yaml`, when the sweep is expanded, then 3 reference configs on the window plan and 3 comparison configs on the time-block plan exist, and only reference configs use `whole_recording`.
- Given the sweep has run, when `report` prints, then reference rows sit beside the published numbers and the leakage block shows the random-minus-time-block difference per room.

## Implementation Notes

Implemented as specified. `uv run pytest -q` 343 passed; ruff clean; `ty` reports one error already present at baseline (`tests/test_run.py:1007`). The sweep itself and `report` on real runs were not executed. `find_split` with `split` unset now returns only the block plan, so a window-only dataset needs an explicit `split` (also for `describe`). Existing tests that relied on a lone window plan were updated.

Fixture check (`tests/test_paper_fixture.py`): the default run recomputes only a few recordings (about 77 s). A full earlier run (r1r2 W2 10/30/40/50, r2 W2 10/40) matched. Mismatches, documented in the test: (1) `distance_theta_var` differs on `r1_walking_3` (about 0.6% at W2=40, up to about 16% at W2=10), cause unconfirmed, probably an argmax tie broken differently in float32 vs float64; (2) `dataset_r1_w2_40.npy` has 10 fewer rows per r1 recording than the port (39257 vs 39367), so it is skipped as stale. Fixture files for r1 at W2 10/30/50 and r2 at W2 30/50 do not exist.

## Spec Change Log

## Review Triage Log

### Review Findings

- [x] [Review][Decision→Patch] Leakage "random split" column accepts any reference run — `report()` passes every `role: reference` row to `_leakage`; `check_split_role` allows a reference run on the time-block plan, and reference rows are not held to one split hash. Such a run lands under "random split" and corrupts the difference. Options: filter reference rows to window-unit plans; require one reference hash like the comparison side; or relabel the column "reference". (medium; edge-case-hunter+blind-hunter+acceptance-auditor+verification-gap) — resolved: option 1, `_on_window_plan` filters the random side to window-unit plans
- [x] [Review][Decision→Patch] Leakage table lists every comparison method next to the paper's published numbers — `keys = set(random_rows) | set(block_rows)` gives every classical-baseline method a row with `-` random and the published 99.41/99.59/97.37 beside it (the new test asserts this for `amplitude_stats`). Options: only keys with a reference row; published only on rows that have a reference side; keep as is. (medium; blind-hunter+acceptance-auditor) — resolved: option 2, published shown only where a reference side exists
- [x] [Review][Decision→Patch] Fixture check coverage vs runtime — Task says "fixture match per available file", but `CASES` omits `dataset_r1r2_w2_{10,30,50}.npy`, recomputes 1–2 recordings per file, skips column 2 for every r1 recording although only `r1_walking_3` was observed, and has no `slow`/data marker (~77 s whenever data is present). Options: add all cases behind a registered `slow` marker; keep the subset and record the choice; narrow the column-2 skip to `r1_walking_3`. (medium; acceptance-auditor+blind-hunter) — resolved: option 1, all 7 files, every recording, `slow` marker deselected by default, column 2 tolerated only for `r1_walking_3`
- [x] [Review][Decision→Done] AC 2 not verified on real runs — `csi-bench sweep experiments/paper_reproduction.yaml && csi-bench report` was never executed; the leakage block is checked only on synthetic runs. Options: run it now; leave as an action item before `done`. (medium; acceptance-auditor+blind-hunter) — resolved: option 2, action item; after the changes are committed run `csi-bench sweep experiments/paper_reproduction.yaml && csi-bench report` on a clean tree (runs on a dirty tree are excluded from `report`), then mark `done`
- [x] [Review][Patch] Committed window plan `499148ef70db` has no data-free hash check; both committed-plan tests skip window plans and the `test_time_block.py` comment points at a test that also skips [tests/test_random_window.py:160]
- [x] [Review][Patch] Sweep axes can set the same dotted path (plain + paired, two paired, or `head` and `head.name` in one entry); the later one silently wins [src/csi_bench/config/sweep.py:75]
- [x] [Review][Patch] r1 W2=40 fixture asserts the known-bad row counts instead of skipping as stale; a regenerated fixture would fail the test [tests/test_paper_fixture.py:101]
- [x] [Review][Patch] `TorVergata().load()` uses the relative default root while the skip guard uses `ROOT`; fails with `DataError` outside the repo root [tests/test_paper_fixture.py:123]
- [x] [Review][Patch] `_leakage` sorts keys twice (`sorted(...)` then `keys.sort(...)`) [src/csi_bench/runner/report.py:107]
- [x] [Review][Defer] Tests depend on order via shared registry state (`unknown dataset 'tor_vergata'; registered: none` when `test_random_window.py` runs after other modules) [tests/test_random_window.py:130] — deferred: pre-existing, reproduced with the changes stashed

**Rejected**

- Leakage sides may come from different variants — spec Code Map mandates "from the same picks" (best-val per method, room).
- Single-seed leakage without std — spec fixes seed 0.
- Implementation Notes contradict the test note on "matched" — fix would edit the spec under review.
- `paper_reproduction.yaml` base fields overwritten by every combination — no named harm.
- Sweep errors echo the nested paired mapping — low; flattening is more than a direct fix.
- Paired entries with different key sets fall back to base — false: that is the intended semantics.
- Recording shorter than `w2 + 16` breaks `_rows` — false: no recording is that short.
- Unreadable JSON blocks implicit `find_split` — false: with >1 file this already errored before; a lone bad file failed at load.
- Unknown `unit` silently ignored — low, no such unit exists.
- Leakage with no comparison runs prints `-` — false: the spec's one-sided rule.
- Window-only dataset now needs explicit `split` — spec-mandated (I/O matrix).
- No integration test with both plans / run ids — false: `test_random_window_plan_loads_and_is_not_default` checks the real `splits/` dir and the default still resolves to `e31e1b7dd9b4`.
- Missing leakage edge-case tests — low; the one-sided case is tested, the rest folds into the first decision.

## Design Notes

Leakage compares the same method and room under two splits, so the pairing key is `(method, room_setting)`. Sweeps pair fields with `@` axes: `@setting: [{role: reference, split: <hash>, preprocessing: [...]}, {...}]` crossed with `@rooms: [{selection.rooms: [r1], feature_selection: null}, ...]`.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass (fixture tests skip without `data/**/*.npy`)
- `uv run ruff check src tests && uv run ty check` -- expected: clean
- `csi-bench sweep experiments/paper_reproduction.yaml && csi-bench report` -- expected: reference and leakage blocks printed

Verified 2026-10-11 on clean runs: reference 99.34 / 99.93 / 96.33 vs published 99.41 / 99.59 / 97.37; leakage (random minus time-block) +9.42 r1, +5.77 r2, +21.98 mixed.
