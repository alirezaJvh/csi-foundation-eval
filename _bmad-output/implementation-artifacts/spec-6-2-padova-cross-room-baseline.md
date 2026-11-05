---
title: 'Padova cross-room baseline'
type: 'feature'
created: '2026-11-05'
status: 'in-progress'
baseline_commit: 'f362cd6680c1d1f1e1b75a459cc54790138e6b8e'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every Padova score so far is same-room, same-recording, later in time, so the supervisor's same-room question has no measured answer and nothing can be set beside SHARP's S6 and S7 tests.

**Approach:** Commit two `room_holdout` plans (`bedroom→living_room`, `bedroom→laboratory`), add `experiments/padova_cross_room.yaml` with the Padova baseline grid under both, run the sweep on the real store, and read the result from `report`.

## Boundaries & Constraints

**Always:**
- Plan params match the `tor_vergata` holdout plans: `fractions [0.75, 0.25]`, `gap_packets 64`. Validation comes from the train room only.
- Same grid as `padova_classical_baseline.yaml` without its room axis: `stride: 9`, `ant2`, `role: comparison`, `seeds: [0]`. That is 72 runs per direction, 144 in total.
- `split` and `selection.rooms` change together, because `selection.rooms` must equal train ∪ test for each plan.
- Sweep runs are made on a clean, committed tree, and the tree stays clean until the sweep ends. `report` hides dirty runs.

- **Decision (2026-11-05, human):** the human runs the sweep and the report. This story's implementation stops before them, with no code review, `git add` or commit.
- **Default taken (2026-11-05, the human did not choose):** the "whole bedroom vs SHARP's S1" statement goes in the `experiments/padova_cross_room.yaml` header, and `report.py` stays unchanged. A results write-up can only follow the sweep.

**Never:**
- No change to `src/`, the committed `b6010d87ce4d` plan, `padova_classical_baseline.yaml` or existing runs.
- No fitting or selecting on the test room, and no oracle-selection row.
- No S1-only training, 5-class subset or longer windows. Those belong to Story 6.3.
- No LWM sweep and no label mapping.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Direction column | a run under either plan | column `bedroom→living_room` or `bedroom→laboratory`, from the plan's params | N/A |
| Single-recording class | `arm_gym`, `sit_stand`, `standing` in `living_room`; six classes in `laboratory` | cell marked `*`, with a footnote naming the classes | N/A |
| Mismatched rooms | `selection.rooms` differs from the plan | — | `ConfigError` from `check_split_rooms` |
| Implicit split | `describe padova` with no `--split` | still picks `b6010d87ce4d` | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/splitters/room_holdout.py` -- `RoomHoldout(Params(train_rooms, test_rooms)).split(recordings, "padova")`. Reuse as is.
- `src/csi_bench/core/splitplan.py:112` -- `write_split_plan(plan, root)`. There is no CLI for plans, so generate them with a short one-off script. Dry run on the real data gives `725a12be6b83` (living_room) and `99d235eb2f0e` (laboratory).
- `src/csi_bench/core/splitplan.py:23` -- `find_split` never picks a holdout plan implicitly, so `describe padova` keeps working.
- `experiments/padova_classical_baseline.yaml` -- the grid to copy.
- `experiments/classical_cross_room.yaml` -- header style for a cross-room sweep. Its `split:` axis is not enough here, because rooms differ per plan.
- `src/csi_bench/config/sweep.py:68` -- paired axis: `@direction: [{split: ..., selection.rooms: [...]}, ...]`. Dry run expands to 144 unique run ids.
- `src/csi_bench/runner/report.py:420` -- `room_holdout` columns come from the rows, so the section renders without code changes. Padova's `time_block` cells still show `-`: a known deferred bug, not fixed here.
- `tests/test_sweep.py:386,409` -- expansion test pattern. `tests/test_padova.py` -- committed-plan pin pattern.

## Tasks & Acceptance

**Execution:**
- [x] `splits/padova/725a12be6b83.json`, `splits/padova/99d235eb2f0e.json` -- generate and commit -- one plan per direction. Stop and report if the hashes differ from the dry run.
- [x] `experiments/padova_cross_room.yaml` -- new sweep with the paired `@direction` axis; the header gives the run count, the plan hashes and the stride rule.
- [x] `tests/test_sweep.py` -- add a Padova cross-room expansion test: 144 unique ids, 72 per plan, grid equal to the Padova baseline apart from rooms and split.
- [x] `tests/test_padova.py` -- add a skip-if-absent pin of both plans to the loader output, with no test-room packets in train or val.
- [ ] Real store (human, after committing) -- run `sweep`, then `report`; record the table, per-class caveats and run time in Implementation Notes.
- [ ] Results write-up (after the sweep) -- state that this protocol trains on the whole bedroom (S1-S5), whereas SHARP trains on S1 only. Decided by the human in code review, 2026-11-05: this meets the epic's "the report states" criterion, and `report.py` stays unchanged.

**Acceptance Criteria:**
- Given the `padova` adapter, when the plans are generated, then both directions are written under `splits/padova/`, ready to commit, with validation from the train room only.
- Given the finished sweep on a clean tree, when `report` runs, then a `padova / room_holdout` section shows one column per direction with confidence intervals, from 144 `done`, non-dirty runs.
- Given the finished story, when `git diff --stat f362cd6` runs, then nothing under `src/` has changed.

### Review Findings

Code review 2026-11-05: 1 decision, 4 patch, 2 defer, 6 rejected.

- [x] [Review][Decision] Epic AC wants "the report" to state the S1–S5 vs SHARP S1 difference, but only a YAML comment does — `epics.md` Story 6.2 says the report states it. This spec took the header as a default "the human did not choose" and forbids `src/` changes, so `csi-bench report` readers never see the statement. Resolved 2026-11-05 by the human: keep the header and repeat the statement in the results write-up after the sweep (new Execution task).
- [x] [Review][Patch] Pin `preprocessing: [ant2]` in the cross-room expansion test; changing it to `[]`, `[ant4]` or `[ant2_rescale]` passes today, and only the sibling cross-room test pins it [tests/test_sweep.py:451]
- [x] [Review][Patch] The header reads the result as a room effect, but `bedroom→living_room` and `bedroom→laboratory` also change day and person, and S4/S5 train with a bookcase blocking the direct path. State that the drop is not a pure room effect [experiments/padova_cross_room.yaml:9]
- [x] [Review][Patch] Replace the vacuous `assert filled or plan.splitter == ROOM_HOLDOUT`, which can never fail, with a plan-level check that every `room_holdout` plan has non-empty train, val and test. An all-empty holdout plan with a correct hash passes where the dataset is absent [tests/test_time_block.py:170]
- [x] [Review][Patch] State in the header that both plans have identical train and val blocks, so the columns share one bedroom training set and are not independent selections [experiments/padova_cross_room.yaml:9]
- [x] [Review][Defer] Padova `time_block` cells show `-`, so `report` gives no same-room number beside the cross-room columns, and that baseline trains on fractions 0.6/0.2/0.2 against 0.75/0.25 here [src/csi_bench/runner/report.py:421] — deferred: pre-existing, already logged; the spec leaves it unfixed
- [x] [Review][Defer] The spec's Verification command does not pass as "all pass": `ruff check` has 3 errors in `notebooks/01_classical_overview.ipynb`, `ruff format --check` flags 1 file and `ty` has 10 diagnostics, none in lines this change touches [tests/test_sweep.py:331] — deferred: pre-existing

**Rejected**

- `low`: fractions and gap are not asserted without data. The values are correct, and a changed plan gets a new hash that the real-data test rejects on the machine that runs the sweep.
- `low`: plan layout has no dataless check. The plan-level patch above covers the harmful case; the rest would add set-name parsing to tests for checkouts that cannot run the sweep.
- `low`: tests fall outside the epic's "`splits/` and `experiments/`" rule. Design Notes already settle this, following Story 6.1.
- `false`: the negative rooms test misses a crossed pair. The per-cell `c.selection.rooms == ["bedroom", test_room]` and `check_split_rooms` assertions fail on a crossed YAML pair.
- `false`: thin test sets and the `*` footnote go unflagged. `report` marks those cells, `tests/test_metrics.py:84` and `tests/test_report.py:434` cover the flag, and the open sweep task already records per-class caveats.
- Fix edits the spec under review: the "no code review" wording in the Decision and Implementation Notes no longer matches this review.

## Implementation Notes

**2026-11-05, implemented directly, no subagent. Code review skipped and nothing staged or committed, at the human's request. The spec stays `in-progress` until the sweep and `report` are done.**
- `tests/test_time_block.py`: `test_committed_plans_are_consistent` required every recording to fall in a part of its plan. Padova's third room breaks that by design, because `room_holdout` leaves recordings outside the plan's rooms all empty. The check now allows this for `room_holdout` only.
- One-cell smoke run in a scratch store (`bedroom→laboratory`, `amplitude_stats`, `knn`, `w50`, dirty tree): done, and `report --include-dirty` showed the `padova / room_holdout` section with intervals and the `*` footnote. Test macro-F1 18.88 [18.38, 19.39]. This checks the wiring only; the real store has no cross-room runs yet.
- The results write-up follows the sweep. Still open: the 144-run sweep, `report`, and recording the table and run time here.

## Spec Change Log

## Review Triage Log

## Design Notes

The tests are outside the story's "`splits/` and `experiments/`" rule on a literal reading. As in Story 6.1, the rule is read as guarding `src/`: pinning plans and sweep expansion matches what 6.1's review required.

The deferred note on 234 non-reproducing runs asks to settle that "before the Story 6.2 sweep". It is not done here: those mismatches are only in `lwm` `mlp` and `logreg`, and this grid uses `svm`, `knn` and `random_forest`, with 0 mismatches on any feature set.

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench describe padova` -- expected: still uses `b6010d87ce4d`, no errors
- `uv run csi-bench sweep experiments/padova_cross_room.yaml --workers 4` -- expected: 144 of 144 runs ok
- `uv run csi-bench report` -- expected: `padova / room_holdout` section with both direction columns
