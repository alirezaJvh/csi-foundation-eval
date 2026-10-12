---
title: '`room_holdout` splitter'
type: 'feature'
created: '2026-10-11'
status: 'done'
route: 'dispatch'
baseline_commit: '0c330255adc17c0654391683e1e47cdb74270f8e'
baseline_tree_pre_story: '112686e675cd47280cba24043496e35f65c9a863' # working tree incl. uncommitted 4.1
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-4-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every split today trains and tests inside the same rooms. Within-room scores are at 98.5–99.9%, so they no longer separate methods, and there is no strict out-of-distribution protocol.

**Approach:** Add a `room_holdout` adapter behind the existing `Splitter` port. It time-blocks the train-room recordings into `train` and `val` (same gap rule as `time_block`) and puts every test-room recording into `test` whole. Commit the r1→r2 and r2→r1 plans. Validate `selection.rooms` against the plan, and give the report a `room_holdout` section with direction columns.

## Boundaries & Constraints

**Always:** Params are `train_rooms`, `test_rooms` (non-empty and disjoint, else `ConfigError`), `fractions: (train, val)` (both > 0, summing to 1; default `(0.75, 0.25)`, the 3:1 train:val ratio of `time_block`) and `gap_packets` (default 64). A train-room recording gets `train=(0,t)`, `val=(t+gap, t+gap+v)` with `usable = n - gap`, `v = floor(f_val·usable)` and `t = usable - v`, which is too short → `DataError`, and `test=(0,0)`. A test-room recording gets `test=(0,n)`, `train=val=(0,0)`. A recording in neither room set gets all three spans `(0,0)`. `(0,0)` means an empty part. A listed room missing from the recordings → `ConfigError`. A class with no packets in `train`, `val` or `test` → `DataError`. The plan's `unit` is `block`. A `room_holdout` plan is never chosen implicitly: `find_split` without `split` ignores it, so configs without `split:` and their `run_id`s are unchanged. With a `room_holdout` plan, `set(selection.rooms)` must equal `train_rooms ∪ test_rooms`. Config validation checks this when the plan resolves, and `run` checks it again after loading the plan; otherwise → `ConfigError`. Report column of a `room_holdout` run = `rs.room_setting(train_rooms) + "→" + rs.room_setting(test_rooms)` (e.g. `r1→r2`), read from the committed plan. It goes into the index `room_setting` field and drives the per-column hash check. The run slug's rooms part is the same direction, written `r1_to_r2`. `room_holdout` sections show only the direction columns present, with no published row. Committed plans use the default params: `{train_rooms:[r1], test_rooms:[r2]}` and the reverse.

**Never:** Add a new port. Let validation, preprocessing fits or selection see test-room packets. Change existing plans, their hashes, or existing `run_id`s. Add intervals (4.3) or the cross-room experiment (4.4). Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| r1→r2 plan | 19 tor_vergata recordings | r1: train+val blocks; r2: whole-recording test | — |
| Same rooms | `train_rooms` ∩ `test_rooms` ≠ ∅ | rejected | `ConfigError` |
| Missing class | a class absent from the test room | rejected, naming class and part | `DataError` |
| Implicit split | config without `split:`, plus 2 holdout plans | the time-block plan is chosen, as today | — |
| Rooms mismatch | holdout split, `selection.rooms: [r1]` | rejected, naming both sets | `ConfigError` |
| describe | `describe tor_vergata --split <holdout>` | table with 0 train/val for test-room rows | — |
| Report | finished runs on both directions | `tor_vergata / room_holdout` section, columns `r1→r2` and `r2→r1` | — |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/splitters/time_block.py` -- the template: `@register`, `VERSION`, a pydantic `Params`, validation in `__init__`, sorted iteration, `compute_split_hash`. The new `room_holdout.py` copies this shape.
- `src/csi_bench/core/splitplan.py:22` `find_split` -- the implicit branch keeps block plans; also drop `read_plan_splitter(f) == ROOM_HOLDOUT`. Add the constant `ROOM_HOLDOUT = "room_holdout"` (the adapter registers under it), `read_plan_params(path)` (sibling of `read_plan_splitter`), and `check_split_rooms(splitter, params, rooms)` next to `check_split_role`.
- `src/csi_bench/config/experiment.py:159` `validate_registry` -- next to `check_split_role`, read the splitter and params of the resolved plan and call `check_split_rooms`. If the plan does not resolve, skip as today.
- `src/csi_bench/runner/run.py:585` `_execute` -- when the splitter is `room_holdout`, pass the direction to `rs.run_slug`. After `load_split_plan` at line 616, call `check_split_rooms(plan.splitter, plan.params, config.selection.rooms)`.
- `src/csi_bench/core/results_schema.py:56,252` -- `run_slug(config, rooms: str | None = None)`, which writes `→` as `_to_` in the slug; add `room_direction(train, test)`. `results_schema` may import from `splitplan` (no cycle).
- `src/csi_bench/runner/report.py:44` `_index_row` -- for `splitter == ROOM_HOLDOUT`, set `room_setting` from `read_plan_params(find_split(dataset, config["split"], split_root))`. On `DataError`, keep `rs.room_setting`. Thread `split_root="splits"` through `refresh_index`/`report`. `_comparison` takes `columns`: `rs.ROOM_SETTINGS`, or the sorted directions for a `room_holdout` section. `_section` passes `published=None` there.
- Needs no change: `core/windows.py` (an empty span yields no windows), `cli.build_table` (counts 0), `_extract_blocks` (it only visits blocks that have windows), `preprocess.train_view`. Note for 4.4: `recording_mean_norm` with `fit_scope: train` already raises `DataError("no data in <test-room id>")`, because test-room recordings have no train block.
- `tests/contracts/splitter.py` -- allow `start == end` (an empty part). Check order and gap only between non-empty spans. Keep `0 <= start <= end <= n`.
- `tests/test_time_block.py:140` `test_committed_plans_are_consistent` -- same relaxation, since it iterates over every committed plan.
- `tests/test_random_window.py:158` -- builds recordings with `n_packets = max(end)`; this still holds for holdout plans (their spans end at `n`).

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/splitplan.py` -- `ROOM_HOLDOUT`, `read_plan_params`, `check_split_rooms`, and leave holdout plans out of the implicit `find_split` -- the shared rules
- [x] `src/csi_bench/adapters/splitters/room_holdout.py` -- the splitter -- the protocol itself
- [x] `tests/contracts/splitter.py`, `tests/test_time_block.py` -- empty-span relaxation -- the contract covers holdout plans
- [x] `splits/tor_vergata/<hash>.json` ×2 -- generate both directions from the real data with `write_split_plan` -- committed plans
- [x] `src/csi_bench/config/experiment.py`, `src/csi_bench/runner/run.py` -- rooms rule at validation and run time; direction slug -- strict selection
- [x] `src/csi_bench/core/results_schema.py`, `src/csi_bench/runner/report.py` -- direction column and holdout section -- report AC
- [x] `tests/test_room_holdout.py` (new), `tests/test_experiment_config.py`, `tests/test_report.py`, `tests/test_describe.py` -- the I/O matrix, the contract test, a real-data test pinning both committed files (skipped without data), distinct hashes, implicit `find_split` -- coverage

**Acceptance Criteria:**
- Given a `room_holdout` comparison config on a tiny fixture, when `run` finishes, then val and test metrics exist, no `val` window comes from a test-room recording, and every `test` window does.
- Given the real store, when `report` runs, then the `time_block` and `random_window` sections and every existing `run_id` are unchanged.

## Implementation Notes

- `write_split_plan` now omits `unit` when it is `block`, mirroring `compute_split_hash`, so newly written block plans match the committed convention checked by `test_random_window.py::test_committed_plans_keep_hash_and_load`. Hashes are unaffected.
- Committed plans: `1e8688472a1f` (r1→r2), `c4faaf799ad5` (r2→r1).
- A train-room recording whose `val` or `train` length comes out ≤ 0 raises `DataError` (not only `usable <= 0`), so a non-`(0,0)` span is never empty.
- `find_split`'s implicit branch treats a plan whose `splitter` cannot be read as not-holdout (old/minimal plans keep working); it is reported when loaded.

## Spec Change Log

## Review Triage Log

### Review Findings

Code review 2026-10-11 (blind-hunter, edge-case-hunter, verification-gap, acceptance-auditor).

- [x] [Review][Decision] `room_holdout` accepts `role: reference`, so a `fit_scope: whole_recording` preprocessing fit sees test-room packets — the epic says `room_holdout` is "Valid for `role: comparison`", and the spec's Never rule forbids preprocessing fits seeing test-room packets. `validate_registry` (`experiment.py:177`) still allows `whole_recording` for any reference run, on any splitter, and `test_room_holdout_section_has_direction_columns` builds a reference holdout run (`x99`). Options: reject `role: reference` on a holdout plan; reject only `whole_recording` on a holdout plan; or accept it, since reference runs are already marked "not comparable". — resolved: option 1, `check_split_rooms` now requires `role: comparison` (config validation and `run`).
- [x] [Review][Patch] `validate_registry` reads unit, splitter and params in one `try`. A plan with a readable `unit: window` but unreadable `splitter`/`params` resets `unit` to `block` and skips `check_split_role` [src/csi_bench/config/experiment.py:165]
- [x] [Review][Patch] The splitter contract and the committed-plan test now allow empty parts for every splitter, so a `time_block` regression that empties a part would pass. Allow `start == end` only when `plan.splitter == ROOM_HOLDOUT` [tests/contracts/splitter.py:18, tests/test_time_block.py:160]
- [x] [Review][Patch] `room_holdout` columns are computed over comparison and reference rows together, so a direction present in only one table adds an all-`-` column to the other. Compute columns per table [src/csi_bench/runner/report.py:336]
- [x] [Review][Patch] `test_committed_plans_are_the_two_directions` asserts there are exactly 4 plans in `splits/tor_vergata` (and the stems are trivially unique), so any future plan breaks it. Assert that the set of `room_holdout` plans is `{R1_TO_R2, R2_TO_R1}` [tests/test_room_holdout.py:167]
- [x] [Review][Patch] No test checks that `write_split_plan` leaves out `unit` for block plans. Removing that line passes every test. Add a test that writes a time_block plan and a room_holdout plan and asserts `"unit" not in` the JSON [src/csi_bench/core/splitplan.py:113]
- [x] [Review][Defer] There is no CLI or script that regenerates the committed `room_holdout` plans; they can only be rebuilt by calling `write_split_plan` from Python [splits/tor_vergata/1e8688472a1f.json] — deferred: pre-existing, time_block plans have no generator command either

**Rejected**

- false: `write_split_plan` changed outside the task list. It is recorded in Implementation Notes, and hashes and committed files are unchanged.
- false: plan params given as strings would break `set(...)`. `Params` enforces `list[str]`, and plans are only written by code.
- low: the class-coverage error names the integer label. This matches `time_block`, and the splitter has no access to class names.
- low: class coverage only looks at the listed rooms. With a holdout plan, `selection.rooms` must equal train ∪ test, so no other class is in scope.
- low: the fallbacks in `_room_column`/`_direction` also catch `KeyError`/`TypeError`. This needs a hand-broken plan, which is unlikely.
- low: when a plan is missing, both directions fall back to `mixed` (and that adds a `mixed` column). The spec mandates this fallback; changing it means editing the spec.
- low: the `report` CLI has no `--splits` option. `run` and `describe` also resolve `splits/` relative to the working directory, so behaviour is consistent.
- low: `room_direction` reuses the `mixed` alias. That only happens for train rooms {r1, r2}, which tor_vergata cannot have with disjoint rooms.
- low: unsorted or duplicate rooms give a different hash. Fixing it would add a validator, and the committed plans are single-room.
- low: `find_split` reports "window plan needs an explicit split" when only holdout plans exist. Unlikely, since every dataset has a time_block plan.
- low: the plan file is read several times per run or report row. The cost is negligible.
- low: the first element of `fractions` is used only for validation. This mirrors `time_block` params, and changing it would change hashes.
- low: the report does not say that validation is in-domain. That is outside this story's scope.

## Design Notes

Empty parts are written as `(0,0)` rather than as missing keys, so `load_split_plan`, `PARTS` iteration and `describe` keep working unchanged. Only the contract test learns that an empty span is allowed. The direction is read from the plan rather than from the config's rooms, because both directions have the same `selection.rooms` (`[r1, r2]`), so their configs alone cannot tell r1→r2 from r2→r1.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests` -- expected: clean
- `uv run csi-bench describe tor_vergata --split <each holdout hash>` -- expected: exit 0
- `uv run csi-bench report --runs runs` -- expected: same sections as before (no holdout runs yet)
