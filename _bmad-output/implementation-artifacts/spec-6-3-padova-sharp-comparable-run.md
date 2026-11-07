---
title: 'Padova SHARP-comparable run'
type: 'feature'
created: '2026-11-07'
status: 'in-progress'
baseline_commit: '3047aadef4523711065d456f7ad6058ac9d0d807'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Padova same-room macro-F1 (0.53–0.78, 8 classes, S1–S5, 20–50 packet windows) sits far below SHARP's 96–99.8 % accuracy, and none of the protocol differences (class set, data subset, window length) has been tested as the cause.

**Approach:** Register a second dataset `padova_sharp` in `adapters/datasets/padova.py` that loads only S1 with SHARP's 5 classes, commit its `time_block` plan (0.6/0.2/0.2), add `experiments/padova_sharp.yaml` covering windows 20–50 and 346, and after the sweep record per change how much of the gap closed.

## Boundaries & Constraints

**Always:**
- `padova_sharp` shares `Padova`'s loading code and on-disk cache; classes are exactly `(empty, sitting, walking, running, jumping)`; recordings of other activities and sets are skipped, not errors; `recording_id` starts with `padova_sharp/`.
- Plan params: `TimeBlock` defaults (`fractions 0.6/0.2/0.2`, `gap_packets 64`).
- Sweep: `stride: 9`, `seeds: [0]`, `role: comparison`, `selection.rooms: [bedroom]`, heads `svm, knn, random_forest`, selection `null` or mRMR k [2, 3]. `amplitude_stats` runs under `ant2` and `ant4`; `paper_beamforming` and both-sets run under `ant2` only (paired axis).
- **Decision (2026-11-07, human):** also register `padova_s1` (S1 only, all 8 `CLASSES` in `padova` order, own plan) so the write-up separates the steps bedroom → S1 (data subset) → 5 classes (class set) → 346 (window length). Its sweep item copies the Padova baseline grid without the room axis: 72 runs.
- **Decision (2026-11-07, human):** fix the deferred `report` `-` cells: a `time_block` section whose rows are not `r1/r2/mixed` takes its columns from the rows' room settings. This knowingly goes beyond the epic's "loader only" diff rule for `runner/report.py`. `tor_vergata` output stays identical.
- **Decision (default, same as Story 6.2 and the human's instruction for this run):** the human commits, runs the sweep and `report`. Implementation stops before them, with no code review, `git add` or commit.

**Never:**
- No change to `Padova`'s registered behaviour, `padova`'s VERSION, its committed plans or existing runs.
- No new feature family (Doppler is only assessed in writing), no LWM sweep, no label mapping, no resampling.
- No fitting or selecting on test data.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Load | `data/padova` with S1–S7 | 18 recordings (S1a–c × E, L, W, R, J1, J2), room `bedroom`, labels 0–4 | N/A |
| Label order | `S1a_L` | label 1 (`sitting`); `S1a_E` → 0, `S1a_J2` → 4 | N/A |
| Skipped files | `S1a_C`, `S2a_W`, `S7a_E` | not loaded | N/A |
| No S1 data | root has only S2–S7 | — | `DataError` |
| `padova_s1` load | same root | 27 recordings (S1a–c × all 9 files), labels as `padova` | N/A |
| Report columns | `padova_sharp` rows, room `bedroom` | section column `bedroom` with values, not `-` | N/A |
| Implicit split | `describe padova_sharp` without `--split` | uses the one committed block plan | N/A |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/datasets/padova.py` -- `Padova.load()` loops all `*/*.mat`, parses via `_parse` (letters `EWRJLSCH` → `CLASSES`), caches arrays by sha under `.cache/v{VERSION}`. Refactor the loop to take a file filter, a label map and an id prefix; register `padova_sharp` and `padova_s1` as subclasses with own `classes`, `VERSION = 1`, same `Params(root="data/padova")`. Keep the cache path keyed on `Padova.VERSION`, not the subclass's.
- `src/csi_bench/core/registry.py` -- `register` rejects duplicates only; several registrations per module are fine.
- `tests/contracts/dataset_source.py` -- `check_dataset_source` to call on the new source.
- `src/csi_bench/adapters/splitters/time_block.py` -- defaults are the required params. Generate the plan with `TimeBlock().split(recs, "padova_sharp")` + `core/splitplan.py` `write_split_plan` (no CLI), as in 6.2.
- `src/csi_bench/core/splitplan.py:23` -- `find_split` picks the only block plan implicitly.
- `src/csi_bench/runner/report.py:421` -- `columns()` returns `rs.ROOM_SETTINGS` for every non-holdout section. Use the sorted row settings when any row setting is outside `ROOM_SETTINGS`. Check `_comparison` (line 232) and the published row (line 238/313) still work. `tests/test_report.py` -- add a non-`r1/r2` `time_block` case.
- `src/csi_bench/config/presets.py:7` -- `ant2`, `ant4` presets.
- `src/csi_bench/config/sweep.py:68` -- `@` paired axes; `sweep:` may be a list of items (for a `padova_s1` item with its own `dataset`/`split`/windows).
- `experiments/padova_classical_baseline.yaml` -- header style and grid to copy.
- `tests/test_padova.py` -- synthetic `_write` helper and skip-if-absent real-data plan pin. `tests/test_sweep.py:411` -- expansion test pattern.
- Data: S1 recordings are 9.9k–21.2k packets, so a 346-packet window fits every 20 % block.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/adapters/datasets/padova.py` -- share the load loop; register `padova_sharp` and `padova_s1`.
- [x] `tests/test_padova.py` -- synthetic tests for the I/O matrix (filter, label order, prefix, no-S1 error, contract test) for both; skip-if-absent real test: 18 and 27 recordings, committed plans match.
- [x] `splits/padova_sharp/<hash>.json`, `splits/padova_s1/<hash>.json` -- generate on real data; record the hashes in the yaml header.
- [x] `experiments/padova_sharp.yaml` -- `sweep:` as a list of two items. Item 1 `padova_sharp`, 120 runs: windows [20, 30, 40, 50, 346] × 4 feature/preprocessing pairs × 2 selection × 3 heads. Item 2 `padova_s1`, 72 runs: windows 20–50 × 3 feature sets (`ant2`) × 2 × 3. Header states run counts, plan hashes, stride rule, 346 ≈ 2 s at 173 Hz, and the attribution chain.
- [x] `tests/test_sweep.py` -- expansion test: 192 unique ids (120 + 72), `ant4` only with `amplitude_stats` alone, window 346 only for `padova_sharp`.
- [x] `src/csi_bench/runner/report.py`, `tests/test_report.py` -- column fix and test.
- [ ] Real store (human, after committing) -- `describe padova_sharp`, `sweep`, `report`; record table and run time in Implementation Notes.
- [ ] Write-up `_bmad-output/reports/padova-sharp-6-3.md` (after sweep) -- SHARP's S1 number (arXiv 2103.09924) beside ours, gap closed per step of the chain (bedroom `padova` → `padova_s1` → `padova_sharp` ≤ 50 → `padova_sharp` 346; `ant4` reported separately), accuracy and macro-F1, whether a Doppler family is worth proposing.

**Acceptance Criteria:**
- Given the code, when `pytest` runs, then the new tests pass and the existing `padova` tests are unchanged and pass.
- Given the real data, when `describe padova_sharp` runs, then it reports 5 classes, 18 recordings in `bedroom` and writes `runs/_dataset/padova_sharp.parquet`.
- Given the finished sweep, when `report` runs, then `padova_sharp / time_block` and `padova_s1 / time_block` sections exist separate from `padova`, with values in a `bedroom` column, and the `padova / time_block` section now shows its room columns.
- Given the story closes, when `git diff --stat 3047aad -- src/` runs, then only `adapters/datasets/padova.py` and `runner/report.py` changed.

### Review Findings

- [x] [Review][Patch] Variant `VERSION` is decoupled from `Padova.VERSION` while `_load` caches under `Padova.VERSION`: bumping `Padova.VERSION` for a conversion change reconverts the variants' arrays but leaves their `dataset_version`, and so their run ids, unchanged, so stale `padova_s1`/`padova_sharp` runs would look current. Inherit `Padova.VERSION` (drop the overrides, value stays 1) and say so in a comment [src/csi_bench/adapters/datasets/padova.py:225,241]

**Rejected (code review 2026-11-07, 16):**
- false: SHARP target missing from report tables. The write-up task puts SHARP's S1 number beside ours, and the spec doesn't route it through `published_reference.yaml`.
- false: the spec isn't in the diff. It was reviewed as the spec, and committing it is the human's step by decision.
- false: a mixed `r1` + `bedroom` section is untested. No dataset produces both, and `test_reference_rows_shown_separately_with_published` already covers the tor_vergata published row.
- false: the Tor Vergata published row could print under another dataset's reference table. Only tor_vergata yields `r1/r2/mixed` settings, and the change narrows when the row is passed, not widens it.
- false: `report` exits 2 when the reference file is missing and only variant reference runs exist. This is pre-existing behaviour, and this sweep is `role: comparison` only, so it adds no reference runs.
- false: the report fix reaches beyond `time_block`. The same rule applies to any non-holdout section, every output restricted to r1/r2/mixed is unchanged, and no harm was named.
- low: concurrent workers can lose `index.json` updates. The race is pre-existing (parallel `padova` loads had it too) and costs only re-hashing.
- low (raised twice): unknown set numbers or letters raise instead of being skipped. The choice is documented in the Implementation Notes, no such file exists in S1–S7, and skipping would need parser changes.
- low: columns aren't in canonical order in mixed sections. That section can't arise.
- low: nothing pins S1 block equality across the three plans. The plans are committed and pinned per file by the real-data tests, and they match today.
- low: the test doesn't compare the base config with the baseline yaml. The base blocks are identical today, and the guard is speculative.
- low: window 346 is confounded with the gap. The spec's Design Notes already require the write-up to state it, and fixing it means a spec change.
- low: the remaining protocol differences from SHARP aren't listed. That belongs to the write-up task.
- low: the variants' `load` overrides repeat code. This is a refactor preference, and the letter order sits next to `SHARP_CLASSES` with a comment.
- low: the published-row check relies on list ≠ tuple. It's correct today, and the new test fails if published rows reappear.

## Implementation Notes

- Loader: `padova.py` now has a module-level `_load(root, labels, prefix, sets)`; `Padova`, `PadovaS1` (`padova_s1`) and `PadovaSharp` (`padova_sharp`) call it. Cache arrays stay under `.cache/v{Padova.VERSION}` and share `index.json`. File names are still validated for every file (a malformed name is an error in all three); well-formed files outside the set/letter filter are skipped. An empty result raises `DataError("no <prefix> recordings in <root>")`. SHARP letters `E,L,W,R,J` → `empty, sitting, walking, running, jumping`; `J1`/`J2` are both `jumping`.
- Plans (generated on real data with `TimeBlock().split` + `write_split_plan`): `padova_sharp` `62b4f776f13e` (18 recordings, min 9911 packets), `padova_s1` `4359a19e3c7d` (27 recordings).
- Sweep: item 0 uses an `@features` paired axis (extractors + preprocessing); item 1 uses `@dataset` (dataset + split). The CLI has no `--dry-run`; the 192-run count is pinned by `test_padova_sharp_sweep_expands`.
- Report: `columns()` returns the sorted row settings when the section is `room_holdout` or any row setting is outside `ROOM_SETTINGS`; such a reference table gets no published row. Checked on the real store: the only change in `report` output is the `padova / time_block` section, which now shows `bedroom`, `laboratory`, `living_room` values instead of `-`; `tor_vergata` sections are byte-identical.
- Verified: `ruff check src tests`, `ruff format --check`, `lint-imports` (6 kept), `pytest` 581 passed (incl. real-data skip-if-absent tests); `describe padova_sharp` shows 5 classes, 18 bedroom recordings and wrote `runs/_dataset/padova_sharp.parquet`.
- Pending (human): commit, `sweep experiments/padova_sharp.yaml`, `report`, table and run time here, write-up.
- 2026-11-07: code review skipped and nothing staged or committed, at the human's request. The spec stays `in-progress` (sprint status `in-progress`) until the sweep, `report` and write-up are done, as in Story 6.2.
- 2026-11-07: code review run afterwards (4 layers): 1 patch applied (variants inherit `Padova.VERSION`, still 1, so run ids and plans are unchanged), 16 rejected; `ruff` clean, `pytest` 581 passed. Still nothing staged or committed.

## Spec Change Log

## Review Triage Log

## Design Notes

Window 346 is longer than `gap_packets 64`. Windows never cross block edges, so no packet is shared across parts, but neighbouring blocks stay closer in time relative to the window than before; the write-up should say so.

## Verification

**Commands:**
- `uv run ruff check src tests && uv run lint-imports && uv run pytest` -- expected: pass (pre-existing notebook ruff and `ty` diagnostics excepted, as logged in 6.2)
- `uv run csi-bench describe padova_sharp` -- expected: 5 classes, 18 recordings
- `uv run csi-bench sweep experiments/padova_sharp.yaml --workers 4 --dry-run` (if supported) -- expected: 192 runs
