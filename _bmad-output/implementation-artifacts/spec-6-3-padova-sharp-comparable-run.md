---
title: 'Padova SHARP-comparable run'
type: 'feature'
created: '2026-11-07'
status: 'done'
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
- **Decision (2026-11-07, human):** add a parallel `paper_beamforming` block-context build so the sweep's cold-cache phase uses `--workers`: in `runner/run.py`, missing block contexts are computed in a fork pool before `_extract_blocks`, which then sees cache hits. No change to cache keys, run ids, `paper_beamforming` or any extractor port; output byte-identical to the serial path; memory stays copy-on-write as in `_extract_parallel` (11 GB RAM machine). This knowingly goes beyond the epic's "loader only" diff rule for `runner/run.py`.
- **Decision (2026-11-07, human): fix window 346.** Window 346 on the gap-64 plan confounds window length with block spacing. `padova_sharp` gets a second committed `time_block` plan with `gap_packets 346` (fractions 0.6/0.2/0.2). The sweep's former item 0 splits in two: gap-64 plan with windows 20–50 (96 runs), and gap-346 plan with windows [50, 346] (48 runs); window 50 on the gap-346 plan is the control that separates the gap's effect from the window's. With two block plans, `padova_sharp` has no implicit split: commands must name it (`--split`, or `split:` in a config). **Report (human choice):** `time_block` rows whose plan's `gap_packets` differs from `TimeBlock`'s default go into their own section `<dataset> / time_block, gap_packets <n>`; the mixed-hash guard keeps working per section; sections of default-gap plans, the leakage table and all `tor_vergata`/`padova` output stay identical. This goes further beyond the epic's "loader only" diff rule for `runner/report.py`.
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
| Implicit split | `describe padova_sharp` without `--split` | — (two block plans since the window-346 decision) | `DataError` naming both hashes and `--split` |
| Gap-346 report | `padova_sharp` comparison runs on both plans, column `bedroom` | two sections: `padova_sharp / time_block` and `padova_sharp / time_block, gap_packets 346`, exit 0 | N/A |
| Same-gap mix | two comparison runs in one column on different plans with equal `gap_packets` | unchanged | exit 2, mixed-hash error |

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
- `src/csi_bench/runner/run.py:392` -- `_extract`, `uses_context` branch (~line 437): builds `_block_context_getter` (line 201, get-or-compute one block via `context_cache_key` + `ContextCache`, logs `block context <ext> <rec>:<split> cached|computed`) and calls `_extract_blocks` (line 156, serial loop over `(recording_id, split)` groups). Cold-cache sweep time is dominated by serial `extractor.compute_context` (~22–135 s per block; `padova_sharp` 54 blocks, `padova_s1` 81).
- `src/csi_bench/runner/run.py:231` -- `_extract_parallel` + `_extract_chunk` + `_FORK_STATE` (line 137) + `_fork_is_safe` (line 109): the fork-pool pattern to copy. `_FORK_STATE`'s type annotation may need widening for the new tuple.
- `tests/test_run.py:1095` (getter cache test), `:1140`–`:1190` (parallel tests, `_fork_is_safe` monkeypatch), `:1320` `_shared_context_runs` (counts `compute_context` calls via a monkeypatched spy; spy calls inside forked children don't reach the parent, so check those tests still hold with the default worker count).
- Data: S1 recordings are 9.9k–21.2k packets, so a 346-packet window fits every 20 % block.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/adapters/datasets/padova.py` -- share the load loop; register `padova_sharp` and `padova_s1`.
- [x] `tests/test_padova.py` -- synthetic tests for the I/O matrix (filter, label order, prefix, no-S1 error, contract test) for both; skip-if-absent real test: 18 and 27 recordings, committed plans match.
- [x] `splits/padova_sharp/<hash>.json`, `splits/padova_s1/<hash>.json` -- generate on real data; record the hashes in the yaml header.
- [x] `experiments/padova_sharp.yaml` -- `sweep:` as a list of two items. Item 1 `padova_sharp`, 120 runs: windows [20, 30, 40, 50, 346] × 4 feature/preprocessing pairs × 2 selection × 3 heads. Item 2 `padova_s1`, 72 runs: windows 20–50 × 3 feature sets (`ant2`) × 2 × 3. Header states run counts, plan hashes, stride rule, 346 ≈ 2 s at 173 Hz, and the attribution chain.
- [x] `tests/test_sweep.py` -- expansion test: 192 unique ids (120 + 72), `ant4` only with `amplitude_stats` alone, window 346 only for `padova_sharp`.
- [x] `src/csi_bench/runner/report.py`, `tests/test_report.py` -- column fix and test.
- [x] `src/csi_bench/runner/run.py`, `tests/test_run.py` -- parallel block-context build. In `_extract`'s `uses_context` branch, when the context cache and bounds are set and the extractor has `compute_context`, before `_extract_blocks`: list the distinct `(recording_id, split)` blocks the windows need, compute each `context_cache_key` (same args as `_block_context_getter`), keep those missing from `context_cache`. New `_compute_contexts_parallel(extractor, by_id, bounds, missing, workers)` following `_extract_parallel`: set `_FORK_STATE` in the parent, fork `ProcessPoolExecutor(max_workers=min(workers, len(missing)))`, each task returns `extractor.compute_context(rec.csi[:, start:end, :])` for one block, reset `_FORK_STATE` in `finally`. Only the parent writes: `cache.put(key, ctx)` and logs `block context <ext> <rec>:<split> computed`, filling in sorted key order for deterministic logs. Serial fallback (leave everything to the existing getter) when `workers == 1`, fewer than 2 missing blocks, or `_fork_is_safe(log)` is False. `_extract_blocks`, the getter and the extractor stay unchanged. Tests: parallel vs serial give equal contexts and equal feature matrices; already-cached blocks are not recomputed; serial fallback when fork is not safe.
- [x] `splits/padova_sharp/<hash>.json` -- generate the gap-346 plan on real data with `TimeBlock(TimeBlock.Params(gap_packets=346)).split(recs, "padova_sharp")` + `write_split_plan`; pin it beside the gap-64 plan in `tests/test_padova.py` `test_real_variants` (a second gap-346 case for `PadovaSharp`).
- [x] `experiments/padova_sharp.yaml` -- split sweep item 0 into item 0 (gap-64 plan `62b4f776f13e`, `window_len: [20, 30, 40, 50]`, 96 runs) and item 1 (`split:` the gap-346 hash, `window_len: [50, 346]`, same `@features`/selection/heads, 48 runs); `padova_s1` becomes item 2. Header: both padova_sharp plans and the s1 plan, the counts 96 + 48 + 72 = 216, the chain step w50 gap 64 → w50 gap 346 (gap effect) → w346 gap 346 (window effect), and that `padova_sharp` needs an explicit split.
- [x] `tests/test_sweep.py` `test_padova_sharp_sweep_expands` -- 216 unique ids; `padova_sharp` runs are 96 on `62b4f776f13e` with windows {20, 30, 40, 50} and 48 on the gap-346 plan with windows {50, 346}; 72 `padova_s1`; `find_split("padova_sharp")` raises `DataError`, each named plan is `time_block` with the expected `gap_packets`; `ant4` only with `amplitude_stats` and `padova_sharp`, 36 runs; the s1 grid check reads item 2.
- [x] `src/csi_bench/runner/report.py`, `tests/test_report.py` -- in `report()`, when building `sections`, a `time_block` row whose plan (`find_split(dataset, row["split_hash"], split_root)` + `read_plan_params`) has `gap_packets` other than `TimeBlock.Params().gap_packets` gets the section key `(dataset, "time_block, gap_packets <n>")`; an unreadable or missing plan keeps the plain key. Respect the import-linter contracts when reading the default. Tests: the two matrix rows "Gap-346 report" and "Same-gap mix"; existing report tests unchanged and passing.
- [x] Real store (human, after committing) -- `describe padova_sharp`, `sweep`, `report`; record table and run time in Implementation Notes.
- [x] Write-up `_bmad-output/reports/padova-sharp-6-3.md` (after sweep) -- SHARP's S1 number (arXiv 2103.09924) beside ours, gap closed per step of the chain (bedroom `padova` → `padova_s1` → `padova_sharp` ≤ 50 → `padova_sharp` w50 on the gap-346 plan → `padova_sharp` 346 on the gap-346 plan; `ant4` reported separately), accuracy and macro-F1, whether a Doppler family is worth proposing.

**Acceptance Criteria:**
- Given the code, when `pytest` runs, then the new tests pass and the existing `padova` tests are unchanged and pass.
- Given the real data, when `describe padova_sharp --split 62b4f776f13e` runs, then it reports 5 classes, 18 recordings in `bedroom` and writes `runs/_dataset/padova_sharp.parquet`.
- Given the finished sweep, when `report` runs, then `padova_sharp / time_block` and `padova_s1 / time_block` sections exist separate from `padova`, with values in a `bedroom` column, the gap-346 runs appear in a separate `padova_sharp / time_block, gap_packets 346` section, and the `padova / time_block` section now shows its room columns.
- Given the story closes, when `git diff --stat 3047aad -- src/` runs, then only `adapters/datasets/padova.py`, `runner/report.py`, `runner/rescore.py` and `runner/run.py` changed.
- Given an uncached `paper_beamforming` run with `workers > 1` and fork safe, when `_extract` runs, then missing block contexts are computed in a fork pool, written by the parent only, and the contexts and feature matrix equal the serial path's byte for byte; cached blocks are not recomputed; with `workers == 1`, < 2 missing blocks or fork unsafe, the serial path runs.

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

**Code review 2 (2026-11-07, diff `3047aad`..working tree, incl. the uncommitted parallel block-context build):**

- [x] [Review][Patch] A failure in the fork pool discards every finished context. `_compute_contexts_parallel` gathers `list(pool.map(...))` and only `put`s after the pool ends. When one task raises, the `with` exit still waits for the remaining tasks to run, and then nothing is cached, so a rerun recomputes every block (about 22–135 s each, up to 81 blocks). Fix: `put` and log each result inside the `with` block as `pool.map` yields it, which stays in key order [src/csi_bench/runner/run.py:375-383]
- [x] [Review][Defer] Pool memory headroom is unverified, and a child killed by OOM aborts the run with no serial fallback (`BrokenProcessPool`) [src/csi_bench/runner/run.py:375] — deferred: the claim is unverified and would be medium. The cold-cache `padova_sharp` sweep with `--workers > 1` settles it: check `max_child_peak_rss_mb` in the extraction log lines against the 11 GB machine.

**Rejected (code review 2, 26):**
- false: an unpicklable worker exception changes the error type. `ExtractorError`, `DataError` and `ConfigError` are plain `Exception` subclasses, so they pickle unchanged.
- low: BLAS/OpenMP oversubscription in the forked children. This is the same pattern as the existing `_extract_parallel`, with no measured slowdown.
- low: a concurrent run writes the same cache key between check and put. The values are deterministic and `put` is atomic, so at worst the work is duplicated.
- false: `test_parallel_block_contexts_skip_cached_blocks` fails with an `IndexError` if r1 has fewer than 2 blocks. `time_block` gives 3 blocks per recording, so the pool always runs for r1.
- low: the skip-cached test assumes r2 has as many blocks as r1. The fixture is symmetric and the test passes; a change to the fixture fails it loudly.
- false: the spy breaks on keyword arguments. The only call site passes positional arguments.
- false: other log lines match `" computed"` / `" cached"`. Only the two `block context` messages contain those words.
- low: a set name with leading zeros (`S01a`) parses as set 1. No such file exists in S1–S7.
- reject (the fix edits the spec): the Implementation Notes order and `review_loop_iteration` are stale. This review now covers the parallel work.
- reject (the fix edits the spec): the frozen-block decisions have no Spec Change Log entry.
- reject (the fix edits the spec): the Verification section lists a `--dry-run` that does not exist.
- low: on a warm cache, `_missing_contexts` reads each entry once more than before. Contexts are about 0.3 MB and there are at most 81, and the notes acknowledge it. Fixing it needs a new `exists` method on the port.
- low: `_FORK_STATE: tuple[Any, ...]` loses tuple typing for both pool users. A mismatch fails at once in the tests, and the fix means restructuring the globals.
- false: the `key not in missing` dedupe is a defect. It is redundant, since keys include the recording and the bounds, but it does no harm.
- low: the run log shows `computed` and then `cached` for each pooled block, with no pool marker. This is by design and documented, and the extraction time on the `path=blocks` line includes the pool phase.
- false: the split plans are missing from the review. They were excluded from the patch as generated files; the Acceptance Auditor read them (18 and 27 recordings, `time_block` 0.6/0.2/0.2, gap 64).
- low: each loader hardcodes its `recording_id` prefix separately from its registered name. This was already rejected in review 1 as a refactor preference.
- false: nothing asserts that `paper_beamforming` never gets `ant4`. The `ids == [[0, 1, 2, 3]]` branch asserts `names == ["amplitude_stats"]`.
- false: in the fallback test, `one_missing` might delete a block the run does not need. The same rooms are re-run, and `len(computed) == 1` fails if the deleted block is not needed.
- low: the real `_fork_is_safe` is not tested under pytest, and the fully warm cache with `workers > 1` has no test. The existing parallel tests use the same monkeypatch; with a warm cache `missing` is empty, so the pool is skipped by the `>= 2` check.
- low: `_compute_contexts_parallel` takes `spec`, `cache` and `log` beyond the spec's signature. The parent needs them to write and log.
- low (same as the warm-cache double read): when fork is unsafe the cache is still read before falling back. Checking `_fork_is_safe` first would log a spurious thread warning on warm caches.
- low: the cache-key arguments are duplicated between `_missing_contexts` and the getter. If they drift, the log-count assertions in `test_parallel_block_contexts_match_serial_path` fail.
- low: no test pins that only the parent writes. `_compute_context_task` only returns a value, so this holds by construction.
- low: the `padova / time_block` part of AC3 is tested only on synthetic `padova_sharp` rows. It is the same code path, and the notes record that it was checked on the real store.
- false: the real-store and write-up items are open. They are the human's steps by decision, and the status is `in-progress`.

**Code review 4 (2026-11-07, diff `3047aad`..working tree incl. untracked `splits/padova_sharp/7640fb8ad320.json`, 4 layers; Verification Gap: no gaps):**

- [x] [Review][Decision] 24 stale w346 runs on the gap-64 plan contaminate the `padova_sharp / time_block` section — `runs/padova_sharp/time_block` holds 24 `*-w346--*` runs with `split: 62b4f776f13e` from the earlier 192-run sweep (checked on disk). `_section_key` puts them in the default gap-64 section, where they can win best-variant and bring back the window/gap confound the window-346 decision removed (AC3's chain step "`padova_sharp` w20–50, gap 64"). Only an Implementation Notes line asks the human to move them; nothing in code catches a `time_block` run with `window_len` > the plan's `gap_packets`. Options: move the runs out by hand before `report`, or add a guard in `report` (warn or exclude), or both. **Resolved (human, 2026-11-07): moved by hand, no code guard** — the 24 runs now sit in `runs_archive/padova_sharp_gap64_w346/` (git-ignored); `runs/padova_sharp/time_block` holds 96 runs, none w346. **Correction (same day): the finding was false.** All 24 runs had `status: failed` (`ConfigError: window_len 346 exceeds plan.gap_packets 64`, raised by `core/windows.py` `build_windows`), and `report` shows only `done` runs, so they never reached a section; `build_windows` already guards window > gap. The move is harmless.

**Rejected (code review 4, 28):**
- low: a failing pool task makes the `with` exit wait for every remaining task, and later blocks that finished are dropped (stored in key order). Extractor failures are rare and deterministic; the fix restructures to `submit`/`as_completed` + `cancel_futures`, and the key-order behaviour is pinned by review 2's test. A failing `cache.put` (disk full) behaves the same way.
- false: `_section_key` mis-sections rows with no `split_hash`. `report` drops such rows from `shown` before sectioning, and `rescore._cells` filters them too.
- false: `_plan_gap` lets `TypeError`/JSON/`OSError` through. `read_plan_params` wraps all of them, and a non-mapping `params`, as `DataError`.
- reject (the fix edits the spec): stale 192-run task lines, Code Map ("only block plan implicitly"), Implementation Notes (item numbering, "parent holds all contexts", "nothing staged or committed", order, test counts), `review_loop_iteration`, and the Verification section missing report/`--split`/`ruff format` checks.
- false: sections should key on all non-default `TimeBlock` params, not `gap_packets` alone. Keying on gap is the human's report decision, and a fractions-only clash still exits loudly through the per-section mixed-hash guard.
- low: a missing plan under `split_root` folds gap-346 rows into the default section, followed by a confusing hash-clash error. The plans are committed, and the failure is loud.
- false: a float `gap_packets` (346.0) is read as None. Plans are written by `TimeBlock` with an int field.
- reject (already deferred in deferred-work.md): the leakage table ignores non-default-gap sections; `BrokenProcessPool` has no serial fallback.
- low: the warm-cache `_missing_contexts` probe loads each context twice. Contexts are `(6, B-16)`, about 0.3 MB; already rejected in review 2 (it needs a port method).
- low: finished contexts pile up in the parent behind a slow early block under `pool.map`. Contexts are about 0.3 MB and there are at most 81.
- false: `_load` runs strict `_parse` before the set filter, so a bad file in another set breaks the variants. The behaviour is the same as `padova`'s, intended, and no such file exists.
- low: report section order follows row order, and the gap-section test slice assumes the default section prints first. Order was already insertion-based for datasets; the fixture is deterministic and a reorder fails the test loudly.
- low: no test drives `describe padova_sharp` without `--split` to pin the `--split` hint wording. The hint is a constant in `cli.py`, and `find_split`'s error naming both hashes is tested.
- low: no test pins that the parallel path stays off on a warm cache with `workers > 1`. `missing` is empty, so the `>= 2` check skips the pool (also raised in review 2).
- low: `TimeBlock.Params()` is built per `time_block` row in `_section_key`. The cost is negligible.
- low: `PadovaS1.classes = CLASSES` restates the inherited value. Cosmetic.
- low: the new deferred-work entries use two formats. Cosmetic.
- reject (human step by decision): `splits/padova_sharp/7640fb8ad320.json` is untracked. The human commits it with the rest; item 1 and `test_real_variants` need it.
- false: AC2/AC3 can't be checked from the diff. They are the human's sweep and `report` steps, and the status is `in-progress`.
- reject (merged into the Decision above): window > gap has no code-level guard; AC3 depends on moving runs by hand.

## Implementation Notes

- Loader: `padova.py` now has a module-level `_load(root, labels, prefix, sets)`; `Padova`, `PadovaS1` (`padova_s1`) and `PadovaSharp` (`padova_sharp`) call it. Cache arrays stay under `.cache/v{Padova.VERSION}` and share `index.json`. File names are still validated for every file (a malformed name is an error in all three); well-formed files outside the set/letter filter are skipped. An empty result raises `DataError("no <prefix> recordings in <root>")`. SHARP letters `E,L,W,R,J` → `empty, sitting, walking, running, jumping`; `J1`/`J2` are both `jumping`.
- Plans (generated on real data with `TimeBlock().split` + `write_split_plan`): `padova_sharp` `62b4f776f13e` (18 recordings, min 9911 packets), `padova_s1` `4359a19e3c7d` (27 recordings).
- Sweep: item 0 uses an `@features` paired axis (extractors + preprocessing); item 1 uses `@dataset` (dataset + split). The CLI has no `--dry-run`; the 192-run count is pinned by `test_padova_sharp_sweep_expands`.
- Report: `columns()` returns the sorted row settings when the section is `room_holdout` or any row setting is outside `ROOM_SETTINGS`; such a reference table gets no published row. Checked on the real store: the only change in `report` output is the `padova / time_block` section, which now shows `bedroom`, `laboratory`, `living_room` values instead of `-`; `tor_vergata` sections are byte-identical.
- Verified: `ruff check src tests`, `ruff format --check`, `lint-imports` (6 kept), `pytest` 581 passed (incl. real-data skip-if-absent tests); `describe padova_sharp` shows 5 classes, 18 bedroom recordings and wrote `runs/_dataset/padova_sharp.parquet`.
- Parallel block contexts: `_extract`'s `uses_context` branch, when `workers > 1` and the context cache, bounds and `compute_context` are present, calls `_missing_contexts` (distinct `(recording_id, split)` blocks of the windows, keyed with the getter's `context_cache_key` args, kept if `cache.get` misses, sorted by key) and, with ≥ 2 missing and `_fork_is_safe`, `_compute_contexts_parallel` (fork pool of `min(workers, missing)`, `_FORK_STATE = (extractor, by_id)` widened to `tuple[Any, ...]`, each task returns `compute_context` of one block; the parent `put`s and logs `computed` in key order). `_extract_blocks`, the getter and the extractor are unchanged; the getter then logs `cached` for those blocks. The parent holds all new contexts until the pool ends (each is `(6, B-16)` float, ~0.3 MB, so ≤ 81 blocks is negligible). When nothing is missing, `_missing_contexts` reads each cached entry once more than before. Tests: `test_parallel_block_contexts_match_serial_path` (cached `.npz` contents, feature matrix bytes and metrics equal to `--workers 1`), `test_parallel_block_contexts_skip_cached_blocks` (r1 cached → only r2 computed), `test_parallel_block_contexts_fall_back_to_serial` (fork unsafe / 1 worker / 1 missing block). Existing `_shared_context_runs` tests use `workers=1` and are unaffected. `ruff`, `lint-imports` (6 kept), `pytest` 586 passed. Not timed on real data.
- 2026-11-07: window-346 change built and reviewed (code review 3, 3 layers: 2 patches applied, 1 deferred, 7 rejected; `ruff`, `lint-imports` 6 kept, `pytest` 591 passed). Before `report`, move the 24 old w346 runs on the gap-64 plan out of `runs/padova_sharp/time_block`. Not staged or committed.
- Pending (human): commit, `sweep experiments/padova_sharp.yaml`, `report`, table and run time here, write-up.
- 2026-11-07 (window-346 fix): gap-346 plan `splits/padova_sharp/7640fb8ad320.json` (`TimeBlock(Params(gap_packets=346))`, 18 recordings), pinned in `test_real_variants` beside `62b4f776f13e`. `experiments/padova_sharp.yaml`: item 0 gap-64 windows 20–50 (96), item 1 `split: ["7640fb8ad320"]` windows [50, 346] (48), item 2 `padova_s1` (72), 216 runs; header updated. `report.py`: `_section_key` reads the row plan's `gap_packets` (`find_split` + `read_plan_params`, cached per `(dataset, split_hash)`) and keys non-default-gap `time_block` rows as `(dataset, "time_block, gap_packets <n>")`; missing/unreadable plans keep the plain key; the default comes from `TimeBlock.Params().gap_packets` (runner may import adapters, `lint-imports` 6 kept). The leakage table still reads only the plain `time_block` section. Tests: `test_time_block_plans_with_another_gap_get_their_own_section`, `test_same_gap_plans_in_one_column_still_refused`. `describe padova_sharp` without `--split` now fails naming both hashes; with `--split 62b4f776f13e` 5 classes, 18 bedroom recordings. Real-store `report` output byte-identical before/after the change (no gap-346 runs yet). `ruff check`/`format --check` clean, `pytest` 590 passed.
- Real store caveat (human): `runs/padova_sharp/time_block` holds 24 w346 runs from the former 192-run sweep, on the gap-64 plan. They are no longer in the yaml but still sit in the `padova_sharp / time_block` section and can be picked as best variant there; move them out of the store before `report` if the section should show only windows 20–50.
- 2026-11-07: code review skipped and nothing staged or committed, at the human's request. The spec stays `in-progress` (sprint status `in-progress`) until the sweep, `report` and write-up are done, as in Story 6.2.
- 2026-11-07: code review run afterwards (4 layers): 1 patch applied (variants inherit `Padova.VERSION`, still 1, so run ids and plans are unchanged), 16 rejected; `ruff` clean, `pytest` 581 passed. Still nothing staged or committed.
- 2026-11-07: code review 2 (4 layers, incl. the parallel block-context build): 1 patch applied (`_compute_contexts_parallel` now `put`s each context as `pool.map` yields it, so a failing task keeps the blocks before it; new `test_parallel_block_contexts_keep_finished_blocks_when_a_task_fails`), 1 deferred (pool memory headroom, settled by the cold-cache sweep's `max_child_peak_rss_mb`), 26 rejected; `ruff` clean, `lint-imports` 6 kept, `pytest` 587 passed. Still nothing staged or committed; status stays `in-progress` until the sweep, `report` and write-up.

- 2026-11-07 (real store, story closed): sweep 216/216 `done`, none dirty. `padova_sharp` gap 64 (96 runs, `737986f`/`8962d69`, 09:07–10:53 UTC, 36 min of run time), `padova_s1` (72, `737986f`, 10:53–12:53, 120 min), `padova_sharp` gap 346 (48, `738f1b9`, 13:58–14:50, 51 min; the 54 cold block contexts took about 23 min with the parallel pool at `--workers 8`, peak child RSS 5232 MB, which settles the deferred pool-memory item). The 48 gap-346 runs first recorded `dirty: true` at `d67bef6` because this spec was edited mid-sweep; their recorded diff is byte-identical to `d67bef6..738f1b9`, so provenance was set to `738f1b9`, clean (originals in `runs_archive/provenance_backup_gap346_dirty/`). The 24 failed w346 gap-64 runs from the 192-run sweep sit in `runs_archive/padova_sharp_gap64_w346/`. `report` exits 0 with sections `padova_s1 / time_block`, `padova_sharp / time_block`, `padova_sharp / time_block, gap_packets 346`; saved as `_bmad-output/reports/padova-sharp-6-3.txt`. Chain, best by val (both / none / `random_forest`), test macro-F1 / accuracy: `padova` bedroom 61.87 / 63.40 → `padova_s1` 69.84 / 69.85 → `padova_sharp` w≤50 81.61 / 81.55 → gap 346 w50 80.82 / 80.79 → w346 89.40 / 89.50, against SHARP S1 99.59 % accuracy, F1 0.996 (Table 3). Write-up: `_bmad-output/reports/padova-sharp-6-3.md` (about 72 % of the accuracy gap closed; class set +11.70, window +8.71, S1 +6.45, gap ≈ 0; `ant4` no gain; Doppler family worth proposing as its own story).

## Spec Change Log

- 2026-11-07 (human): window 346 fixed with a second `padova_sharp` plan (`gap_packets 346`) and a w50 control on it; sweep item 0 split (192 → 216 runs); `report` gives non-default-gap `time_block` plans their own section (human chose this over separate run stores); implicit split for `padova_sharp` removed by consequence. Frozen block, matrix, tasks, ACs, Design Notes and Verification updated.
- 2026-11-07 (code review 3): `runner/rescore.py` `_cells` now groups by `report._section_key`, so `rescore-report` keeps the gap-346 section separate like `report`. This follows from the human's report decision; the diff-stat AC now lists `rescore.py`.

## Review Triage Log

Code review 3 (2026-11-07, window-346 change, diff HEAD..working tree, 3 layers):

| Finding | Verdict | Route | Evidence |
|---|---|---|---|
| `rescore-report` `_cells` groups by `(dataset, splitter, role)`, not `report`'s gap sections | medium | patch | `runner/rescore.py:185`; mixed gap-64/346 winners in one table |
| yaml header chain lines misaligned, one lacks "below" | low | patch | direct correction |
| leakage ignores non-default-gap block runs, untested | low (unreachable) | defer | only tor_vergata has `random_window`, all gap 64 |
| gap-346 section's best variant spans w50 and w346 | low | reject | same as before for item 0; write-up reads per-window via `--all-variants` |
| missing plan falls back to default section with misleading error | low | reject | plans committed; still exits 2 naming hashes; fix adds a branch |
| plan file and spec not in diff | false | reject | excluded on purpose; human commits |
| section key depends on `TimeBlock` default; only gap gets a section | false | reject | design per Design Notes |
| gap sections sort lexically | low | reject | one non-default gap exists |
| report test slices by section order | low | reject | fixture-stable; fix adds parsing |
| no check that item 1 grid equals item 0 | low | reject | adds a test; drift unlikely |

## Design Notes

Window 346 is longer than `gap_packets 64`. Windows never cross block edges, so no packet is shared across parts, but neighbouring blocks stay closer in time relative to the window. Since the window-346 decision, w346 runs use a plan with `gap_packets 346`, so the gap is never shorter than the window; w50 on that plan isolates the effect of the larger gap (and the slightly smaller blocks) from the effect of the window length.

Report sections key on plan `gap_packets`, not on the raw split hash: two plans with the same params in one column still signal an accidental mix (e.g. a stale plan) and keep the exit-2 guard.

## Verification

**Commands:**
- `uv run ruff check src tests && uv run lint-imports && uv run pytest` -- expected: pass (pre-existing notebook ruff and `ty` diagnostics excepted, as logged in 6.2)
- `uv run csi-bench describe padova_sharp --split 62b4f776f13e` -- expected: 5 classes, 18 recordings
- `uv run pytest tests/test_sweep.py -k padova_sharp` -- expected: 216 runs pinned
