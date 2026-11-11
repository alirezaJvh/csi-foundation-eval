---
title: 'CSI-Bench loader'
type: 'feature'
created: '2026-11-07'
status: 'done'
baseline_commit: '5620b158f68172e8df07e28d48d6868bd9ca61fb'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CSI-Bench (NeurIPS 2025 D&B) is downloaded as `data/csi_bench/CSI-Bench.zip` (79 GB, not extracted) but has no loader, so the study can't split, describe or sweep it.

**Approach:** Add a `csi_bench_hp` `DatasetSource` adapter for the HP devices' HAR files in `Multitask/sub_Human_mat/`, each `.mat` one 2-antenna complex `Recording`. Commit its `time_block` split and add a CSI-Bench copy of the classical sweep.

**Format confirmed (2026-11-07, scan of all 438 HP HAR files in the zip):**
- Layout: `Multitask/sub_Human_mat/user_<U>/act_<activity>/env_<E>/device_HP/Continuous_Data_Collection-<datetime>_<serial>_5G_concat.mat`. One `device_HP` folder holds three HP devices (serials `1B31WA000383`, `…430`, `…514`: 140/150/148 files) that record the same sessions at the same time. The `device_HP_<serial>` folders hold only non-HAR (localization) files.
- Each file: `csi_trace` struct with `csi` complex128 `(2 tx, 2 rx, 58, packets)`, `mactimer` (µs) and `SINR`. Packets 6,843–32,705 (median 21,323); none under 1,000. No non-finite values, no all-zero packet, no all-zero subcarrier; subcarrier 0 and 57 sit 20 % below the mean amplitude (band edges), no null bins.
- HAR activities: `jumping`, `running`, `seated-breathing`, `walking-case1`–`4`, `wavinghand`. Rooms: E01 (126 files), E02 (48), E03 (56), E04 (57), E05 (57), E06 (40), E27 (54); every room holds all 5 classes.
- Antenna pairing: amplitude correlation is high between streams sharing the 2nd (rx) index (tx0rx0–tx1rx0 0.50, tx0rx1–tx1rx1 0.62, mean over 438 files, same pattern per serial) and near 0 between rx0 and rx1 (−0.001). So `csi[0, 0]` and `csi[0, 1]` are two distinct receive chains of one transmitter, as assumed.
- Rate from `mactimer` (diagnostic only): median spacing 8,030 µs (124.5 Hz), effective 109–124 Hz with gaps.
- Extractor fit: `amplitude_stats` and `paper_beamforming` take any subcarrier count; LWM takes 58 (15 patches ≤ 512).

## Boundaries & Constraints

**Always:**
- Data: extract only the 438 HP HAR members from the zip, keeping their paths, so the root is `data/csi_bench/Multitask/sub_Human_mat` (`Params.root` default, extra keys forbidden). `data/csi_bench` is already git-ignored; the zip stays.
- Load only `user_*/act_<HAR>/env_*/device_HP/*.mat`; other activities and devices are skipped. Classes, in order: `(jumping, running, seated_breathing, walking, waving_hand)`; `walking-case1`–`4` → `walking`.
- Conversion: `csi_trace.csi[0]` (tx0, rows rx0 and rx1) → `(2, packets, 58)` complex64, read-only, `antenna_ids=(0, 1)`. No subcarrier dropped or shifted. `mactimer` and `SINR` unused; `sample_rate_hz=None`, `downsample_factor=None`.
- `recording_id = "csi_bench_hp/<path relative to root, no extension>"`, `room` = `E01` etc. (from `env_` folder), `source_sha256` of the `.mat` bytes.
- Same memmap cache design as `padova` (stat index → sha, arrays under `root/.cache/v{VERSION}/<sha>.npy`, atomic writes), copied, not imported.
- `DataError` naming the file when an in-scope file has no `csi_trace.csi`, is not complex, is not shaped `(2, 2, 58, packets)`, or has non-finite values or an all-zero packet; also for a missing root or no in-scope files. Out-of-scope files are skipped, never errors.
- Sweep `experiments/csi_bench_hp_classical_baseline.yaml`: `classical_baseline.yaml` grid with `dataset: csi_bench_hp`, the committed split, `stride: 6` (≈50 ms at ≈122 Hz, the Padova rule; not tuned) and the room axis.
- **Decision (2026-11-07): room axis = each room alone plus all 7 pooled**, `[[E01], [E02], [E03], [E04], [E05], [E06], [E27], [E01, …, E27]]` → 4×3×2×3×8 = 576 runs.
- **Decision (2026-11-08, amendment): long-window sweep `experiments/csi_bench_hp_long_windows.yaml`.** Windows 122, 244, 366, 610 packets (≈1, 2, 3, 5 s at ≈122 Hz), each paired with stride = half the window (61, 122, 183, 305; `@window`), on a second `time_block` plan with gap 610 (never shorter than the longest window). Features, selection, heads and room axis as above → 4×3×2×3×8 = 576 runs. Reason: 20–50 packets (0.16–0.41 s) is shorter than one step or breath, and stride 6 gives ≈1.3 M heavily overlapping pooled windows, which makes the RBF SVM impractical (one pooled SVM run took over 7 h). `csi_bench_hp_classical_baseline.yaml` stays as the cross-dataset comparison; the long-window sweep is the CSI-Bench result.

**Never:**
- No raw or converted data in git. No change to `core/`, `runner/`, `config/` or other adapters.
- No H5 files, no non-HP devices (the 1×4×14 devices are a later adapter), no label mapping to other datasets, no resampling.
- No `room_holdout` plans and no LWM sweep in this story.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Normal file | HP jumping file, csi `(2,2,58,9183)` | csi `(2, 9183, 58)` c64, label `jumping`, room `E01` | N/A |
| Walking case | `act_walking-case3/...` | label `walking` | N/A |
| Out of scope | `act_localization-*`, `device_AmazonPlug/` | skipped | N/A |
| Bad content | no `csi_trace`, real-valued, 1 rx, 56 subcarriers, NaN, zero packet | — | `DataError` with file |
| Missing root / no HP HAR file | — | — | `DataError` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/datasets/padova.py` -- template: `_load` loop with stat index + sha-keyed memmap cache (`_open_cache`, `_replace_atomic`, `_read_index`), `Params`, `VERSION`, sorted output. Copy the cache helpers into the new module: the import-linter `independence` contract over `csi_bench.adapters.*.*` forbids a shared module or import.
- `src/csi_bench/core/contracts.py` -- `Recording`, `RecordingMeta`.
- `tests/contracts/dataset_source.py` -- `check_dataset_source`; the real-data test must call it.
- `tests/test_padova.py` -- test layout: synthetic `.mat` via `scipy.io.savemat` in `tmp_path`, real-data test and committed-plan pin (`TimeBlock().split(recs, ds) == load_split_plan(...)`) skipped when data is absent. `savemat({"csi_trace": {"csi": arr, "mactimer": ...}})` writes a struct that `loadmat` reads back as `mat["csi_trace"][0,0]["csi"]`.
- `src/csi_bench/adapters/splitters/time_block.py` + `src/csi_bench/core/splitplan.py` `write_split_plan` -- generate the plan (defaults 0.6/0.2/0.2, gap 64) from Python; no CLI exists.
- `src/csi_bench/config/presets.py` -- `ant2` selects ids `[0, 1]`, matching `antenna_ids=(0, 1)`.
- `experiments/padova_classical_baseline.yaml` -- header style; `tests/test_sweep.py:416` `test_padova_sweep_expands` -- expansion-test pattern.
- `src/csi_bench/runner/report.py` -- `columns()` already uses sorted row settings for non-`r1/r2` rooms (Story 6.3).
- Extraction: `unzip data/csi_bench/CSI-Bench.zip '<members>' -d data/csi_bench` with the 438 member names (filter as in the scan; ~4.1 GB).

## Tasks & Acceptance

**Execution:**
- [x] `data/csi_bench/Multitask/sub_Human_mat/` -- extract the 438 HP HAR members (not committed).
- [x] `src/csi_bench/adapters/datasets/csi_bench_hp.py` -- `@register("dataset", "csi_bench_hp")` adapter per the rules above.
- [x] `tests/test_csi_bench_hp.py` -- synthetic fixtures for every matrix row; tx0 slicing and axis order against a hand-built array; cache hit and re-convert on change; real-data test (438 recordings, 7 rooms × 5 classes, shapes, contract check) and committed-plan pin.
- [x] `splits/csi_bench_hp/<split_hash>.json` -- generate and commit the `time_block` plan.
- [x] `experiments/csi_bench_hp_classical_baseline.yaml` + expansion test in `tests/test_sweep.py` -- run count, stride 6, dataset/split, room axis, grid equal to `classical_baseline.yaml` apart from rooms.
- [x] (amendment 2026-11-08) `splits/csi_bench_hp/d6b97e4f25b3.json` -- `time_block` plan with gap 610, generated with `TimeBlock(TimeBlock.Params(gap_packets=610)).split(recs, "csi_bench_hp")` + `write_split_plan`; pinned beside the gap-64 plan in `tests/test_csi_bench_hp.py` `test_real_dataset`.
- [x] (amendment 2026-11-08) `experiments/csi_bench_hp_long_windows.yaml` + `test_csi_bench_hp_long_windows_sweep_expands` in `tests/test_sweep.py` -- 576 runs, 144 per (window, stride) pair, plan params (0.6/0.2/0.2, gap 610), window ≤ gap, room axis.

**Acceptance Criteria:**
- Given the extracted data, when `uv run csi-bench describe csi_bench_hp --split a17121b9b216` runs (since the amendment the dataset has two plans, so `--split` is required), then it prints counts per class and room and writes `runs/_dataset/csi_bench_hp.parquet`.
- Given the committed plan, when a run loads it, then every class has train, val and test windows.
- Given the sweep, when one cell runs and then `report`, then a `csi_bench_hp / time_block` section appears.
- Given the long-window sweep, when one cell runs and then `report`, then its runs appear in their own `csi_bench_hp / time_block, gap_packets 610` section.
- Given the finished story, when `git diff --stat` runs, then `src/` changes are only in `adapters/datasets/`, plus the new tests, split and experiment file.

### Review Findings

Code review 2026-11-07 (Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor). 0 decision-needed, 1 patch, 0 defer, 25 rejected.

- [x] [Review][Patch] No test covers re-conversion when the index hits but the array is missing (VERSION bump or deleted `.cache/v1/`). The `if raw is None: raw = path.read_bytes()` re-read is never exercised, so a regression there would make every load fail after the next VERSION bump with all tests green. Add a test: load, monkeypatch `CsiBenchHp.VERSION = 2` (or delete the `.npy`), load again, assert the same sha and CSI and that `v2/<sha>.npy` exists. [tests/test_csi_bench_hp.py; src/csi_bench/adapters/datasets/csi_bench_hp.py:150-153] (verification-gap+blind-hunter)

**Rejected:**
- low: the all-zero check runs per receive chain, which is stricter than "all-zero packet". It is already recorded in Implementation Notes, the real data passes, and a zero chain is unusable anyway. The only other fix is a spec edit.
- false: the finite check covers the unused tx1. The scan of all 438 files found no non-finite values, so it cannot trigger.
- false: "no test for every class having train/val/test windows". `TimeBlock.split` raises `DataError` when a class has no val or test packets (`time_block.py:59`), and the real-data test regenerates the plan.
- reject (spec edit): spec `status: done` and the "review skipped" triage log conflict with this review. This is process metadata, synced in this step.
- false: the plan pin runs only when the data is present. This is the Padova precedent and what the spec prescribes.
- reject (spec decision): cross-device same-session leakage. The Design Notes record and accept it.
- reject (spec decision): `sample_rate_hz=None` is mandated by the spec, and the stride rationale cites the measured `mactimer` rate.
- false: unknown `act_*` folders are skipped silently. The spec says out-of-scope means skipped, and the real-data test pins 438 recordings × 5 classes, so a dropped HAR folder would fail it.
- low: the cache never prunes orphaned `.npy` or index entries. It is a copy of the Padova design, the dataset is static, and the fix adds complexity.
- low: the stat is taken before `read_bytes` (TOCTOU). The files are static and the fix adds complexity.
- low: a read-only root raises `OSError`, not `DataError`. The root is a local extraction and the design is copied from Padova.
- low: a `DataError` mid-loop discards the newly computed index entries. Only the error path pays the cost of re-hashing.
- false: "no class-imbalance guidance". This is not a code defect. The report uses F1 and the imbalance is recorded.
- low: no test for a malformed index entry. The guard is a direct type check and the cost is only a re-hash.
- false: nothing checks `classes` against the `_ACTIVITIES` labels. The real-data test asserts the label set over all 5 classes.
- low: the serial is parsed with `rsplit("_", 3)` in the real-data test. The file names are fixed, and a change would fail the test loudly, not silently.
- false: the hash and 576 are hard-coded in the sweep test. The spec requires the run count to be pinned, and this follows the Padova test pattern.
- false: `splits/csi_bench_hp/` is untracked. It is part of this change and gets committed with it.
- false: complex128 → complex64 overflow. Hardware CSI magnitudes are far below the float32 range.
- low: a same-size rewrite within one mtime tick reuses a stale sha. The data is static and the fix needs extra index fields.
- low (dup): a read-only root gives `OSError`. See above.
- false: a directory matching `*.mat` would crash. The release layout has none, and it would fail loudly.
- false: a MATLAB single-packet trace drops its trailing dimension. Every file has ≥ 6,843 packets.
- false: a cached `.npy` with zero packets would be accepted. Only `_write_cache` writes these files, after `_convert` has rejected zero packets.
- low (dup): the cache grows without bound. See above.

## Implementation Notes

- Extracted the 438 HP HAR members with Python `zipfile` (same filter as the scan) into `data/csi_bench/Multitask/sub_Human_mat/`; first full load (conversion + cache) takes about 1 min.
- Committed plan: `splits/csi_bench_hp/a17121b9b216.json` (time_block 0.6/0.2/0.2, gap 64). Every class and every room has train, val and test packets. Class balance is uneven: walking holds 4.16 M of 8.10 M packets (4 walking cases merged), the other four classes 0.86-1.03 M each.
- The all-zero check is per receive chain and packet (stricter than per packet); the real data passes it.
- `ruff check`/`ruff format --check` fail only on `notebooks/01_classical_overview.ipynb` and `ty check` reports 12 diagnostics in `runner/` and older tests; all pre-existing at the baseline, none in the new files.
- Smoke run: `amplitude_stats | none | knn`, E06, w40 -> `csi_bench_hp / time_block` report section (F1 57.3, dirty-tree run, so `report --include-dirty`).
- Amendment (2026-11-08): plan `splits/csi_bench_hp/d6b97e4f25b3.json` (time_block 0.6/0.2/0.2, gap 610), 438 recordings, shortest 6,843 packets, so the shortest val/test block (≈1,368 packets) still holds windows of 610. With two plans `describe csi_bench_hp` needs `--split`, as for `padova_sharp`; both sweep files name their split. Expected pooled window counts: ≈130 k at w122 down to ≈27 k at w610, against ≈1.34 M at w20/stride 6.
- Speed-up for the 20–50 sweep (commit `d424372`, outside this story's scope rule, results unchanged): `experiments/csi_bench_hp_parallel/` splits it into 8 files with the same 576 run ids so several sweeps run at once; `tune_and_refit(parallel_refit=True)` gives the knn and random_forest refit `n_jobs` = runner workers; SVM kernel cache 1000 MB. Run sweeps with `ulimit -n 4096`: the loader keeps all 438 memmaps open, which exceeds macOS's default 256 open files.

## Spec Change Log

- 2026-11-08 (human renegotiation): added the long-window sweep decision, its gap-610 plan, tasks and acceptance; `describe` now needs `--split`. Trigger: the 20–50 grid is too short for the activities and its SVM cells take hours. The original sweep and its results are kept.
- 2026-11-10: After 6.4 was done, the Mac run added long-window CSI-Bench runs (w122/244/366/610, stride w/2, plan `d6b97e4f25b3`, commit `05237c4`) outside a story; Story 6.6 adopts and commits them. Parallel sweeps on the Mac hit `cache file unreadable` (w20 runs); Story 6.5 fixes it. The short-window classical CSI-Bench runs are incomplete and are not rerun now (sprint-change-proposal-2026-11-10).

## Review Triage Log

- 2026-11-07: review skipped at the user's request (no code review); no findings recorded.
- 2026-11-07: code review run (4 layers). 1 patch applied: `test_reconvert_when_index_hits_but_array_missing` (fails when the re-read is removed); 25 rejected. See Review Findings.

## Design Notes

The three HP serials record the same sessions, so their recordings are near-duplicates in time. `time_block` splits each recording at the same fractions, so a device's test block covers about the same time span as the other devices' test blocks, not their train blocks. Before committing the plan, check that same-session files of the three serials have similar packet counts; if one differs a lot, record it here. They are kept as separate recordings, like the own data's per-file recordings.

Check (2026-11-07, all 438 files): 151 sessions (session = file name without serial): 139 with all three serials, 9 with two, 3 with one. Max/min packet ratio per multi-serial session: median 1.008; 9 sessions above 1.10, 5 above 1.25. Outliers (all with one serial shorter or longer than the other two): `U22/act_walking-case4/env_E27 …2025-6-11_17_13_57` (383: 9,159 vs 430/514: 32,161/32,705, ratio 3.57), `U04/act_walking-case3/env_E03 …2025-4-25_10_20_11` (383: 20,728 vs 10,972/10,414, 1.99), `U22/act_wavinghand/env_E27 …2025-6-11_17_41_50` (383: 15,304 vs 21,960/22,658, 1.48), two more at 1.33 and 1.36. In these sessions the fractional `time_block` boundaries of the odd serial do not cover the same time span as the others', so a little cross-device train/test time overlap is possible there; kept as is (no per-session alignment, `mactimer` unused).

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench describe csi_bench_hp --split a17121b9b216` (and `--split d6b97e4f25b3`) -- expected: 5 classes × 7 rooms, no errors
- one sweep cell + `uv run csi-bench report` -- expected: `csi_bench_hp / time_block` section
