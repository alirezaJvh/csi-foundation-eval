---
title: 'Padova 80 MHz loader'
type: 'feature'
created: '2026-10-28'
status: 'done'
baseline_commit: '8111e225dd6344e8a66d54302993ebe5910ad284'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The study only runs on the own `tor_vergata` data. The Padova 80 MHz dataset (SHARP, Meneghello et al., IEEE TMC 2022) is downloaded in `data/padova/` but has no loader, so it can't be split, described or swept.

**Approach:** Add a `padova` `DatasetSource` adapter that turns each `.mat` trace into one `Recording` with the same 4×234 complex64 layout as `tor_vergata`. Commit its `time_block` split and add a Padova copy of the classical sweep.

**Confirmed format (2026-10-28, from the files and the SHARP paper):**
- There are 105 `.mat` files in `data/padova/<set>/<set>_<activity>[n].mat` across 12 sets (S1a–c, S2a–b, S3a, S4a–b, S5a, S6a–b, S7a). `dataset.zip` is the original archive, already extracted.
- Each file holds one variable, `csi_buff`, a complex128 array of shape (rows, 256). Its rows cycle through the 4 monitor antennas (row i belongs to antenna i mod 4). Rows 4 apart correlate at 0.998, adjacent rows at 0.47. Subcarriers are not FFT-shifted. After `fftshift`, the guard bands (0–5, 251–255) and DC (127–129) are near zero, which matches the 22 null, DC and pilot indices that `csi_proc_python/csi_decoder.py` drops for the own data.
- The packet rate is 173 pkt/s. Traces last about 120 s (about 20.4k packets per antenna). Trials with a number suffix (J1/J2 and so on) are roughly half as long.
- Activity letters: E empty, W walking, R running, J jumping, L sitting, S standing, C sitting down/standing up, H arm gym. C, H and S are missing from S2b, S4b and S6b.
- Environments: S1–S5 were recorded in a bedroom (S4 and S5 with a bookcase blocking the direct path), S6 in a living room and S7 in a university lab. Sets differ in day and person. The meaning of the a/b/c suffix is not documented.

## Boundaries & Constraints

**Always:**
- Each `.mat` file becomes one recording with `recording_id = "padova/<set>/<stem>"`. The label comes from the activity letters, with the trailing digits removed. `source_sha256` is the hash of the `.mat` bytes. `sample_rate_hz=173.0` and `downsample_factor=None`.
- Conversion, in order: drop trailing rows so the row count is a multiple of 4 → split the rows into antennas (`rows[a::4]`) → `fftshift` on the subcarrier axis → delete the same 22 indices as the own data → complex64 with shape `(4, packets, 234)` → read-only, with `antenna_ids=(0,1,2,3)`.
- An all-zero CSI row, an unknown file name, a missing `csi_buff` or a wrong width → `DataError` naming the file. Never drop rows silently, because that would break the antenna cycle.
- The `Params` model forbids extra keys. The root defaults to `data/padova`.
- **Decision (2026-10-28): all 8 activity classes**, in the order `(empty, walking, running, jumping, sitting, standing, sit_stand, arm_gym)` for letters E, W, R, J, L, S, C, H.
- **Decision (2026-10-28): memmap cache.** The first `load()` converts each `.mat` to `data/padova/.cache/<sha256>.npy` (complex64, `(4, packets, 234)`), writing atomically (temp file, then rename). Later loads open it with `np.load(mmap_mode="r")`. The cache is keyed by the source sha, so a changed `.mat` is converted again. It stays at full rate, with no subsampling.
- **Decision (2026-10-28): `room` is the physical place.** `bedroom` for S1–S5, `living_room` for S6, `laboratory` for S7. The set (e.g. `S1a`) stays visible in `recording_id`.
- **Decision (2026-10-28): the Padova sweep uses `stride: 9`**, one window start about every 50 ms at 173 Hz. Stride is not tuned. The yaml header states the rule and notes that `tor_vergata` uses stride 1. The `selection.rooms` axis is `[[bedroom], [living_room], [laboratory]]`. Everything else stays as in `classical_baseline.yaml`.

**Never:**
- No raw or converted data in git. No changes to `core/`, `runner/` or other adapters.
- No label mapping onto the `tor_vergata` classes. No resampling.
- No `room_holdout` plans and no LWM sweep in this story.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Normal trace | `S1a/S1a_C.mat` (81588×256) | csi (4, 20397, 234) c64, label for C, room per environment | N/A |
| Row count not a multiple of 4 | 4k+2 rows | trailing 2 rows dropped, n_packets = k | N/A |
| Repeated trial | `S7a_L2.mat` | same label as L, own recording_id | N/A |
| Bad name | `S1a_X.mat` / `foo.mat` | — | `DataError` |
| Zero row | any all-zero row | — | `DataError` with file name |
| Missing root / no .mat | — | — | `DataError` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/datasets/tor_vergata.py` -- template for the adapter's shape (registration, `Params`, `VERSION`, sorted output, `DataError` messages). Copy its pattern; don't import from it.
- `data/csi_proc_python/csi_decoder.py:42` -- the 22 indices dropped after the shift: `[0-5, 25, 53, 89, 117, 127-129, 139, 167, 203, 231, 251-255]`.
- `src/csi_bench/core/contracts.py` -- `Recording` and `RecordingMeta`.
- `tests/contracts/dataset_source.py` -- `check_dataset_source(source)`. The real-data test must call it.
- `tests/test_tor_vergata.py` -- test layout: synthetic fixtures in `tmp_path` plus a real-data test skipped when the data is absent.
- `src/csi_bench/core/splitplan.py:112` -- `write_split_plan(plan, root)`. Build the plan with `adapters/splitters/time_block.py` (`fractions [0.6,0.2,0.2]`, `gap_packets 64`, as for `tor_vergata`).
- `src/csi_bench/runner/cli.py` -- `describe <dataset>`, used to verify the adapter.
- `experiments/classical_baseline.yaml` -- copy it to the Padova sweep. `paper_beamforming` needs `ant2` (antennas 0 and 1), which works with 4 antennas.
- `.gitignore` -- `data/padova/` is already ignored.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/adapters/datasets/padova.py` -- new `@register("dataset", "padova")` adapter, following the conversion rules plus the answers to the open questions. Uses `scipy.io.loadmat`.
- [x] `tests/test_padova.py` -- synthetic `.mat` fixtures (`scipy.io.savemat`) covering every row of the matrix. A cache test: a second load hits the cache, and a changed source is converted again. A check that antenna splitting, shift and drop match a hand-built array. A real-data test: 105 recordings, the expected rooms and classes, and the contract check.
- [x] `splits/padova/<split_hash>.json` -- generate and commit the `time_block` plan for the real data.
- [x] `experiments/padova_classical_baseline.yaml` -- `classical_baseline.yaml` with `dataset: padova`, the committed split, `stride: 9` and the three-room axis. A header comment gives the run count (216) and the stride rule.

**Acceptance Criteria:**
- Given the real data, when `uv run csi-bench describe padova` runs, then it prints counts per class and room and writes `runs/_dataset/padova.parquet`.
- Given the committed plan, when a Padova run loads it, then every class has train, val and test windows (no `DataError` from the splitter).
- Given `experiments/padova_classical_baseline.yaml`, when one cell is run and then `report`, then a `padova / time_block` section appears.
- Given the finished story, when `git diff --stat main` runs, then `src/` changes are only in `adapters/datasets/`, plus the new test, the split and the experiment file.

### Review Findings

- [x] [Review][Patch] A cache hit still reads and hashes every `.mat` (5.7 GB, 7.0 s per `load()`, about 25 min over the 216-run sweep). Decision (2026-10-28, Jj): keep a stat index `(path, size, mtime_ns) → sha` in `.cache/`, and re-hash a file only when its size or mtime changes. The cache stays keyed by the source sha [src/csi_bench/adapters/datasets/padova.py:131]
- [x] [Review][Patch] The cache key ignores `Padova.VERSION`, so a changed conversion (`_DROPPED`, shift, antenna split) with a bumped VERSION still serves old arrays under a new run_id [src/csi_bench/adapters/datasets/padova.py:135]
- [x] [Review][Patch] No test pins the committed split `splits/padova/b6010d87ce4d.json` to the loader output. Add a skip-if-absent `TimeBlock().split(...) == load_split_plan(...)` test, mirroring `test_real_data_plan_and_committed_file` [tests/test_padova.py]
- [x] [Review][Patch] No expansion test for `experiments/padova_classical_baseline.yaml` (216 ids, stride 9, dataset/split, three room sets, grid equals `classical_baseline.yaml` apart from rooms) [tests/test_sweep.py]
- [x] [Review][Patch] `test_bad_name["S9a_E"]` writes into `S1a/`, so it hits the folder-mismatch branch and never the "unknown recording set" guard. Write it to `S9a/` and match that message [tests/test_padova.py:88]
- [x] [Review][Patch] Untested error branches: file in the wrong set folder (`S1a/S2a_E.mat`), real-valued `csi_buff`, and fewer than 4 rows [tests/test_padova.py]
- [x] [Review][Patch] The real-data test checks rooms and labels separately, never 8 classes in each room, which the per-room sweep needs. The data does satisfy it today [tests/test_padova.py:31]
- [x] [Review][Patch] The sweep yaml keeps the "Original header follows" block, which repeats the run count and describes the tor_vergata room axis. Merge it into one Padova header [experiments/padova_classical_baseline.yaml:1]
- [x] [Review][Defer] AC2/AC3 not shown by a real run [experiments/padova_classical_baseline.yaml] — deferred: maybe-false, would be medium. No Padova run exists under `runs/`. Settle by running one cell of the sweep, then `report`, and checking for a `padova / time_block` section.

Rejected:
- `false`: a stray non-matching `.mat` aborts the load. The spec requires `DataError` on an unknown file name.
- `low`: deeper nesting or `.MAT` files are silently skipped. The dataset layout is fixed, and the fix adds a guard.
- `low`: `_open_cache` does not check width or packet count. Cache files are sha-keyed and only this code writes them. Moot after the VERSION patch.
- `low`: a zip-magic cache file makes `np.load` return an `NpzFile`, which raises `AttributeError`. Not reachable in practice.
- `false`: a read-only root raises `OSError`. That loud failure is correct.
- `low`: orphan `.tmp` files, never-removed stale cache entries, and the cache doubling disk use. The cache design is a spec decision, and the fixes add cleanup logic.
- `false`: NaN or inf in `csi_buff`. None of the 105 real recordings has non-finite values.
- `false`: a row dropped mid-file misaligns the antennas. In all 840 sampled segments of the real data, within-antenna lag-1 correlation exceeds cross-antenna correlation. The absolute antenna identity cannot be checked from the data.
- Rejected because the fix edits this spec: the Design Notes data scan (zero rows, row counts not a multiple of 4) is not recorded in the spec. Flagged to the user.
- `false`: the zero-row check runs after the tail is dropped. The spec puts truncation first, and the dropped tail is specified behavior.
- `low`: the real-data test writes about 15 GB of cache into `data/padova`. The spec requires both the real-data test and that cache location.
- `low`: the room check slices `recording_id[7:9]`. Fragile, but correct for this layout.
- `low`: no assert on `len(_DROPPED) == 22`, and the test copies the list. The test list is an intentional independent oracle.
- `false`: the parent-folder check is stricter than the spec. Harmless, and consistent with the `recording_id` rule.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Design Notes

Matching the own data's 234-subcarrier layout (shifted, with nulls, DC and pilots removed) keeps every extractor, and the `phase_sanitize` null-bin caveat, unchanged across the two datasets. SHARP keeps 245 bins; the difference is the 8 pilots plus a slightly wider guard band. Before writing the adapter, scan all 105 files for all-zero rows and for row counts that aren't multiples of 4, and record the findings here.

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench describe padova` -- expected: 8 classes × 3 rooms, no errors
- A second `load()` -- expected: no `.mat` reads (cache hit), with arrays as read-only memmaps
