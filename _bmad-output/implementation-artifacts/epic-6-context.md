# Epic 6 Context: Public datasets (activity recognition)

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher rerun the whole study (classical baselines, LWM and later the other pretrained models, under within-room and cross-room protocols, with confidence intervals) on the public activity-recognition datasets the supervisor verified on 2026-10-08: Padova 80 MHz first, then CSI-Bench. Each dataset is added by writing only a new `DatasetSource` loader, its committed split and experiment configs, with no change to the evaluation code. This proves the harness is extensible and tests whether the findings hold on independent data. Padova comes first because it uses the same hardware family as the own `tor_vergata` data, so it is likely the easiest import, and its several environments probably give a second cross-environment generalisation test. The exact formats of both datasets are not yet verified, so each story starts by confirming them. After the loaders, long-window runs (1, 2, 3, 5 s) and frozen LWM cover both public datasets, with a runner device option so LWM embedding can use a GPU (MPS or CUDA).

## Stories

- Story 6.1: Padova 80 MHz loader
- Story 6.2: Padova cross-room baseline
- Story 6.3: Padova SHARP-comparable run
- Story 6.4: CSI-Bench loader
- Story 6.5: Device option and parallel-safe dataset cache
- Story 6.6: Padova long-window plan and classical baseline
- Story 6.7: LWM on Padova and CSI-Bench

## Requirements & Constraints

- **Confirm before coding.** Each story first confirms the dataset's format, class list, environments and recording layout, and records them in the story spec before writing any code.
- **Loader only.** Success means: switching only the `dataset:` field of the classical sweep runs it on the new dataset, `report` shows a section for that dataset, and `git diff` shows no change outside `adapters/datasets/`, `splits/` and `experiments/`.
- **Contract fit.** A new adapter declares its own ordered class list, passes the `DatasetSource` contract test, gets its own committed `time_block` split, and works with `describe`.
- **Environments map to `room`**, so `room_holdout` gives a cross-environment test with no extra code.
- **Stop rule (CSI-Bench).** If the data is amplitude-only or otherwise cannot fill the complex-CSI `Recording` contract, the story stops and asks for a spine amendment. Do not work around the contract. A blocked 6.4 may be deferred without holding up Epic 7.
- **No label mapping across datasets** in this phase. Zero-shot cross-dataset transfer and people counting (WiMANS, EHUNAM) are out of scope.
- **Rate.** Recordings are used at their saved rate, with no resampling in later stages. Any resampling or cleaning is a `DatasetSource` option (e.g. `equalize_rate`).
- **No leakage.** Same splitters, windows, metrics and test-once rule as every other dataset. Results are reported per `(dataset, splitter)` and never mixed in one table.
- **Data location.** Padova data lives in `data/padova/`, which is git-ignored. Do not commit raw data.
- **Sources.** Padova: SHARP paper (arXiv 2103.09924), code github.com/francescamen/SHARP (Asus RT-AC86U, Nexmon, 802.11ac 80 MHz). Source of the S1–S7 files in `data/padova/`: SHARP (Meneghello et al., IEEE TMC 2022, doi 10.1109/TMC.2022.3185681, arXiv 2103.09924), based on the file layout matching its Table 1; the download source is unconfirmed. arXiv 2305.03170 describes a related, larger dataset that is not the one loaded. CSI-Bench: NeurIPS 2025 Datasets & Benchmarks, large and multi-device. Its H5 files are amplitude-only; `Multitask/sub_Human_mat` holds complex CSI.

## Technical Decisions

- **Ports are fixed.** A new dataset is a new adapter behind the existing `DatasetSource` port, in `adapters/datasets/<name>`. It registers with `@register("dataset", "<name>")`, declares a pydantic `Params` and an integer `VERSION`, and imports only `core`, never another adapter. Discovery is automatic.
- **`Recording` contract.** `csi` is complex64 with axes `(antennas, packets, subcarriers)`. `antenna_ids` holds the original receiver antenna index of each row. `recording_id` is `"<dataset>/<path relative to dataset root, without extension>"`. Each recording has `room` and `label` (an index into the adapter's class list). `meta: RecordingMeta` (`n_packets`, `downsample_factor | None`, `sample_rate_hz | None`, `source_sha256`) is filled only by the `DatasetSource`. Packet indices are 0-based and end-exclusive. No later stage reorders, drops or resamples packets.
- **Splits.** `time_block` assigns contiguous packet blocks per recording to train/val/test, separated by `gap_packets` (initially 64). Plans are committed at `splits/<dataset>/<split_hash>.json` with each input's `source_sha256`. The splitter raises `DataError` if any class has no val or test windows, so short or single-recording classes need checking. `room_holdout` needs explicit `train_rooms`/`test_rooms`, one committed plan per direction, and validation only from train rooms.
- **Selections.** Room settings are `selection: {rooms: [...]}` filters applied after splitting. With `room_holdout`, `selection.rooms` must equal `train_rooms ∪ test_rooms`.
- **Report.** One section per `(dataset, splitter)`. `time_block` sections have room columns and `room_holdout` sections have direction columns. A column must come from a single `split_hash`. `describe <dataset>` writes counts to `runs/_dataset/<dataset>.parquet`, which serves as the rate-confound guard.
- **Device (AD-12 amendment).** The runner sets the device (default `cpu`; `--device` or `CSI_BENCH_DEVICE` selects `mps`/`cuda`), torch thread count and determinism flags, and records them in provenance; adapters never set them. Device is `affects_result=False` (never changes `run_id`). Same config + same device reproduces metrics; across devices only `run_id` is guaranteed equal. Per-packet embedding caches include the device in their key. Only torch extractors use the device; sklearn heads, poolers and classical extractors stay on CPU.
- **Torch wheels.** CPU index by default via `tool.uv.sources`; macOS wheels include MPS; an optional `cuda` uv extra selects the CUDA index on Linux (exact uv mechanism decided in Story 6.5).
- **Dataset caches must be parallel-safe.** Several sweep processes may load the same dataset at once; parallel CSI-Bench sweeps hit `cache file unreadable`. The cause must be reproduced by a test before fixing.
- **Extractor compatibility.** Extractors declare `requires: {phase, n_antennas}`, which the runner checks at config validation. `paper_beamforming` and the 2-antenna variants assume antennas `[0, 1]`.
- **Known gaps to check on new data:**
  - LWM declares `n_antennas: "any"` but fails at extraction for more than 4 antennas. A dataset with more antennas passes validation and then fails.
  - `phase_sanitize` may bias its slope fit on zero-magnitude (null/DC) subcarriers. Scan new datasets for null bins before reusing it.

## Cross-Story Dependencies

- All stories depend on Epics 1, 2 and 4: the `DatasetSource` contract test, `describe`, `time_block` and `room_holdout` splitters, the classical sweep, per-`(dataset, splitter)` report sections and bootstrap intervals. Epic 5 LWM configs are reused with only a `dataset:` change.
- 6.4 (CSI-Bench) follows 6.1 but has no code dependency on it.
- Order: 6.5 → 6.6 → 6.7 → Epic 7. 6.7's embedding cost depends on 6.5 (device, cache fix); 6.6 commits the long-window plans 6.7 uses. CSI-Bench HP has 2 receive chains, so `ant4` is invalid there.
- Downstream: Story 7.4 runs the full comparison under `time_block` on every verified public dataset, so Epic 6 comes before Epic 7.
