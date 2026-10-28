# Epic 6 Context: Public datasets (activity recognition)

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher rerun the whole study (classical baselines, LWM and later the other pretrained models, under within-room and cross-room protocols, with confidence intervals) on the public activity-recognition datasets the supervisor verified on 2026-10-08: Padova 80 MHz first, then CSI-Bench. Each dataset is added by writing only a new `DatasetSource` loader, its committed split and experiment configs, with no change to the evaluation code. This proves the harness is extensible and tests whether the findings hold on independent data. Padova comes first because it uses the same hardware family as the own `tor_vergata` data, so it is likely the easiest import, and its several environments probably give a second cross-environment generalisation test. The exact formats of both datasets are not yet verified, so each story starts by confirming them.

## Stories

- Story 6.1: Padova 80 MHz loader
- Story 6.2: CSI-Bench loader

## Requirements & Constraints

- **Confirm before coding.** Each story first confirms the dataset's format, class list, environments and recording layout, and records them in the story spec before writing any code.
- **Loader only.** Success means: switching only the `dataset:` field of the classical sweep runs it on the new dataset, `report` shows a section for that dataset, and `git diff` shows no change outside `adapters/datasets/`, `splits/` and `experiments/`.
- **Contract fit.** A new adapter declares its own ordered class list, passes the `DatasetSource` contract test, gets its own committed `time_block` split, and works with `describe`.
- **Environments map to `room`**, so `room_holdout` gives a cross-environment test with no extra code.
- **Stop rule (CSI-Bench).** If the data is amplitude-only or otherwise cannot fill the complex-CSI `Recording` contract, the story stops and asks for a spine amendment. Do not work around the contract. A blocked 6.2 may be deferred without holding up Epic 7.
- **No label mapping across datasets** in this phase. Zero-shot cross-dataset transfer and people counting (WiMANS, EHUNAM) are out of scope.
- **Rate.** Recordings are used at their saved rate, with no resampling in later stages. Any resampling or cleaning is a `DatasetSource` option (e.g. `equalize_rate`).
- **No leakage.** Same splitters, windows, metrics and test-once rule as every other dataset. Results are reported per `(dataset, splitter)` and never mixed in one table.
- **Data location.** Padova data lives in `data/padova/`, which is git-ignored. Do not commit raw data.
- **Sources.** Padova: IEEE DataPort "CSI dataset for wireless human sensing on 80 MHz Wi-Fi channels", paper arXiv 2305.03170, code github.com/francescamen/SHARP (Asus RT-AC86U, Nexmon, 802.11ac 80 MHz). CSI-Bench: NeurIPS 2025 Datasets & Benchmarks, large and multi-device, possibly amplitude-only.

## Technical Decisions

- **Ports are fixed.** A new dataset is a new adapter behind the existing `DatasetSource` port, in `adapters/datasets/<name>`. It registers with `@register("dataset", "<name>")`, declares a pydantic `Params` and an integer `VERSION`, and imports only `core`, never another adapter. Discovery is automatic.
- **`Recording` contract.** `csi` is complex64 with axes `(antennas, packets, subcarriers)`. `antenna_ids` holds the original receiver antenna index of each row. `recording_id` is `"<dataset>/<path relative to dataset root, without extension>"`. Each recording has `room` and `label` (an index into the adapter's class list). `meta: RecordingMeta` (`n_packets`, `downsample_factor | None`, `sample_rate_hz | None`, `source_sha256`) is filled only by the `DatasetSource`. Packet indices are 0-based and end-exclusive. No later stage reorders, drops or resamples packets.
- **Splits.** `time_block` assigns contiguous packet blocks per recording to train/val/test, separated by `gap_packets` (initially 64). Plans are committed at `splits/<dataset>/<split_hash>.json` with each input's `source_sha256`. The splitter raises `DataError` if any class has no val or test windows, so short or single-recording classes need checking. `room_holdout` needs explicit `train_rooms`/`test_rooms`, one committed plan per direction, and validation only from train rooms.
- **Selections.** Room settings are `selection: {rooms: [...]}` filters applied after splitting. With `room_holdout`, `selection.rooms` must equal `train_rooms ∪ test_rooms`.
- **Report.** One section per `(dataset, splitter)`. `time_block` sections have room columns and `room_holdout` sections have direction columns. A column must come from a single `split_hash`. `describe <dataset>` writes counts to `runs/_dataset/<dataset>.parquet`, which serves as the rate-confound guard.
- **Extractor compatibility.** Extractors declare `requires: {phase, n_antennas}`, which the runner checks at config validation. `paper_beamforming` and the 2-antenna variants assume antennas `[0, 1]`.
- **Known gaps to check on new data:**
  - LWM declares `n_antennas: "any"` but fails at extraction for more than 4 antennas. A dataset with more antennas passes validation and then fails.
  - `phase_sanitize` may bias its slope fit on zero-magnitude (null/DC) subcarriers. Scan new datasets for null bins before reusing it.

## Cross-Story Dependencies

- Both stories depend on Epics 1, 2 and 4: the `DatasetSource` contract test, `describe`, `time_block` and `room_holdout` splitters, the classical sweep, per-`(dataset, splitter)` report sections and bootstrap intervals. Epic 5 LWM configs are reused with only a `dataset:` change.
- 6.2 follows 6.1 but has no code dependency on it.
- Downstream: Story 7.4 runs the full comparison under `time_block` on every verified public dataset, so Epic 6 comes before Epic 7.
