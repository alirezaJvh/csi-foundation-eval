# Padova cross-room baseline (Story 6.2)

Date: 2026-11-07. Sweep: `experiments/padova_cross_room.yaml`, 144 of 144 runs `done`, none dirty, all at commit `0d39871`.

## Question

Does a classical model trained in one Padova room still work in another room? Until now, every Padova score was same-room: same recording, later in time (`time_block`).

## Protocol

- Splitter `room_holdout`, two committed plans:
  - `725a12be6b83`: train `bedroom`, test `living_room`.
  - `99d235eb2f0e`: train `bedroom`, test `laboratory`.
- Plan params: `fractions [0.75, 0.25]`, `gap_packets 64`. Train and val come from the bedroom only. The test room is never used to fit or to select.
- Grid: the same as `padova_classical_baseline.yaml`, without its room axis.
  - Windows 20, 30, 40 and 50 packets, stride 9 (about 52 ms at about 173 Hz).
  - Feature sets: `amplitude_stats`, `paper_beamforming`, and the two together.
  - Feature selection: none, or mRMR with `k` 2 or 3.
  - Heads: `svm`, `knn`, `random_forest`.
  - Antennas `ant2`, seed 0.
  - 72 runs per direction, 144 in total.
- Selection: each cell shows the test score of the variant with the best tuning-stage val macro-F1. No oracle selection is used.

**Not SHARP's protocol.** This protocol trains on the whole bedroom (S1–S5) with all 8 classes. SHARP trains on S1 only, with 5 classes. These numbers are therefore not directly comparable to SHARP's S6 and S7 results. Story 6.3 runs the closer comparison.

## Results

Test macro-F1 in %, with the 95% interval from a recording-level bootstrap (1,000 resamples). Chance is 12.5% (8 classes). Same-room columns use `time_block` plan `b6010d87ce4d` (fractions 0.6/0.2/0.2) and are picked with the same rule (best tuning val per method and room). `report` shows those cells as `-` because of a known deferred bug, so they were computed from `runs/index.parquet`.

| Method | Same-room bedroom | Same-room laboratory | Same-room living_room | bedroom→laboratory | bedroom→living_room |
|---|---|---|---|---|---|
| `amplitude_stats` / none / `knn` | 35.89 [32.52, 38.94] | 54.82 [51.72, 57.73] | 45.15 [40.25, 49.81] | 18.88 [18.38, 19.39] | 7.87 [7.02, 8.82] |
| `amplitude_stats` / none / `random_forest` | 35.61 [32.40, 38.42] | 53.39 [50.55, 56.11] | 45.75 [40.54, 50.70] | 15.13 [14.60, 15.66] | 8.02 [7.27, 8.83] |
| `amplitude_stats` / none / `svm` | 31.34 [25.54, 36.63] | 53.49 [47.40, 58.53] | 44.09 [37.72, 50.32] | 12.89 [12.66, 13.14] | 11.27 [10.56, 12.10] |
| `paper_beamforming` / none / `knn` | 44.16 [39.93, 47.94] | 65.20 [62.30, 67.98] | 57.06 [54.17, 59.93] | 13.57 [12.78, 14.11] | 14.79 [14.11, 15.68] |
| `paper_beamforming` / none / `random_forest` | 52.50 [48.43, 55.96] | 70.98 [69.40, 72.62] | 61.71 [57.49, 65.82] | 15.14 [14.01, 16.24] | 18.62 [16.72, 20.48] |
| `paper_beamforming` / none / `svm` | 45.58 [40.38, 49.95] | 67.49 [65.76, 69.59] | 59.88 [56.71, 63.30] | 18.24 [17.57, 18.90] | 14.32 [13.75, 15.04] |
| `paper_beamforming` / mRMR / `knn` | 32.03 [29.39, 34.29] | 54.01 [50.33, 57.39] | 50.25 [48.68, 52.09] | 16.63 [15.90, 17.35] | 18.04 [16.57, 19.30] |
| `paper_beamforming` / mRMR / `random_forest` | 37.01 [33.29, 40.13] | 59.37 [56.56, 62.49] | 52.13 [49.80, 54.70] | **19.36 [17.74, 20.90]** | **21.04 [19.63, 22.51]** |
| `paper_beamforming` / mRMR / `svm` | 28.59 [24.66, 32.22] | 54.65 [50.32, 58.61] | 53.83 [50.90, 57.01] | 18.09 [17.13, 19.03] | 19.48 [18.57, 20.34] |
| both / none / `knn` | 55.43 [51.08, 58.96] | 72.83 [69.77, 75.99] | 57.70 [53.39, 61.81] | 10.13 [9.43, 10.70] | 10.14 [9.07, 11.23] |
| both / none / `random_forest` | **61.87 [57.94, 65.32]** | **78.16 [76.15, 79.42]** | **66.15 [62.71, 69.53]** | 12.71 [11.81, 13.25] | 11.80 [11.30, 12.34] |
| both / none / `svm` | 58.25 [53.63, 62.05] | 76.69 [74.89, 78.50] | 64.57 [60.55, 68.55] | 11.09 [9.14, 12.82] | 10.55 [9.90, 11.18] |

Rows the table leaves out are identical to rows shown, for the reasons below:

- `amplitude_stats` with mRMR equals `amplitude_stats` with no selection in the cross-room columns. The set has only 3 features, and `k=3` keeps them all.
- Cross-room, "both / mRMR" equals "`paper_beamforming` / mRMR". mRMR keeps only `paper_beamforming` features from the combined set: `distance_mean` and `distance_theta_var` in every selection, plus `flatness_theta_var` or `flatness_mean`.

## Findings

1. **Cross-room transfer mostly fails.** The best cross-room score is 21.04 (bedroom→living_room) and 19.36 (bedroom→laboratory), against 61.87 same-room in the bedroom and 66.15 and 78.16 same-room in the test rooms. Every cross-room cell lies between 7.87 and 21.04. Several cells are at or below chance.
2. **The best same-room model transfers worst.** Both feature sets, with no selection and `random_forest`, lead every same-room column. Cross-room, the same method scores 12.71 and 11.80, close to chance. Its w50 variant had the highest tuning val score of all 144 runs (61.0), so selecting on bedroom val does not predict transfer.
3. **A few `paper_beamforming` features transfer best.** mRMR down to 2 or 3 `paper_beamforming` features gives the best cross-room scores in both directions, while in the bedroom same-room it costs 15–25 points against no selection.
4. **`amplitude_stats` does not transfer to the living room** (7.87–11.27), and only barely to the laboratory (12.89–18.88).

## Caveats

- **Not a pure room effect.** Padova sets differ in day and person, so each direction changes room, person and day together. S4 and S5 also train with a bookcase blocking the direct path. The drop is an upper bound on the room effect alone.
- **Thin test sets.** Every cross-room cell carries `*` in `report`. In the laboratory, 6 of 8 classes (empty, walking, running, standing, sit_stand, arm_gym) have a single test recording; in the living room, 3 of 8 (standing, sit_stand, arm_gym). Per-class results and intervals rest on those single recordings.
- **Columns are not independent.** Both plans have identical train and val blocks, so the two directions share one bedroom training set and the same chosen variants.
- **Training volume differs from same-room.** The same-room baseline trains on fractions 0.6/0.2/0.2 of each room, while cross-room trains on 0.75/0.25 of the bedroom.
- **Single seed.** All runs use seed 0.

## Run time

About 32 h wall time, from 2026-11-05 18:11 to 2026-11-07 04:11 UTC, with `--workers 4` on a 16-core, 11 GiB machine. The first attempt was killed by the out-of-memory killer during the fifth run (`random_forest`), after 4 runs had finished. After 8 GiB of swap was added, the sweep resumed, skipped those 4 runs and finished the rest. RBF `svm` takes about 95% of the time: 15–23 min per run without mRMR and 40–50 min with it. One run took 326 min against 52 min for its twin, most likely during heavy swapping.

## Reproduce

```
uv run csi-bench sweep experiments/padova_cross_room.yaml --workers 4
uv run csi-bench report
```
