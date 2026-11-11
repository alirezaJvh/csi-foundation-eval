# Padova long-window classical baseline (Story 6.6)

Date: 2026-11-11 · Config: `experiments/padova_long_windows.yaml` · Plan: `5cfd50d7d8ae` (`time_block`, 0.6/0.2/0.2, gap 865) · Full section: `padova-long-windows-6-6.txt`

## Runs

| Item | Runs | Commit | Device | Wall clock (UTC) | Sum of run durations |
|---|---|---|---|---|---|
| `padova`, gap 865, w173/346/519/865, stride w/2 | 216/216 `done`, 0 dirty | `56798ca` | cpu | 2026-11-11 20:20–20:39 | 18 min |

The CSI-Bench long-window runs (576, plan `d6b97e4f25b3`, made at `05237c4`) are adopted unchanged: `csi_bench_hp_long_windows.yaml` expands to their run ids, and `report` shows them under `csi_bench_hp / time_block, gap_packets 610`.

## Results

Each cell is in-room (train, val and test from the same room, split by time). Selection is by tuning-stage val F1, as in `report`. Test macro-F1 / accuracy in %, 95% recording-bootstrap CI.

Best per room (selected by val):

| Room | Method | Window | Test F1 [CI] | Test acc |
|---|---|---|---|---|
| bedroom (S1–S5) | amplitude_stats+paper_beamforming, none, svm | 865 | 75.60 [69.41, 80.43] | 76.61 |
| laboratory (S7) * | amplitude_stats+paper_beamforming, none, random_forest | 865 | 79.53 [68.55, 82.62] | 79.59 |
| living_room (S6) * | amplitude_stats+paper_beamforming, none, random_forest | 865 | 77.23 [67.46, 84.56] | 79.78 |

\* Several classes have a single test recording in this room, so the CI is wide.

Highest test F1 among the per-method cells: amplitude_stats+paper_beamforming, none, svm in laboratory, w865, 83.44 [76.67, 90.97].

- Longer windows win: across the 54 (method, room) cells, the val-selected window is 865 in 32, 519 in 18, 346 in 2 and 173 in 2.
- `paper_beamforming` carries the score; `amplitude_stats` alone stays at 36–70 F1.
- Bedroom, which has the most recordings, rises from 61.87 F1 at the short windows (Stories 6.2/6.3) to 75.60.

## Not covered

No cross-room (`room_holdout`) or pooled-rooms cell at long windows: the room-holdout plans have gap 64 < window, and the grid lists the three single rooms only.
