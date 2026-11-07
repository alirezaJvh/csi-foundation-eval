# Padova SHARP-comparable run (Story 6.3)

Date: 2026-11-07. Sweep: `experiments/padova_sharp.yaml`, 216 of 216 runs `done`, none dirty. Full `report` output: `padova-sharp-6-3.txt`.

## Question

Padova same-room bedroom macro-F1 was 55–62 for the best classical models (8 classes, sets S1–S5, windows of 20–50 packets). SHARP reports 99.59 % accuracy on S1. Which protocol differences explain the gap, and how much of it does each one close?

## SHARP's S1 number

From SHARP (Meneghello et al., IEEE TMC 2022, arXiv 2103.09924), Table 3, set S1: **accuracy 99.59 %, F1-score 0.996** (mean over 5 classes). Their protocol, quoted from the paper:

- Five classes: empty, sitting, walking, running, jumping.
- Set S1 only. "60% of the data in S1 makes up the training set, while the remaining 40% is evenly split between the validation and the test sets". The paper does not say whether this split is by time or random.
- Input: Doppler traces of `Nw = 340` Doppler vectors, "roughly 340·Tc = 2 s of measurements". Each Doppler vector comes from `N = 31` sanitized CFR samples over a sliding window, at 173 packets/s.
- Model: a CNN per antenna, with decision fusion across the 4 antennas.

## Protocol

- Each step changes one thing from the step before it:
  0. `padova`, bedroom: sets S1–S5, 8 classes, windows 20–50, plan `b6010d87ce4d` (gap 64). Data from the Story 6.1 baseline.
  1. `padova_s1`: S1 only, still 8 classes. Plan `4359a19e3c7d` (gap 64), windows 20–50.
  2. `padova_sharp`: S1 with SHARP's 5 classes. Plan `62b4f776f13e` (gap 64), windows 20–50.
  3. `padova_sharp`, window 50 on the gap-346 plan `7640fb8ad320`. This is the control: same window, larger gap, slightly smaller blocks.
  4. `padova_sharp`, window 346 (about 2 s, close to SHARP's 340) on the gap-346 plan.
- Every plan is `time_block` with fractions 0.6/0.2/0.2, the same as SHARP's 60/20/20. Our blocks are split in time, with a gap between them.
- Grid at each step: features `amplitude_stats`, `paper_beamforming` or both; selection none or mRMR k 2/3; heads `svm`, `knn`, `random_forest`; stride 9; seed 0.
- Antennas `ant2` for the chain. `ant4` (all 4 antennas) only with `amplitude_stats`, reported separately below.
- Selection: each cell is the test score of the variant with the best tuning-stage val macro-F1. The window is part of the variant at steps 0–2. Steps 3 and 4 are read per window from `runs/index.parquet`. `report` merges them into one `gap_packets 346` section, where w346 wins every cell.
- Runs and commits: steps 1–2 at `737986f` and `8962d69`; steps 3–4 at `738f1b9`; step 0 at `2335ef9`.

## Results: the chain

The best method at every step, by val, is both feature sets / no selection / `random_forest`. Test scores in %, with the 95 % recording-level bootstrap interval.

| Step | Change | Macro-F1 | Accuracy | Accuracy gap to SHARP (99.59) |
|---|---|---|---|---|
| 0 | `padova` bedroom, S1–S5, 8 classes, w ≤ 50 | 61.87 [57.94, 65.32] | 63.40 [59.21, 67.53] | 36.19 |
| 1 | S1 only | 69.84 [64.53, 74.73] | 69.85 [64.79, 74.80] | 29.74 |
| 2 | 5 SHARP classes | 81.61 [78.32, 84.60] | 81.55 [78.28, 84.52] | 18.04 |
| 3 | gap 346, w50 | 80.82 [77.21, 84.04] | 80.79 [77.16, 84.07] | 18.80 |
| 4 | window 346 | 89.40 [83.94, 93.70] | 89.50 [84.43, 93.78] | 10.09 |

How much of the 36.19-point accuracy gap each step closes:

| Change | Points closed | Share of gap |
|---|---|---|
| Data subset (S1 only) | 6.45 | 18 % |
| Class set (8 → 5) | 11.70 | 32 % |
| Gap 64 → 346 at w50 | −0.76 | 0 % (within noise) |
| Window 50 → 346 | 8.71 | 24 % |
| **Remaining** | **10.09** | **28 %** |

Best test cell overall: both / none / `svm` at w346, 89.84 F1 [85.79, 93.41], accuracy 89.97. Val did not pick it.

### Per method, macro-F1 (ant2)

| Method | 0 padova | 1 s1 | 2 sharp w≤50 | 3 g346 w50 | 4 g346 w346 |
|---|---|---|---|---|---|
| `amplitude_stats` / none / `knn` | 35.89 | 44.00 | 62.49 | 63.00 | 68.60 |
| `amplitude_stats` / none / `random_forest` | 35.61 | 43.90 | 60.69 | 61.55 | 62.75 |
| `amplitude_stats` / none / `svm` | 31.34 | 46.35 | 61.42 | 61.49 | 68.39 |
| `paper_beamforming` / none / `knn` | 44.16 | 53.17 | 66.62 | 66.49 | 81.58 |
| `paper_beamforming` / none / `random_forest` | 52.50 | 58.93 | 73.65 | 73.48 | 85.40 |
| `paper_beamforming` / none / `svm` | 45.58 | 55.65 | 68.84 | 69.03 | 87.44 |
| both / none / `knn` | 55.43 | 65.88 | 77.00 | 77.85 | 85.37 |
| both / none / `random_forest` | **61.87** | **69.84** | **81.61** | **80.82** | 89.40 |
| both / none / `svm` | 58.25 | 68.74 | 80.66 | 79.23 | **89.84** |
| `paper_beamforming` / mRMR / `random_forest` | 37.01 | 42.81 | 59.65 | 59.36 | 73.77 |
| both / mRMR / `random_forest` | 37.01 | 42.61 | 61.72 | 53.35 | 67.73 |

`amplitude_stats` with mRMR equals no selection (3 features, `k = 3` keeps them all). Every other mRMR row is 10–25 points below its no-selection row; see `padova-sharp-6-3.txt`.

### `ant4` (all four antennas, `amplitude_stats` only)

| Head | ant2 w≤50 | ant4 w≤50 | ant2 g346 w50 | ant4 g346 w50 | ant2 w346 | ant4 w346 |
|---|---|---|---|---|---|---|
| `knn` | 62.49 | 59.85 | 63.00 | 58.22 | 68.60 | 63.39 |
| `random_forest` | 60.69 | 58.31 | 61.55 | 60.29 | 62.75 | 68.01 |
| `svm` | 61.42 | 61.78 | 61.49 | 62.58 | 68.39 | 69.71 |

Macro-F1, no selection. All four antennas do not help `amplitude_stats`: every difference lies inside the other's interval (for example, w346 `svm` ant4 69.71 [62.90, 75.16] against ant2 68.39 [54.28, 78.14]).

## Findings

1. **Protocol alignment closes about 72 % of the gap**, from 36.19 to 10.09 accuracy points. The best aligned model reaches 89.50 % accuracy and 89.40 macro-F1, against SHARP's 99.59 % and 0.996.
2. **The class set is the largest single step (+11.70).** Part of this is mechanical: chance rises from 12.5 % to 20 %, and the three classes dropped (standing, sit_stand, arm_gym) are the static or slow ones that are hardest to tell apart.
3. **Window length is the second-largest step (+8.71), and it is the window, not the gap.** At w50, the gap-346 plan scores the same as the gap-64 plan (80.82 vs 81.61, intervals overlapping). With 20–50 windows, val picks w50 in almost every cell, so longer windows help across the whole range. `paper_beamforming` gains most from the long window (+12 to +18 F1); `amplitude_stats` gains least (+1 to +7).
4. **The data subset helps (+6.45).** Training and testing on S1 alone removes the variation between S1–S5, where S4–S5 also move the monitor and block the direct path.
5. **mRMR down to 2–3 features always hurts same-room**, by 10–25 points, as in Stories 6.1 and 6.2.

## Is a Doppler feature family worth proposing?

**Yes, as its own story.** Three reasons:

- **Temporal features help most.** After the protocol is aligned, the strongest lever is the window, and the feature family that gains most from it is `paper_beamforming`, which carries phase and velocity-like information. At w346, mRMR picks `paper_beamforming.doppler_theta_var` in half of its selections; in no Padova section with w ≤ 50 does it appear.
- **SHARP's input is a Doppler spectrogram**, built from phase-sanitized CFR (STFT over 31 samples, 100 velocity bins, 340 vectors). None of our families computes it.
- **The remaining 10 points can't be closed by protocol changes.** Every protocol difference we can match is now matched, except the split method (below) and SHARP's 4-antenna fusion. On `amplitude_stats`, `ant4` gave nothing.

Scope the proposal carefully. SHARP pairs Doppler traces with a CNN and 4-antenna decision fusion, so a Doppler family with classical heads tests the feature, not SHARP's whole pipeline. It also needs windows of at least about 340 packets and a plan whose gap is at least as long, such as `7640fb8ad320`.

## Caveats

- **The split method may differ.** Our blocks are contiguous in time and separated by a gap at least as long as the window, so no test window is next to a training window. SHARP does not say how its 60/20/20 is drawn. If it used random windows from overlapping sliding traces, its S1 number would be optimistic and the true gap smaller than 10 points.
- **Small test sets at window 346.** On the gap-346 plan, each test block holds few windows of 346 packets, so the step 4 intervals are wider than at w ≤ 50 (best method: 9.8 points wide against 6.8 at w50; `amplitude_stats` / `random_forest`: 26.7 against 7.1). For the best method the w346 lower bound (83.94) is still above the w50 point estimate (80.82).
- **One seed.** Every run uses seed 0. The intervals reflect test-set resampling, not training variance.
- **The steps are ordered.** Each share above depends on the order of the steps. The class-set step is measured on S1 only, so it may be larger or smaller on S1–S5.

## Run time

| Item | Runs | Commit | Wall clock (UTC) | Sum of run durations |
|---|---|---|---|---|
| `padova_sharp`, gap 64, w20–50 | 96 | `737986f` (84), `8962d69` (12) | 09:07–10:53 | 36 min |
| `padova_s1` | 72 | `737986f` | 10:53–12:53 | 120 min |
| `padova_sharp`, gap 346, w50/w346 | 48 | `738f1b9` | 13:58–14:50 | 51 min |

The gap-346 item computed the plan's 54 `paper_beamforming` block contexts on a cold cache with the parallel pool (`--workers 8`), in about 23 min. Peak child memory was 5.2 GB (`max_child_peak_rss_mb=5232`), well within the 11 GB machine. This settles the deferred pool-memory item.

The 48 gap-346 runs first recorded `dirty: true` at `d67bef6`. The only uncommitted change was the 6.3 spec file. Their recorded diff is byte-identical to `d67bef6..738f1b9`, so their provenance was rewritten to commit `738f1b9`, clean. The original files are kept in `runs_archive/provenance_backup_gap346_dirty/`.
