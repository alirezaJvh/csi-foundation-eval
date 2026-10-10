# Proposal: Story 2.10 — compute `paper_beamforming` inner-window metrics once per packet

Status: proposal (input for `bmad-correct-course`). Written 2026-10-10 after Story 2.9.
Explainer with flowcharts: https://claude.ai/artifact/Cy4bxysz3doBGv3mTMTgnu

## Why

Story 2.9 added multi-core extraction, but it did not speed up `paper_beamforming`. Measured on the first `paper_beamforming` run of `experiments/classical_baseline.yaml` (window_len 20, r1, svm; 37,937 windows), see `spec-2-9-faster-feature-extraction-and-head-training.md` Implementation Notes:

| workers | extraction | whole run | peak RSS |
|---|---|---|---|
| 1 | 956 s | 16:29 | 1.33 GB |
| 4 | 989 s | 17:03 | 1.32 GB |

The implementing agent diagnosed the extractor as memory-bandwidth bound, from microbenchmarks; this has not been independently re-checked. The larger cause is redundant work: with stride 1, neighbouring windows recompute almost the same inner-window metrics.

## Current logic (`src/csi_bench/adapters/extractors/paper_beamforming.py`)

For each window `[a, a+L)` (window-relative inner start `START = 1`, `n_metric = L - 16`):
- doppler: inner windows of 8 packets × 40 weights, FFT → θ and max per inner position (computes `L-8` positions, keeps the first `L-16`).
- distance: inner windows of 16 packets × 40 weights, pair loop (`fix_distance_loop` flag) → θ and max (`L-16` positions). Dominant cost.
- flatness: 1 packet × 40 weights, variance over subcarriers → θ and min (computes `L-1`, keeps `L-16`).
- 6 features = `var(θ)` and `mean(max/min)` over the `L-16` positions.

Each inner metric at absolute packet `p` depends only on packets `p .. p+size-1`, yet it is recomputed by every window that contains it (about `L-16` times).

## Proposed logic

1. Once per recording and split block: for every absolute inner start `p` whose inner windows fit inside the block, compute the six per-position arrays (θ_dop, max_dop, θ_dist, max_dist, θ_flat, min_flat).
2. Per window `[a, a+L)`: slice positions `a+1 .. a+L-16` and take the same `np.var` / `np.mean` as today.

Step 1 does not depend on `window_len`, so it can be shared across all window lengths (e.g. cached per recording/block and extractor params).

## Hard requirement: identical results

- The new feature matrix must equal today's for the same windows (same values, same row order). Add an exact-equality test old vs new on synthetic and real windows; if bit-equality turns out impossible because of batched reductions, the decision on tolerance goes back to the human.
- The existing paper-parity test (`tests/test_paper_beamforming.py` vs `tests/oracles/paper_processing.py`) must still pass, for both `fix_distance_loop` values.
- Run ids and cached results may keep their meaning only if values are identical; otherwise bump `VERSION`.

## Expected gain (estimate from operation counts, not measured)

Distance computations per window today = `L-16`; new = 1 per packet.

| window_len | today per window | new per packet | approx. extraction speed-up |
|---|---|---|---|
| 20 | 4 | 1 | ~4× |
| 30 | 14 | 1 | ~14× |
| 40 | 24 | 1 | ~24× |
| 50 | 34 | 1 | ~34× |

Sharing step 1 across the four window lengths: about 1 unit instead of 4+14+24+34 = 76 for the sweep (~75× less extraction work).

## What it touches (decision needed in correct-course)

- The extractor port: today `extract(window_slices)` gets independent window slices (architecture: "extractors read only their own window's packets", "pure per input"). The new logic needs recording/block context plus the window positions. This is the item Story 2.9 listed as out of scope ("needs recording context in the extractor interface").
- Leakage stays safe: step 1 runs within each split block, and every inner position a window uses lies inside that window.
- The feature cache (runner-owned) may store the per-position arrays so all window lengths reuse them.

## Related open items from Story 2.9

- Story 2.9 is uncommitted and pending code review at the time of writing.
- Consider defaulting extraction workers to 1 (no gain measured, adds fork overhead and ~5 min test time) while keeping head jobs at `min(4, cpus)`.
- Head speed-up (`random_forest`, `knn`) was not timed; measure per-stage time (load, preprocess, extract, head fit) on a few sweep cells, since `svm` on 40–60k rows may dominate once extraction is fast.
