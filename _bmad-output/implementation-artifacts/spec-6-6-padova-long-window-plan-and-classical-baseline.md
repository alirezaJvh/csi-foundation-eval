---
title: 'Padova long-window plan and classical baseline'
type: 'feature'
created: '2026-11-10'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Padova is scored only at 20–50-packet windows, so it can't be compared with the CSI-Bench 1/2/3/5 s runs, and Story 6.7's long-window LWM rows have no classical baseline. The CSI-Bench long-window runs (576, all done, in `runs/csi_bench_hp/time_block/*-w{122,244,366,610}--*`) were made on the Mac at commit `05237c4`, which never reached `origin`, so their YAML and plan `d6b97e4f25b3` are missing from the repo.

**Approach:** Commit a Padova `time_block` plan (0.6/0.2/0.2, gap 865 → `5cfd50d7d8ae`) and `experiments/padova_long_windows.yaml` (Padova classical grid, windows 173/346/519/865 at ≈1/2/3/5 s, stride window/2, 3 rooms, 216 runs). Rebuild the CSI-Bench plan (gap 610 reproduces `d6b97e4f25b3` exactly) and YAML from the run configs, with expansion tests; run the Padova sweep and `report`.

**Facts (2026-11-10):** Padova: 105 recordings, 6,599–52,684 packets; at gap 865 every (room, class, part) has ≥ 3 windows of 865 (laboratory val/test lowest). Mac runs: grid = `csi_bench_hp_classical_baseline.yaml` with `(window_len, stride)` ∈ {(122,61), (244,122), (366,183), (610,305)}, split `d6b97e4f25b3`, `ant2`, seed 0, role `comparison`, 8 room sets; 144 per window.

</frozen-after-approval>

## Implementation Notes

- **Files:** `splits/padova/5cfd50d7d8ae.json`, `splits/csi_bench_hp/d6b97e4f25b3.json` (generate with `TimeBlock(Params(gap_packets=…)).split(recs, ds)` + `write_split_plan`); `experiments/padova_long_windows.yaml`, `experiments/csi_bench_hp_long_windows.yaml` (header style of `padova_sharp.yaml`; pair window and stride with a `"@window"` axis since stride is window/2, floored: 173→86, 346→173, 519→259, 865→432).
- **Tests:** `tests/test_sweep.py` — two expansion tests (count, dataset/split/stride pairs, grid equals the short-window baseline apart from windows, plan params); the CSI-Bench one pins the expanded run ids to the 576 Mac run ids (sha256 of sorted ids). `tests/test_padova.py`, `tests/test_csi_bench_hp.py` — real-data pins for the new plans (skipped without data).
- **Implicit split goes away:** with two block plans, `padova` and `csi_bench_hp` no longer resolve a split implicitly (like `padova_sharp`). Every committed YAML already sets `split:`. Update the two assertions in `tests/test_sweep.py` (`find_split("padova")`, `find_split("csi_bench_hp")`) to expect the "several split plans" `DataError`; `describe padova|csi_bench_hp` now needs `--split`.
- **No code change** outside `splits/`, `experiments/`, `tests/`. Existing runs untouched; the CSI-Bench sweep must expand to existing run ids (nothing to rerun).
- **Run:** `uv run csi-bench sweep experiments/padova_long_windows.yaml`, then `uv run csi-bench report`; check sections `padova / time_block, gap_packets 865` and `csi_bench_hp / time_block, gap_packets 610`.
- **Status (2026-11-11):** plans, YAMLs and tests done; full `pytest` 668 passed, 1 skipped. The Padova sweep (`--workers 8`) was stopped by host memory pressure (11 GB) after 18/216 runs; rerun with `--workers 2`–`3` (resumes), then `report`. Story stays `in-progress` until then. Code review skipped at the user's request; nothing committed.
- **Closed (2026-11-11, Mac):** merged `origin/main` into the Mac branch holding `05237c4` (`56798ca`; conflicts resolved to origin's YAML and tests, both 6.4 change-log lines kept; `pytest` 664 passed, 8 skipped). Padova sweep 216/216 `done`, 0 dirty, commit `56798ca`, cpu, 20:20–20:39 UTC (18 min of run time). `report` exits 0 with `padova / time_block, gap_packets 865` and `csi_bench_hp / time_block, gap_packets 610` (576 adopted runs). Best by val per room (test F1): bedroom 75.60, laboratory 79.53, living_room 77.23, all w865. Write-up: `_bmad-output/reports/padova-long-windows-6-6.md`, section text `padova-long-windows-6-6.txt`.
