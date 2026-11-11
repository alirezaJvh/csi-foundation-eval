---
title: 'LWM on Padova and CSI-Bench'
type: 'feature'
created: '2026-11-10'
status: 'in-progress'
baseline_commit: 'a33195179c775e09ac5189ed16ecbd64fadc57a5'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-6-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Frozen LWM has rows only for `tor_vergata`; Padova and CSI-Bench have classical rows only, so the LWM-vs-classical comparison doesn't cover every dataset before Epic 7.

**Approach:** Add four LWM sweeps on the shared reduced grid from sprint-change-proposal-2026-11-10 (presets `lwm_rescale_phase`, `lwm_amplitude`; embedding `cls`; pooler `mean_std`; heads `svm`, `mlp`, `logreg`; seed 0; `role: comparison`; `feature_selection: null`), each with an expansion test, then run them and `report`.

## Boundaries & Constraints

**Always:**
- Sweeps (header: run count, stride rule, why each grid value was kept, citing the `tor_vergata` medians):

| File | Dataset / plans | Antennas | Windows, stride | Columns | Runs |
|---|---|---|---|---|---|
| `experiments/padova_lwm.yaml` | `padova`: `b6010d87ce4d` (gap 64); `room_holdout` `725a12be6b83`, `99d235eb2f0e` | `ant2`, `ant4` | 10, 30, 50; stride 9 | 3 rooms + 2 directions | 180 |
| `experiments/padova_lwm_long.yaml` | `padova`: `5cfd50d7d8ae` (gap 865, Story 6.6) | `ant2`, `ant4` | 173/86, 346/173, 519/259, 865/432 | 3 rooms | 144 |
| `experiments/padova_sharp_lwm.yaml` | `padova_sharp`: `7640fb8ad320` (gap 346) | `ant2`, `ant4` | 346 and 50 (control); stride 9 | `bedroom` (S1) | 24 |
| `experiments/csi_bench_hp_lwm.yaml` | `csi_bench_hp`: `d6b97e4f25b3` (gap 610) | `ant2` | 122/61, 244/122, 366/183, 610/305 | 7 rooms + all 7 pooled | 192 |

- Windows, strides, rooms and plans equal the matching classical sweeps (`padova_classical_baseline`, `padova_cross_room`, `padova_long_windows`, `padova_sharp` item 1, `csi_bench_hp_long_windows`), so each LWM row lands in the same report section and column as its classical counterpart.
- Preprocessing `[antN, <lwm preset>]` (antenna preset first, as in `lwm.yaml`); `head.name` is the outermost axis, cheap heads first (`logreg`, `mlp`, `svm`).
- `room_holdout` uses short windows only (gap 64 must exceed the window); CSI-Bench uses long windows only (its short-window classical runs are incomplete).
- Before running `lwm_rescale_phase` on Padova, scan its 234 subcarriers for zero-magnitude bins (`phase_sanitize` caveat, deferred-work); if any exist, stop and ask.
- **Decision (2026-11-10): runs execute elsewhere.** This story adds the configs and tests and smoke-runs one cheap run per file here on CPU. The researcher runs the full sweeps on the Mac (`--device mps`) or a CUDA machine. The story stays `in-progress` until `report` shows the LWM rows.

**Never:**
- No code change outside `experiments/` and `tests/`; no new plans; no change to existing runs or classical configs.
- No `ant4` on CSI-Bench (2 receive chains); no other presets, embeddings, poolers or heads.

</frozen-after-approval>

## Code Map

- `experiments/lwm.yaml` -- base shape: `preprocessing: [ant2, lwm_rescale]`, `extractors: [{name: lwm, params: {embedding: cls}}]`, `pooler`, paired `@protocol` axis.
- `experiments/padova_cross_room.yaml` -- `@direction` pairs (`split` + `selection.rooms`) for the two `room_holdout` plans.
- `experiments/padova_long_windows.yaml`, `csi_bench_hp_long_windows.yaml` (Story 6.6) -- `@window` pairs (`window_len` + `stride`).
- `experiments/padova_sharp.yaml` item 1 -- gap-346 plan, `window_len: [50, 346]`, stride 9, `selection.rooms: [bedroom]`.
- `src/csi_bench/config/presets.py` -- `lwm_rescale_phase`, `lwm_amplitude`, `ant2`, `ant4`.
- `src/csi_bench/adapters/extractors/lwm/adapter.py` -- accepts 1–4 antennas, 234 and 58 subcarriers (checked: `_validate` passes `(2|4, n, 234)` and `(2, n, 58)`).
- `tests/test_sweep.py` -- `test_lwm_sweep_expands` and `_long_window_sweep` (6.6): reuse their style; assert each LWM config's `(dataset, split, window_len, stride, selection.rooms)` set equals the matching classical sweep's.

## Tasks & Acceptance

**Execution:**
- [x] `experiments/padova_lwm.yaml`, `padova_lwm_long.yaml`, `padova_sharp_lwm.yaml`, `csi_bench_hp_lwm.yaml` -- write per the table.
- [x] `tests/test_sweep.py` -- one expansion test per file: run count, grid values, cheap-heads-first order, cells match the classical counterpart.
- [x] Padova null-bin scan -- record the result in Implementation Notes.
- [ ] Smoke run here: one cheap cell per file (`logreg`, smallest window, smallest room set) via `csi-bench run` on CPU from a scratch directory with `splits/` and `data/` symlinked (`run` has no `--runs`; the store is `./runs`), so the real store only gets the researcher's device runs; record times in Implementation Notes.
- [ ] Full sweeps + `uv run csi-bench report` -- researcher, on the Mac / CUDA machine.

**Acceptance Criteria:**
- Given the four files, when expanded, then they give 180 + 144 + 24 + 192 = 540 distinct run ids.
- Given the finished runs, when `report` runs, then every Padova and CSI-Bench `(dataset, protocol)` section shows LWM rows next to the classical rows, with confidence intervals.

## Implementation Notes

- **Configs.** Four sweeps written per the table; expansion gives 180 + 144 + 24 + 192 = 540 distinct run ids (`test_story_6_7_lwm_sweeps_are_540_distinct_runs`). `head.name` is the first sweep axis (`logreg`, `mlp`, `svm`); `preprocessing.0` sweeps `ant2`/`ant4` (Padova only), `preprocessing.1` the two LWM presets. `padova_lwm.yaml` pairs `split` and `selection.rooms` in one `@protocol` axis (3 time_block rooms + 2 room_holdout directions); the long-window files reuse the classical `@window` pairs.
- **Window 10 is LWM-only.** The table's Padova short windows are 10, 30, 50, but the classical Padova sweeps run 20–50 (paper_beamforming needs ≥ 18 packets). So w30/w50 cells equal classical cells, and w10 cells keep the same dataset, plan, stride and room/direction column but have no classical row at that window. The test checks exactly this, not full equality. The other three files match their classical cells exactly.
- **Padova null-bin scan (2026-11-10).** All 105 `padova` recordings (1,870,984 packets per antenna, 4 antennas × 234 subcarriers): **no null bins.** No subcarrier is zero in more than 0.043 % of packets (the worst is subcarrier 227: 878 zero entries across all antennas). There are 17,584 isolated zero-magnitude entries in total (1e-5 of all values), spread over almost every subcarrier, so they look like quantisation zeros, not null/DC bins. The adapter already drops the null, DC and pilot indices (`_DROPPED`). Proceeded with `lwm_rescale_phase`. A zero entry gets phase 0 in `phase_sanitize`'s unwrap, a negligible per-packet effect at this rate. `padova_sharp` is a subset of the same files (S1), so it is covered.
- **Smoke runs (CPU, 1 torch thread, scratch dir with `splits/`, `data/`, `.cache/weights` symlinked; base config of each file with `logreg`, `ant2`, `lwm_rescale_phase`, smallest window, laboratory for the `padova` files):** only 1 of 4 finished. Scratch store: `/tmp/claude-1000/-home-jj-master/de0e1808-952f-4fc7-94b8-82d76e221441/scratchpad/smoke/runs`.
  - `padova_sharp_lwm` (w50, bedroom/S1, run `5d9c44d8e512`): **done in 4132 s (~69 min)**: preprocess 53 s, LWM extract 3954 s (15 S1 recordings, about 5–10 min each on 1 thread), head fit 113 s; peak RSS 4.3 GB. Test macro-F1 0.915 (95 % CI 0.895–0.932), val 0.970. This shows that the LWM pipeline works end to end on Padova data with `lwm_rescale_phase`.
  - `padova_lwm` (w10, laboratory, `0051858a92b6`), `padova_lwm_long` (w173, laboratory, `47d82c9efe45`), `csi_bench_hp_lwm` (w122, E01, `1f9a0ed16e85`): **killed, incomplete.** Each passed config validation, load and preprocess (45 s, 47 s, 124 s; 18,055, 1,636 and 34,352 windows), then the host ran out of memory during LWM extraction: these three ran in parallel with the sharp run and a full `pytest` on an 11 GB machine, at about 4 GB RSS each. Their `status.json` is left at `running` in the scratch store, and nothing was written to the real `runs/`. Not restarted.
  - Cost on CPU: about 5 min per 20k-packet Padova recording on 1 thread, so a full Padova embedding pass (1.87M packets) is about 8 h per (antennas, preset, fitted state) on CPU. The full sweeps need the GPU device. Run at most 2 LWM processes at a time on an 11 GB host.
- **Full `pytest` not completed here.** It was killed with the smoke runs during the memory spike. `ruff check`, `ruff format --check`, `lint-imports` and `pytest tests/test_sweep.py -k lwm` (6 tests) pass.
- **Checks (2026-11-11):** ruff, format, lint-imports pass; full `pytest` 668 passed, 1 skipped. Code review skipped at the user's request; nothing committed.

## Spec Change Log

## Verification

**Commands:**
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run lint-imports && uv run pytest` -- expected: all pass
- `uv run csi-bench report` -- expected: LWM rows in sections `padova / time_block`, `padova / room_holdout`, `padova / time_block, gap_packets 865`, `padova_sharp / time_block, gap_packets 346`, `csi_bench_hp / time_block, gap_packets 610`
