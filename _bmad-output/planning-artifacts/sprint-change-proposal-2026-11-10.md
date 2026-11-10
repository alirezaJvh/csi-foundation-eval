# Sprint Change Proposal: LWM on the public datasets, long windows and a GPU option

Date: 2026-11-10 · Author: Jj (with Claude) · Mode: Incremental · Scope: Moderate · Status: approved 2026-11-10, applied
Inputs: Story 5.3 LWM results on `tor_vergata` (4,500 runs in `runs/index.parquet`), Story 6.4 and the Mac CSI-Bench runs (`runs/csi_bench_hp/`), `experiments/lwm.yaml`, `ARCHITECTURE-SPINE.md` (AD-12, Stack), `runner/run.py`

## 1. Issue summary

LWM has rows only for `tor_vergata` (Story 5.3). Padova (Stories 6.1–6.3) and CSI-Bench (Story 6.4) have classical results only. The researcher wants LWM scored on both public datasets before Epic 7 adds the other pretrained models. This is a new requirement from the study owner, not a failure.

Copying `experiments/lwm.yaml` with only a `dataset:` change is not feasible:

- That grid is 4,500 runs (5 heads × 2 antenna sets × 5 presets × 2 embeddings × 9 pooling pairs × 5 protocols).
- LWM embeds every packet. `tor_vergata` has 63k packets per antenna, about 3.5–9 min per embedding pass on 1 torch thread. Padova has 1.87M (about 30×) and CSI-Bench HP 8.1M (about 128×), and the full grid needs about 68 passes.
- CSI-Bench HP has 2 receive chains, so `ant4` is invalid there.
- AD-12 fixes `device = "cpu"`, so embedding cannot use a GPU.

Evidence found while assessing the change:

- **Long-window CSI-Bench runs exist outside a story.** On the Mac, the classical grid ran at windows 122, 244, 366 and 610 (1, 2, 3 and 5 s at about 122 Hz), stride window/2, plan `d6b97e4f25b3`, 144 runs per window (576 done), at commit `05237c4`. That commit, its YAML and the plan are not on `origin`.
- **Parallel sweeps break the dataset cache.** Short-window CSI-Bench runs on the Mac failed with `DataError: …: cache file unreadable: data/csi_bench/Multitask/sub_Human_mat/.cache/v1/<sha>.npy` (w20: 25 failed, 7 done; several runs left at `running`). They ran as several sweep processes at once (commit `d424372`, `experiments/csi_bench_hp_parallel/`).

Median LWM test macro-F1 on `tor_vergata`, used to reduce the grid:

| Axis | `time_block` | `room_holdout` | Kept |
|---|---|---|---|
| Preset | rescale+phase 0.990, rescale+amplitude 0.989, power+phase 0.991, rescale 0.904, power 0.885 | rescale+amplitude 0.25, rescale+phase 0.245, others ≤ 0.22 | `lwm_rescale_phase`, `lwm_amplitude` |
| Embedding | cls 0.972, channel_mean 0.972 | cls 0.201, channel_mean 0.189 | `cls` |
| Pooler | mean_std 0.985, mean 0.965 | mean_std 0.216, mean 0.182 | `mean_std` |
| Window | w1 0.957 … w50 0.980 | w10 0.221 best | per dataset, see 4.1 |
| Head | svm 0.985, mlp 0.981, logreg 0.976, random_forest 0.964, knn 0.958 | mlp 0.231, knn 0.206, logreg 0.191, random_forest 0.19, svm 0.167 | `svm`, `mlp`, `logreg` |

## 2. Impact analysis

- **Epic 6:** three new stories, 6.5 (device option and parallel-safe cache), 6.6 (Padova long-window plan and classical baseline, adopting the CSI-Bench long-window runs) and 6.7 (LWM on Padova and CSI-Bench). Story 6.4 stays `done`; its spec gets a change-log note.
- **Epic 7:** scope unchanged; it starts after 6.7. Story 7.4 reuses the 6.6 and 6.7 configs.
- **Spec (`SPEC.md`):** no change. CAP-3 and CAP-6 already cover LWM on the public datasets.
- **Architecture:** AD-12 gains a device option and rules for reproducibility and the embedding cache key. The Stack row for torch changes from CPU-only wheels to CPU by default with MPS on macOS and an optional CUDA index.
- **Code (via stories):** `runner/run.py` (device), the LWM embedding cache key, `pyproject.toml` (torch sources), `adapters/datasets/csi_bench_hp.py` and `padova.py` (cache race), new `experiments/*` and `splits/padova/*`.
- **Results store:** no existing run changes. New runs only.
- **UI/UX:** not applicable.

## 3. Recommended approach

**Direct adjustment, additive.** New stories in Epic 6 behind existing ports, plus one architecture amendment.

- **Rollback** is not needed: no completed result is wrong.
- **MVP review** is not needed: scope stays within CAP-3, CAP-6 and CAP-8.
- **Effort:** medium. 6.5 is the only story with runner code; 6.6 and 6.7 are plans, configs and sweeps.
- **Risk:** low to medium. GPU embeddings may differ from CPU by floating-point noise, which the AD-12 amendment makes explicit. The cache-race cause is not yet confirmed; 6.5 must reproduce it before fixing it.
- **Time:** the 6.7 run time is estimated after 6.5 measures one embedding pass on CPU and on GPU. With stride window/2, long windows give few windows per block, so embedding, not the heads, should dominate.

Execution order: **6.5 → 6.6 → 6.7 → Epic 7.**

## 4. Detailed change proposals

### 4.1 Stories (`epics.md`, Epic 6, after Story 6.4)

#### Story 6.5: Device option and parallel-safe dataset cache

As a researcher,
I want LWM embedding on a GPU (MPS or CUDA) and dataset caches that survive parallel sweeps,
So that the public-dataset LWM runs finish in hours, not days.

**Acceptance Criteria:**

**Given** `--device cpu|mps|cuda` or `CSI_BENCH_DEVICE` (default `cpu`)
**When** a run starts
**Then** the runner sets the device and records it in provenance, adapters never choose it, and `run_id` does not include it

**Given** the LWM per-packet embedding cache
**Then** its key includes the device, so CPU and GPU embeddings are never mixed

**Given** the same config on one device
**When** it is rerun
**Then** it reproduces the same metrics; across devices only `run_id` is guaranteed equal, and this is documented

**Given** the torch sources in `pyproject.toml`
**Then** CPU stays the default, macOS wheels provide MPS, and Linux can select a CUDA index

**Given** two sweep processes loading `csi_bench_hp` (and `padova`) at the same time
**When** both convert or open the same cache entry
**Then** no run fails with `cache file unreadable`, and a test reproduces the race before the fix

#### Story 6.6: Padova long-window plan and classical baseline

As a researcher,
I want Padova scored at 1, 2, 3 and 5 s windows,
So that it matches the CSI-Bench long-window runs and the LWM long-window rows have a classical comparison.

**Acceptance Criteria:**

**Given** the `padova` adapter
**Then** a `time_block` plan (fractions 0.6/0.2/0.2, `gap_packets` 865) is committed under `splits/padova/`

**Given** `experiments/padova_long_windows.yaml`, the Padova classical grid with `window_len` [173, 346, 519, 865], stride window/2 and the 3 rooms
**When** `sweep` and then `report` run
**Then** the runs appear in their own section (`padova / time_block, gap_packets 865`)

**Given** the CSI-Bench long-window runs made on the Mac (commit `05237c4`)
**Then** their YAML and plan `d6b97e4f25b3` are committed with an expansion test, and `report` shows their section

#### Story 6.7: LWM on Padova and CSI-Bench

As a researcher,
I want frozen LWM rows for Padova and CSI-Bench,
So that the LWM-vs-classical comparison covers every dataset before Epic 7.

**Acceptance Criteria:**

**Given** the shared reduced grid (presets `lwm_rescale_phase` and `lwm_amplitude`, embedding `cls`, pooler `mean_std`, heads `svm`, `mlp` and `logreg`, seed 0, `role: comparison`, `feature_selection: null`), justified by the `tor_vergata` medians above
**Then** these sweeps exist, each with a header giving its run count, stride rule and why each grid value was kept, and an expansion test:

| File | Dataset and plan | Antennas | Windows and stride | Columns | Runs |
|---|---|---|---|---|---|
| `experiments/padova_lwm.yaml` | `padova`, `time_block` `b6010d87ce4d` (gap 64) and `room_holdout` `725a12be6b83`, `99d235eb2f0e` | `ant2`, `ant4` | 10, 30, 50; stride 9 | 3 rooms + 2 directions | 180 |
| `experiments/padova_lwm_long.yaml` | `padova`, the Story 6.6 plan (gap 865) | `ant2`, `ant4` | 173, 346, 519, 865; stride window/2 | 3 rooms | 144 |
| `experiments/padova_sharp_lwm.yaml` | `padova_sharp`, `7640fb8ad320` (gap 346) | `ant2`, `ant4` | 346 and 50 (control, as in Story 6.3) | S1 | 24 |
| `experiments/csi_bench_hp_lwm.yaml` | `csi_bench_hp`, `d6b97e4f25b3` | `ant2` | 122, 244, 366, 610; stride window/2 | 7 rooms + all 7 pooled | 192 |

**When** the sweeps and then `report` run
**Then** each (dataset, protocol) section shows the LWM rows next to the classical rows, with confidence intervals

`room_holdout` uses short windows only, because its plans have gap 64 and the gap must exceed the window. CSI-Bench uses long windows only, because its short-window classical runs are incomplete.

### 4.2 Execution order and Epic 6 intro (`epics.md`)

Epic List, after the "Remaining order (sprint-change-proposal-2026-11-02)" paragraph, ADD:

> **Remaining order** (sprint-change-proposal-2026-11-10): Story 6.5 → Story 6.6 → Story 6.7 → Epic 7. LWM is scored on Padova and CSI-Bench before the other pretrained models, so Epic 7 adds rows to tables that already hold LWM on every dataset. Story 6.5 (device option, cache race fix) comes first because 6.7's embedding cost depends on it; 6.6 commits the long-window plans 6.7 uses.

Epic 6 intro, OLD:

> … Padova 80 MHz comes first, as the likely easiest import, followed by its cross-room and SHARP-comparable runs; CSI-Bench follows.

NEW:

> … Padova 80 MHz comes first, as the likely easiest import, followed by its cross-room and SHARP-comparable runs; CSI-Bench follows. Long-window runs (1, 2, 3, 5 s) and frozen LWM then cover both public datasets, with a runner device option so LWM embedding can use a GPU.

### 4.3 Architecture (`ARCHITECTURE-SPINE.md`)

AD-12 Rule, OLD:

> - The runner sets `device = "cpu"`, the torch thread count and the determinism flags, and records them. Adapters never set them.

NEW:

> - The runner sets the device, the torch thread count and the determinism flags, and records them in provenance. Adapters never set them.
> - Device is `cpu` by default; `--device` or `CSI_BENCH_DEVICE` selects `mps` or `cuda`. It is `affects_result=False`: it never changes `run_id`.
> - Reproducibility: the same config on the same device reproduces the same metrics. Across devices only `run_id` is guaranteed equal; metrics may differ by floating-point noise, and provenance shows which device produced them.
> - Per-packet embedding caches include the device in their key, so CPU and GPU embeddings are never mixed in one run.
> - Only torch extractors use the device. Sklearn heads, poolers and classical extractors always run on CPU.

Stack, OLD:

> | torch / torchvision (CPU index via `tool.uv.sources`) | 2.14.1+cpu / 0.29.1+cpu |

NEW:

> | torch / torchvision (CPU index by default via `tool.uv.sources`; macOS wheels include MPS; an optional `cuda` uv extra selects the CUDA index on Linux) | 2.14.1 / 0.29.1 |

The exact uv mechanism for the CUDA index is decided in Story 6.5.

### 4.4 Tracking

`sprint-status.yaml`:

- ADD comment: `# Remaining order 2026-11-10 (sprint-change-proposal-2026-11-10): 6.5, 6.6, 6.7, Epic 7`
- ADD under `epic-6`, after `6-4-csi-bench-loader: done`:
  - `6-5-device-option-and-parallel-safe-dataset-cache: backlog`
  - `6-6-padova-long-window-plan-and-classical-baseline: backlog`
  - `6-7-lwm-on-padova-and-csi-bench: backlog`
- `epic-6` stays `in-progress`; `last_updated: 2026-11-10`.

`spec-6-4-csi-bench-loader.md`, Spec Change Log, ADD:

> - 2026-11-10: After 6.4 was done, the Mac run added long-window CSI-Bench runs (w122/244/366/610, stride w/2, plan `d6b97e4f25b3`, commit `05237c4`) outside a story; Story 6.6 adopts and commits them. Parallel sweeps on the Mac hit `cache file unreadable` (w20 runs); Story 6.5 fixes it. The short-window classical CSI-Bench runs are incomplete and are not rerun now.

## 5. Implementation handoff

- **Scope:** Moderate. New stories and an architecture amendment; no replan.
- **Prerequisite (researcher):** push commit `05237c4` from the Mac, so plan `d6b97e4f25b3` and its YAML reach `origin`; then `git fetch origin && git merge --ff-only origin/main` on the mini-PC.
- **Developer (`bmad-build`):** Story 6.5, then 6.6, then 6.7, each in a fresh context.
- **Success criteria:**
  - 6.5: a GPU embedding pass works on the Mac (MPS) and on a CUDA machine, its time is measured against CPU, and the cache-race test passes.
  - 6.6: both long-window sections appear in `report`.
  - 6.7: 540 LWM runs done, and every Padova and CSI-Bench section shows LWM rows next to the classical rows.
