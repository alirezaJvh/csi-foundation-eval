# Sprint Change Proposal: honest validation scores, Padova follow-ups and CSI-Bench findings

Date: 2026-11-02 · Author: Jj (with Claude) · Mode: Batch · Scope: Moderate · Status: approved 2026-11-02, applied
Inputs: Story 6.1 results (216 Padova runs), supervisor questions (2026-10-30), SHARP paper (arXiv 2103.09924), CSI-Bench format scan (2026-11-02), `core/head_search.py`, `runner/run.py`, `runner/report.py`

## 1. Issue summary

The supervisor asked three questions about the Padova results: are they similar to the paper, is the pipeline the same for both datasets, and are train and test in the same room? Answering them exposed five issues.

1. **Reported validation scores are in-sample (new, affects every table).** A head tunes on val and then refits on train ∪ val (`core/head_search.py:47`), as AD-11 requires. `runner/run.py:664` then scores val with that refitted model, so val is scored on data the model trained on. The honest score, `head.val_score`, is computed but is not what the val metrics contain. `runner/report.py:163` selects each table's best variant on the inflated number, which favours heads that memorise. Test scores are not contaminated, but which variant each table shows, and therefore its number, may change.

   | Head | Padova val | Padova test | Runs at val = 1.0 | Tor Vergata val | Tor Vergata test | Runs at val = 1.0 |
   |---|---|---|---|---|---|---|
   | `random_forest` | 0.735 | 0.520 | 29 of 72 | 1.000 | 0.956 | 512 of 615 |
   | `knn` | 0.630 | 0.488 | 0 of 72 | 1.000 | 0.952 | 334 of 612 |
   | `svm` | 0.554 | 0.516 | 0 of 72 | 0.999 | 0.978 | 237 of 612 |
   | `logreg` | | | | 0.998 | 0.976 | 158 of 540 |
   | `mlp` | | | | 1.000 | 0.981 | 414 of 540 |

   Values are medians of macro-F1 under `time_block`. How far the tables will move is unknown until the runs are rescored.

2. **Padova has no cross-room result.** Story 6.1 excluded `room_holdout`, so every Padova score is same-room, same-recording, later in time. SHARP trains on S1 and tests on other days, people and rooms.

3. **The gap to SHARP is unexplained.** SHARP reports 96–99.8% accuracy on 5 classes and 95.4–99.5% on 8 classes (S1, S2, S6). Provisional macro-F1 here is 0.53–0.78 on 8 classes. The protocols, class sets, window lengths and models all differ, and none has been tested as the cause. The closest SHARP protocol is its S1 test: a 60/20/20 split within S1. The SHARP input window is about 2 s; ours are 20–50 packets (0.12–0.29 s at 173 Hz).

4. **The Padova citation is doubtful.** `datasets.md` and `epic-6-context.md` cite arXiv 2305.03170. The abstract of that paper describes 13 subjects and 23.6 GB. The files in `data/padova/` are 105 `.mat` files (5.7 GB) in sets S1–S7 across a bedroom, a living room and a laboratory, which match SHARP's Table 1.

5. **CSI-Bench is unblocked, with constraints.** The H5 files are amplitude-only and fail the `Recording` contract. The raw `Multitask/sub_Human_mat/` files hold complex CSI, so the 6.2 stop rule does not fire and no spine amendment is needed. `data/csi_bench/` is not yet git-ignored and holds a 79 GB zip.

## 2. Impact analysis

- **Epic 4:** new Story 4.5 fixes the validation score and rescores existing runs.
- **Epic 5 and all earlier tables:** no code change beyond 4.5, but selected variants may change. `_bmad-output/reports/lwm-vs-classical.html` must be regenerated.
- **Epic 6:** new Stories 6.3 and 6.4. Story 6.2 gains the confirmed format and scope.
- **Epic 7:** unchanged. Its stories inherit the fix.
- **Spec (`SPEC.md`):** no capability changes. `datasets.md` gets the corrected Padova row and the CSI-Bench format.
- **Architecture:** AD-11 gains one rule. No new port.
- **Code (via stories):** `runner/run.py`, `core/results_schema.py`, `runner/report.py`, `adapters/datasets/padova.py`, a new CSI-Bench adapter, new `experiments/*` and `splits/*`.
- **Results store:** 5,084 reported runs need a new score: `tor_vergata/time_block` 2,919, `tor_vergata/room_holdout` 1,946, `padova/time_block` 216, `tor_vergata/random_window` 3.
- **Supervisor:** the Padova numbers already quoted or drafted are provisional until 4.5 is done.
- **UI/UX:** not applicable.

## 3. Recommended approach

**Direct adjustment, additive.** The honest score already exists in the code, and the other changes are new stories, plans and adapters behind existing ports.

- **Rollback** is not viable. Test scores are valid, and earlier results must be kept.
- **MVP review** is not needed. Scope stays within CAP-5, CAP-6 and CAP-8.
- **Effort:** medium, mostly rerun time for 4.5. Measure it on a sample before starting the full rescore.
- **Risk:** low to medium. Headline tables may change, which is the purpose of the fix.

Execution order: **4.5, then 6.3, 6.4, 6.2, then Epic 7.** Story 4.5 goes first so no further sweep is selected on the inflated score. Stories 6.3 and 6.4 answer the supervisor's open questions and are cheaper than 6.2. Story 6.2 keeps its number, so here the numbers do not follow execution order.

**Deferred:** a Doppler feature family and a CNN. The CNN needs a trainable-model port, which the spec keeps for "Next phase (conditional)". Revisit after 6.4.

## 4. Detailed change proposals

### 4.1 New stories in `epics.md`

#### Story 4.5: Honest validation score for selection (after Story 4.4)

As a researcher,
I want variants selected on a validation score from a model that did not train on val,
So that every comparison table picks its best variant fairly.

**Acceptance Criteria:**

**Given** a finished run
**When** `metrics.json` is written
**Then** it records the tuning-stage val macro-F1 (`head.val_score` of the chosen candidate) per seed, with its mean and std over seeds
**And** the existing `val` metrics remain, documented as in-sample diagnostics of the refitted model

**Given** the run index and `report`
**When** the best variant per method and room setting is chosen, including for the leakage rows
**Then** selection uses the tuning-stage score
**And** a done run without that score is excluded, with a note giving how many were excluded and the command that fixes them

**Given** a synthetic case where a memorising head wins on the in-sample val but loses on the tuning-stage score
**When** `report` selects
**Then** the head with the better tuning-stage score wins, and a test fails if selection reverts to the in-sample score

**Given** the existing store
**When** the 5,084 reported runs are rerun
**Then** a copy of every old `metrics.json` is kept first, outside `runs/`, and no run or result is lost
**And** run ids are unchanged, and each run's test metrics match the old ones, with any mismatch listed rather than ignored
**And** a before/after table per `(dataset, splitter)` section shows every cell whose selected variant or score changed
**And** `_bmad-output/reports/lwm-vs-classical.html` is regenerated

#### Story 6.3: Padova cross-room baseline (after Story 6.1)

As a researcher,
I want Padova scored across environments,
So that the supervisor's same-room question has a measured answer and the result is comparable to SHARP's S6 and S7 tests.

**Acceptance Criteria:**

**Given** the `padova` adapter
**When** `room_holdout` plans are generated
**Then** the plans `bedroom→living_room` and `bedroom→laboratory` are committed under `splits/padova/`, with validation from the train room only

**Given** `experiments/padova_cross_room.yaml`, using the Padova baseline grid with `stride: 9` (72 runs per direction, 144 in total)
**When** `sweep` and then `report` run
**Then** a `padova / room_holdout` section appears with one column per direction and confidence intervals
**And** `git diff` shows no change outside `splits/` and `experiments/`

**And** the report states that this protocol trains on the whole bedroom (S1–S5), whereas SHARP trains on S1 only

#### Story 6.4: Padova SHARP-comparable run (after Story 6.3)

As a researcher,
I want classical results on Padova under SHARP's closest protocol,
So that the gap to the paper is measured against its class set, data subset and window length.

**Acceptance Criteria:**

**Given** the `padova` adapter module
**When** a second dataset `padova_sharp` is registered in `adapters/datasets/padova.py`, sharing its loading code
**Then** it loads only the S1 sets, and its classes are `empty, sitting, walking, running, jumping`
**And** it passes the `DatasetSource` contract test, has its own committed `time_block` split (0.6/0.2/0.2), and works with `describe`
**And** its results appear in their own report section, separate from `padova`

**Given** `experiments/padova_sharp.yaml`
**When** `sweep` runs
**Then** it covers `window_len` 346 (about 2 s) as well as the 20–50 packet windows, with the existing feature families and heads
**And** `amplitude_stats` also runs on all 4 antennas, while `paper_beamforming` stays on 2 antennas because it requires exactly 2

**Given** the results
**When** the story closes
**Then** the spec records, next to SHARP's S1 number, how much of the gap each change closed (class set, data subset, window length)
**And** it records whether a Doppler family is worth proposing

### 4.2 Changes to existing stories in `epics.md`

**Story 6.2, new paragraph after the acceptance criteria**

OLD: no format notes.

NEW:

> **Format confirmed 2026-11-02 (recorded here so the story does not stop on its rule).** Use `Multitask/sub_Human_mat/` (`csi_trace.csi`, complex128, `[tx, rx, subcarriers, packets]`, plus `mactimer`). The H5 files are amplitude-only and are not used. There are 1,778 HAR files, 5 classes (`walking-case1`–`4` merge into `walking`), environments E01–E06 and E27 as rooms, and no empty class. No file has non-finite values, all-zero packets or all-zero subcarriers. Scope for this story: the HP devices (438 files, 2×2×58, about 122 Hz), using `tx0` with `rx0` and `rx1` as antennas `[0, 1]`, as adapter `csi_bench_hp`. The 1×4×14 devices are a later, separate adapter. Exclude `EightreePlug` (bursty timing) and the 6 files under 1,000 packets. Do not use `mactimer` for timing. The story must confirm the antenna pairing across more than the one file sampled so far, and check whether `paper_beamforming`, `amplitude_stats` and LWM accept 58 subcarriers.

Rationale: the format is settled, so the story should not repeat the scan. Most environments have one user, so room and person are confounded, and the report must say so.

**Epic 6 intro**

OLD: `Padova 80 MHz comes first, as the likely easiest import; CSI-Bench follows.`

NEW: `Padova 80 MHz comes first, as the likely easiest import, followed by its cross-room and SHARP-comparable runs; CSI-Bench follows.`

**FR6**

OLD: `Train the classifiers ..., choosing hyperparameters on val. Evaluate accuracy, macro-F1, per-class scores and the confusion matrix on val and on test, with mean ± std over seeds.`

NEW: `Train the classifiers ..., choosing hyperparameters on val. Evaluate accuracy, macro-F1, per-class scores and the confusion matrix on val and on test, with mean ± std over seeds. Also record the val macro-F1 of the chosen candidate before it is refitted on train ∪ val; this is the score used for selection.`

**FR12, second bullet**

OLD: `Each cell is the variant with the best val macro-F1, reported with its test score.`

NEW: `Each cell is the variant with the best tuning-stage val macro-F1 (from a model not trained on val), reported with its test score.`

### 4.3 `ARCHITECTURE-SPINE.md`, AD-11

Add after the `Evaluator` bullet:

> - Variant selection uses the tuning-stage val score: the val macro-F1 of the chosen candidate, fitted on train only. The val metrics of the refitted model are in-sample diagnostics and never select anything.

Rationale: AD-11 specifies the refit but not which val score counts. The refit itself stays.

### 4.4 Citations and format notes

**`specs/spec-lwm-wifi-har/datasets.md`, Padova row, and `implementation-artifacts/epic-6-context.md`, Sources**

OLD: cites arXiv 2305.03170 as the paper for the loaded data.

NEW: cites SHARP (Meneghello et al., IEEE TMC 2022, doi 10.1109/TMC.2022.3185681, arXiv 2103.09924) as the source of the S1–S7 data in `data/padova/`. It notes that arXiv 2305.03170 describes a related, larger dataset that is not the one loaded.

Applied with the download source of `dataset.zip` unconfirmed. The match rests on the file layout and SHARP's Table 1.

**`datasets.md`, CSI-Bench row**

OLD: `Story 6.2 confirms the format (possibly amplitude-only), classes and splits.`

NEW: `Format confirmed 2026-11-02: H5 is amplitude-only; Multitask/sub_Human_mat holds complex CSI and fits the contract. Kaggle, CC BY-NC-ND 4.0.`

**`.gitignore`** (separate chore, before any commit)

Already done by Jj: `.gitignore` has `data/csi_bench`.

### 4.5 `sprint-status.yaml`

Add, all as `backlog`:

- `4-5-honest-validation-score-for-selection` under `epic-4`
- `6-3-padova-cross-room-baseline` and `6-4-padova-sharp-comparable-run` under `epic-6`

## 5. Implementation handoff

**Scope: Moderate.** Backlog changes plus code in `runner/` and `core/`, with no replan.

| Who | What |
|---|---|
| Developer (`bmad-correct-course`, on approval) | Apply sections 4.1–4.5 to the planning documents and `sprint-status.yaml` |
| Developer (`bmad-build`, fresh context per story) | Stories 4.5, 6.3, 6.4, 6.2, in that order |
| Jj | Confirm the Padova download source. Tell the supervisor the current Padova numbers are provisional, then send the updated table after 4.5 |

**Success criteria:**
- `report` selects on the tuning-stage score in every section, and a test guards this.
- All 5,084 reported runs are rescored with no results lost, and test metric mismatches are listed.
- A before/after table exists for every section.
- Padova has `time_block`, `room_holdout` and SHARP-comparable sections.
- CSI-Bench has its own section, or a recorded reason why not.
