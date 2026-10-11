# Sprint Change Proposal: generalisation protocols, public datasets and the conditional next phase

Date: 2026-10-11 · Author: Jj (with Claude) · Mode: Incremental · Scope: Moderate
Inputs: Epic 3 results (Story 3.2), supervisor chat (2026-10-08), `docs/technical_guide_classical_ml_vs_dnn_for_rf_sensing.md`, `implementation-artifacts/deferred-work.md` (Epic 3 results discussion)

> **Renumbered 2026-10-11 (after approval):** epics now follow execution order. Old Epic 7 (generalisation protocols) is Epic 4, old Epic 4 (LWM) is Epic 5, old Epic 5 (other pretrained models) is Epic 7, and Epic 6 keeps its number. This document uses the new numbers. Earlier proposals (2026-10-10) and `epic-2-context.md` keep the numbering of their time.

## 1. Issue summary

Epic 3 closed with clear results, and new supervisor input arrived at the same time:

- **Ceiling.** The honest within-room classical baseline (time-block split, 216 runs) already reaches 98.74 / 99.85 / 98.49 test accuracy (r1 / r2 / mixed). The paper reproduction matched the published numbers (99.34 / 99.93 / 96.33 vs 99.41 / 99.59 / 97.37) and measured the leakage of the paper's own pipeline (+9.42 / +5.77 / +21.98 points). With 1–1.5 points of headroom and 19 test blocks, Epics 4–5 cannot separate methods within a room.
- **Missing protocol.** The supervisor's guide (§5.2) asks for strict out-of-distribution evaluation, train on room 1 and test on room 2. No epic covers it.
- **Datasets verified.** The supervisor accepted the public datasets and split them into two use cases. Activity recognition (Padova 80 MHz, CSI-Bench) comes first, and people counting comes later. Epic 6 is no longer blocked.
- **Trainable encoders.** The guide (§5.1) lists self-supervised encoders (RF-JEPA, RF-BYOL, SimCLR). They need training, which this phase excludes. The researcher wants them kept as a conditional next step.
- **Uncertainty.** All numbers are single-seed on a fixed split. Seeds change only head randomness, so test-set variance is unmeasured.
- **Report limit.** `report` refuses mixed split hashes, so cross-room plans and new datasets could not share one results store.

User constraints: keep all previous results, runs, code and epics (additive changes only), and keep the code decoupled so new datasets, models and protocols plug in as adapters.

## 2. Impact analysis

- **Epics 1–3:** unchanged and kept. Existing runs are moved to readable directory names, and nothing is deleted.
- **Epic 4 (new):** generalisation protocols. 4.1 readable run layout and a report section per protocol, 4.2 `room_holdout` splitter, 4.3 recording-bootstrap confidence intervals, 4.4 cross-room classical baseline.
- **Epic 5 (was Epic 4, LWM):** 5.1 and 5.2 are unchanged. 5.3 also scores LWM under the cross-room plans, with confidence intervals.
- **Epic 7 (was Epic 5, other pretrained models):** 7.1–7.3 are unchanged. 7.4 covers every (dataset, protocol) section.
- **Epic 6:** unblocked. The placeholder becomes 6.1 (Padova loader) and 6.2 (CSI-Bench loader).
- **Spec:** CAP-6 is rewritten, CAP-8 is added, and constraints, success signal and non-goals are amended. A "Next phase (conditional)" section is added and the resolved open question is removed.
- **Architecture:** AD-6, AD-11, AD-13 and AD-17 are amended, the capability map and naming are extended, and the Deferred table is reworded. No new port is added (AD-1 is unchanged).
- **Code (via stories):** `core/results_schema.py`, `core/metrics`, `runner/report.py`, a new `adapters/splitters/room_holdout.py`, new `adapters/datasets/*`, a one-off run migration, and new `experiments/*.yaml` and `splits/*`.
- **Leakage:** cross-room validation uses the train room only, so the test room is never seen before scoring.

## 3. Recommended approach

**Direct adjustment, additive.** Every new capability is an adapter behind an existing port (`Splitter`, `DatasetSource`) or a change confined to the results layout and the report. Rollback is not viable because the user requires everything kept. An MVP review is not needed: the scope grows within the same design. Effort is medium and risk is low.

Execution order: Epic 4 → Epic 5 → Epic 6 → Epic 7, in number order. Epic 4 comes first because every later score uses it, and Epic 6 precedes Epic 7 so the new models are scored on every verified dataset. If CSI-Bench (6.2) blocks on a spine amendment, it can be deferred without holding up Epic 7.

Risks:
- A class with a single test recording (r1 `empty`) gives no bootstrap spread. It is flagged in the report.
- The public-dataset formats are unverified. 6.1 and 6.2 confirm them first, and an amplitude-only format that breaks the `Recording` contract stops for a spine amendment.
- The run migration touches the results store, so it must be idempotent, delete nothing, and preserve the run count.

## 4. Detailed change proposals (all approved)

### 4.1 `specs/spec-lwm-wifi-har/SPEC.md`

- **CAP-6** intent: repeat CAP-3 to CAP-5 and CAP-8 on public activity-recognition datasets. The supervisor verified Padova 80 MHz and CSI-Bench, and the work starts with the easier import. Success: each dataset gets a loader and its own report section with no change to the evaluation code. People counting (e.g. WiMANS) stays a separate, later use case.
- **CAP-8 (new)** intent: measure out-of-distribution generalisation by training on one room and testing on the other (r1→r2, r2→r1), for every method, with uncertainty on every test score. Success: `report` shows a cross-room section next to the within-room one, and every test score carries a recording-bootstrap confidence interval.
- **Constraints:** the split protocols are `time_block` (primary, within-room), `random_window` (reference only) and `room_holdout` (cross-room OOD). Results are reported per dataset and per split protocol, never mixed in one table.
- **Success signal:** add a cross-room table, and the same tables on each verified public dataset.
- **Non-goals:** "Pretraining or fine-tuning any model in this phase; see 'Next phase (conditional)'."
- **Next phase (conditional), new section:** starts only if the frozen-model results (CAP-3, CAP-7) under CAP-8 justify it, for example if frozen embeddings don't beat classical features across rooms. Candidates:
  - trainable self-supervised encoders (not LLMs): RF-JEPA, RF-BYOL, SimCLR, on sanitized time × subcarrier tensors
  - Transformer baselines as in CSI-Bench
  - fine-tuning or LoRA of the frozen models
  - zero-shot transfer across datasets

  It needs a trainable-model port and probably a GPU.
- **Open Questions:** remove the public-dataset verification item.

### 4.2 `specs/spec-lwm-wifi-har/datasets.md`

- The section becomes "Public datasets — verified by supervisor (2026-10-08)", with the two use cases.
- Padova 80 MHz is in scope and first. It uses the same hardware family as the own data, probably has several environments, and Story 6.1 confirms the details.
- CSI-Bench is in scope and second. It is large and multi-device and its paper reports Transformer baselines. It may be amplitude-only, and Story 6.2 confirms that.
- WiMANS is kept for the later people-counting phase. EHUNAM is unchanged.
- Note: each dataset keeps its own class list. Mapping labels across datasets and zero-shot transfer belong to the next phase.

### 4.3 Architecture spine (`ARCHITECTURE-SPINE.md`)

- **AD-6:** add the `room_holdout` exception.
  - Params `train_rooms`, `test_rooms`, `fractions` and `gap_packets`.
  - Train-room recordings are split into time blocks for train and val, and test-room recordings go entirely to test.
  - Each direction is its own committed plan, valid for `role: comparison`.
  - Room settings are selections within a protocol, and cross-room is a protocol of its own.
  - Decided 2026-10-11: cross-room validation uses the train room only (strict OOD), and there is no oracle-selection row.
- **AD-17:** with `room_holdout`, `selection.rooms` must equal `train_rooms ∪ test_rooms`.
- **AD-11:** the `Evaluator` also computes a 95% confidence interval for test accuracy and macro-F1. It uses a class-stratified bootstrap over test recordings, as a pure function with a fixed resampling seed and `B` resamples. The interval is stored in `metrics.json`, and a class with a single test recording is flagged.
- **AD-13:**
  - The run directory becomes `runs/<dataset>/<splitter>/<slug>--<run_id>/`, with `slug = <extractors>-<selector>-<head>-<rooms>-w<window_len>`.
  - `run_id` remains the identity, and the layout lives only in `core/results_schema.py`.
  - A one-off migration moves existing runs and deletes nothing.
  - `report` prints one section per `(dataset, splitter)`, with room columns for `time_block` and `random_window` and direction columns for `room_holdout`. It refuses mixed split hashes only within one column, and cells show the score with its interval.
- **Capability map:** add CAP-8 (`splitters/room_holdout`, `core/metrics`, `report`), governed by AD-6, AD-11, AD-13 and AD-17.
- **Naming:** add `room_holdout`.
- **Deferred:** the "training deep networks" row now names self-supervised pretraining (SimCLR, BYOL, JEPA), fine-tuning or LoRA, and CNNs or Transformers on raw CSI. It points to the spec's "Next phase (conditional)".

### 4.4 `planning-artifacts/epics.md`

- **Requirements:**
  - FR17 is unblocked (Padova first, then CSI-Bench).
  - New FR18: the `room_holdout` splitter.
  - New FR19: recording-bootstrap confidence intervals.
  - New FR20: readable run layout and a report section per (dataset, splitter).
  - The coverage map adds FR18–FR20 under Epic 4.
- **Epic list:**
  - Epic 6 is renamed "Public datasets (activity recognition)", with 2 stories.
  - Epic 4 "Generalisation protocols" is added (FR18–FR20).
  - Order note: 4 → 5 → 6 → 7.
- **Epic 4 stories:**
  - 4.1 Readable run layout and a report section per protocol.
  - 4.2 `room_holdout` splitter.
  - 4.3 Confidence intervals from a recording bootstrap, backfilled from saved predictions with no retraining.
  - 4.4 Cross-room classical baseline (`experiments/classical_cross_room.yaml`).
  - The acceptance criteria are as approved in the session.
- **Story 5.3 (was 4.3):** append that `experiments/lwm.yaml` also runs under both `room_holdout` plans, and LWM rows appear in the cross-room section with intervals.
- **Story 7.4 (was 5.4):** `full_comparison.yaml` runs under `time_block` and both `room_holdout` plans on `tor_vergata`, and under `time_block` on each verified public dataset. Each (dataset, protocol) section lists every method with its best validation variant and intervals. The other criteria are unchanged.
- **Epic 6:**
  - 6.1 Padova 80 MHz loader: confirm format, classes and environments first. Environments map to `room` so `room_holdout` gives a cross-environment test. The `dataset:`-only sweep change and the `git diff` limits apply.
  - 6.2 CSI-Bench loader: the same criteria. If the data is amplitude-only and breaks the `Recording` contract, the story stops for a spine amendment.

### 4.5 Tracking files

- **`sprint-status.yaml`:**
  - Insert the generalisation block as Epic 4 (4-1 … 4-4 and its retrospective, all `backlog`) after Epic 3, and renumber LWM to Epic 5 and the other pretrained models to Epic 7.
  - Replace `6-1-loader-for-the-first-verified-public-dataset` with `6-1-padova-80-mhz-loader` and `6-2-csi-bench-loader`.
  - Set `last_updated: 2026-10-11`.
- **`deferred-work.md`:** keep both Epic 3 results entries verbatim, each with a "→ Planned: Epic 4 …" line.

## 5. Implementation handoff

- **Scope: Moderate.** A backlog reorganisation, a spec and architecture amendment, and a new epic. Planning edits are applied directly. Stories are implemented through the developer workflow (`bmad-build`), one at a time, in the order above.
- **Order:** apply edits §4.1–§4.5. Then implement Stories 4.1 → 4.2 → 4.3 → 4.4. Then continue with Epic 5.
- **Success criteria:**
  - `report` prints separate `tor_vergata / time_block`, `random_window` and `room_holdout` sections from one `runs/` store.
  - Every cell has a confidence interval.
  - All 225 existing runs survive the migration with unchanged `run_id`s and metrics.
  - The cross-room classical baseline gives the first OOD numbers to compare with LWM.
