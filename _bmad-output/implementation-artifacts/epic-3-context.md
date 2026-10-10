# Epic 3 Context: Paper reproduction and leakage measurement

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Reproduce the published paper's accuracy with its window-level random split, and show what the same method scores under the leakage-free time-block split, so the thesis can quantify how much the random split inflates accuracy.

## Stories

- Story 3.1: Window-level random splitter for reference runs
- Story 3.2: Reproduce the paper and measure leakage

## Requirements & Constraints

- The paper's method (all `paper_beamforming` features + random forest) is recomputed in the pipeline, run once with a random window split (reference row) and once with the time-block split (comparison row).
- Random-window splits are valid only for reference runs; a comparison run must never use one. `role` (`comparison` | `reference`) is a required config field.
- Reference rows are reported separately, next to the published 99.41 / 99.59 / 97.37.
- `saved_dataset/*.npy` is only a test fixture checking the port.

## Technical Decisions

- The Splitter alone owns train/val/test assignment; its randomness comes from a seed in its own params, never from model, preprocessing, window length or selection.
- `random_window` assigns individual windows (`unit: window`) instead of packet blocks; its plan hash must differ from every time-block plan.
- With `role: reference`, `recording_mean_norm` may use `fit_scope: whole_recording`; config validation rejects that scope for comparison runs.
- Adapters self-register by snake_case name matching the module (`random_window` in `adapters/splitters/random_window.py`); splits are committed as `splits/<dataset>/<split_hash>.json`.

## Cross-Story Dependencies

- Story 3.2 depends on 3.1's splitter and on the Epic 2 harness (paper_beamforming, sweeps, report).
