---
title: 'LWM preprocessing variants'
type: 'feature'
created: '2026-10-16'
status: 'done'
baseline_commit: '0b15d752a8ce1572174d977cfcabc29ce0e3f686'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-5-context.md'
  - '{project-root}/_bmad-output/specs/spec-lwm-wifi-har/preprocessing.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Only `global_rescale` exists today, so the LWM domain-gap variants in `preprocessing.md` (per-packet power, phase sanitisation, amplitude-only) can't be expressed as experiment variables.

**Approach:** Add three stateless `PreprocessStep` adapters (`per_packet_power`, `phase_sanitize`, `amplitude_only`) and antenna-agnostic LWM presets that cover the 4 variants. The presets compose with the existing `ant2`/`ant4` presets in a `preprocessing:` list.

## Boundaries & Constraints

**Always:** Each step follows the existing adapter shape (`@register("preprocess", ...)`, `Params` with `extra="forbid"`, `VERSION = 1`). `fit` returns `{}`, and `transform` is pure and returns complex64. All three work per packet only, so nothing leaks across split blocks.
- `per_packet_power`: divide each packet by `sqrt(mean |h|²)` taken over all of its (selected) antennas and subcarriers, so the packet has unit average power. This keeps the power ratios between antennas. A zero-power packet is left as-is (no NaN).
- `phase_sanitize`: unwrap the phase across subcarriers (`np.unwrap`, axis = subcarriers), least-squares fit `a·k + b` with `k = 0..n_sub-1` (the dataset does not carry subcarrier indices), then multiply the CSI by `exp(-j(a·k + b))`. Magnitudes are unchanged. **Decision (Jj, 2026-10-16): one fit per packet, shared across antennas.** Fit `φ_r(k) = a·k + b_r` jointly over all antennas r of the packet (one slope `a`, per-antenna intercepts `b_r` only to absorb unwrap offsets), then subtract `a·k + b` with `b = angle(Σ_r exp(j·b_r))` (the circular mean). The relative phase between antennas is kept.
- `amplitude_only`: `|csi|` cast to complex64 (phase 0).
- Presets in `config/presets.py`: `lwm_rescale` = [global_rescale]; `lwm_power` = [per_packet_power]; `lwm_rescale_phase` = [global_rescale, phase_sanitize]; `lwm_power_phase` = [per_packet_power, phase_sanitize]; `lwm_amplitude` = [global_rescale, amplitude_only] (rescaling is mandatory for LWM).

**Never:** Change existing steps, presets or any existing `run_id`. Add poolers, `experiments/lwm.yaml` or runner changes (5.3). Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| power | random CSI, amplitude ~1e12 | every packet has `mean |h|² == 1` (rtol 1e-5) | — |
| zero packet | one all-zero packet | that packet stays zero, others normalised, all finite | — |
| phase slope | CSI `x·exp(j(a_p·k + b_p))` with random `a_p` in [-0.6, 0.0], `b_p` per packet | `sanitize(x·slope) ≈ sanitize(x)` within atol 1e-4; magnitudes unchanged | — |
| antennas keep relative phase | 4 antennas with fixed offsets `c_r` plus a per-packet slope | `angle(out[r]·conj(out[0]))` ≈ `c_r − c_0` (wrapped, atol 1e-4) | — |
| pure linear phase | constant amplitude, phase `a·k + b` | output phase ≈ 0 everywhere | — |
| amplitude only | any CSI | `out == |csi|`, imaginary part 0 | — |
| preset use | `preprocessing: [ant2, lwm_power_phase]` | expands to antennas, per_packet_power, phase_sanitize in order | — |

</frozen-after-approval>

## Code Map

- `src/csi_bench/adapters/preprocess/global_rescale.py` -- adapter shape to copy (stateless variant: `fit` returns `{}`; params optional in `__init__`).
- `src/csi_bench/adapters/preprocess/{per_packet_power,phase_sanitize,amplitude_only}.py` (new) -- one step each. `discover()` registers them automatically.
- `src/csi_bench/config/presets.py` -- add the 5 `lwm_*` presets. `_expand_preprocessing` (`config/experiment.py:42`) already expands several preset names in one list.
- `tests/contracts/preprocess_step.py` -- `check_preprocess_step` is used unchanged. Stateless steps trivially pass its train-only check.
- `tests/test_preprocess.py` -- `_steps()`/`test_contract` parametrisation: extend it to the new steps. `test_registered_steps_all_covered` hard-codes the registry names, so update the list. `_recs()` gives `(4, 100, 3)` CSI; the phase tests need their own fixture with about 234 subcarriers.
- `tests/test_experiment_config.py:179` -- pattern for a preset-expansion test.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/adapters/preprocess/per_packet_power.py`, `phase_sanitize.py`, `amplitude_only.py` -- the three steps as specified -- the new variants
- [x] `src/csi_bench/config/presets.py` -- the 5 `lwm_*` presets -- the 4 variants as a sweep axis for 5.3
- [x] `tests/test_preprocess.py` -- contract test for each new step, the matrix rows, the updated registry-name list, and preset expansion order (`ant2` + `lwm_power_phase`) -- coverage

**Acceptance Criteria:**
- Given the new steps, when `check_preprocess_step` runs, then each passes.
- Given synthetic CSI with an added linear phase slope, when `phase_sanitize` is applied, then the slope is removed to within tolerance.
- Given the whole suite, when `pytest`, `ruff`, `ruff format --check` and `lint-imports` run, then they pass, and `ty` reports no new errors in touched files.

### Review Findings

- [x] [Review][Patch] Only `lwm_power_phase` has its exact step chain pinned by a test. The other four presets are checked only for "names ⊆ registry", so dropping `global_rescale` from `lwm_amplitude`, dropping `phase_sanitize` from `lwm_rescale_phase`, or reordering steps would all pass. Parametrize an exact `_expand_preprocessing([preset])` check over all 5. [tests/test_preprocess.py:212]
- [x] [Review][Patch] The preset comment doesn't say the antenna preset must come first. `[lwm_power, ant2]` passes validation but loses unit power, and the phase fit would include dropped antennas. Say "put ant2/ant4 first" in the comment. [src/csi_bench/config/presets.py:17]
- [x] [Review][Patch] `test_contract` hard-codes `range(6)`, so a step added to `_steps()` silently skips the contract test. Use `range(len(_steps()))`. [tests/test_preprocess.py:50]
- [x] [Review][Defer] `phase_sanitize` treats zero-magnitude bins as phase 0, which can break `np.unwrap` and bias the shared slope [src/csi_bench/adapters/preprocess/phase_sanitize.py:37] — deferred: unverified, would be medium. `tor_vergata` has no zero subcarriers and no zero packets (1 zero sample in 63,272 packets). To settle it, scan the Epic 6 datasets for null/DC bins before reusing this step on them.

**Rejected:**
- `false` — Non-finite CSI spreads silently through `per_packet_power` / `phase_sanitize`: the loaded data has 0 non-finite values, and `runner/run.py:447` raises `ExtractorError` on non-finite features in any case.
- `false` — `test_lwm_presets_use_registered_steps` depends on test order through `discover()`: after the purge, re-imports register again. A second `discover()` is a no-op import and the registry stays populated.
- `false` — `AmplitudeOnly` is missing a class docstring: this matches the existing `GlobalRescale`, which has none either.
- `low` — `n_sub == 0` / `n_ant == 0` mean-of-empty warnings: the loader enforces the `(4, packets, subcarriers)` shape, and real data has 234 subcarriers.
- `low` — Unwrap aliases slopes near ±π per subcarrier: measured slopes are −0.35 ± 0.17. The limit is already recorded in Implementation Notes.
- `low` — The circular-mean offset is unstable when antenna intercepts cancel: the spec mandates this design, and the relative phase is still kept. Changing it would mean editing the spec.
- `low` — Contiguous `k = 0..n_sub-1` ignores DC/edge tone gaps: a spec decision (the dataset carries no indices).
- `low` — Antennas get equal weight in the slope/offset fit; amplitude weighting was suggested: this is a design choice outside the spec's method.
- `low` — Denormal power overflows `1/sqrt(power)`: unrealistic at real amplitudes (~1e12).
- `low` — Nonsensical or duplicate step combinations (e.g. `lwm_rescale` + `ant2_rescale`, `phase_sanitize` after `amplitude_only`) are not rejected: these are user config choices, and validation would add complexity.
- `low` — No test for an all-zero packet through `phase_sanitize`: checked by hand (slope 0, offset `angle(n_ant)=0`, so it stays zero), and the data has no zero packets.
- `low` — Narrow slope-test range, no noise test, no input-immutability test, and the relative-phase test uses constant per-antenna offsets: the matrix rows are covered. `astype` copies, so the input is never mutated.

## Implementation Notes

- 2026-10-16: the phase tests use CSI whose phase varies smoothly across subcarriers. `np.unwrap` is only reliable when neighbouring subcarriers differ by less than π after the slope is added. The measured slope of about −0.35 rad per subcarrier is within that, but noisy real packets may be fitted imperfectly.
- With a single subcarrier the slope is 0 (the fit is undefined).
- The preset-order test calls the private `_expand_preprocessing` directly.
- `pytest` 468 passed; `ruff`, `ruff format --check` and `lint-imports` are clean. `ty` reports no errors in the touched files.
- Code review skipped at the user's request; nothing staged or committed.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run lint-imports` -- expected: clean
- `uv run ty check` -- expected: no errors in touched files (pre-existing errors in `runner/` and `tests/test_run.py`/`test_sweep.py` remain)
