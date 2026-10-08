---
title: 'Load the own recordings as labelled Recordings'
type: 'feature'
created: '2026-10-07'
status: 'done'
baseline_commit: 'cc62c25750c7b93b4732fea3582d0af0d8d8daed'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-1-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** No stage can consume data yet: there are no core data contracts and no dataset source, so recordings, labels, rooms and metadata have no single consistent view.

**Approach:** Add the frozen `Recording`/`RecordingMeta` contracts and a `DatasetSource` port to `core/`, a registered `tor_vergata` source that loads the 19 saved CSI files, and a reusable port contract test.

## Boundaries & Constraints

**Always:**
- `Recording` and `RecordingMeta` are frozen dataclasses (AD-4): `csi` complex64 `(antennas, packets, subcarriers)`, `antenna_ids`, `recording_id = "tor_vergata/<file stem>"`, `room`, `label`, `meta`. Only the source fills `meta`.
- Each source declares an ordered class list; `tor_vergata` is `[empty, sitting, standing, walking]`, labels are indices. Room is `r1` or `r2`.
- File-name parsing: `r<room>_<activity>[_<n>]` with activity in `empty`, `sit|sitting`, `standing`, `walk|walking`; anything else raises `DataError` naming the file.
- `meta`: `n_packets`, `source_sha256` (file hash), `downsample_factor` (r2_standing_2 = 5, r2_standing_1 = 3, every r1_sitting_* = 2, else `None`), `sample_rate_hz = None`.
- Registered via `@register("dataset", "tor_vergata")` with pydantic `Params` (data root, default `data/csi_proc_python/saved_csi`) and `VERSION = 1`. Recordings returned sorted by `recording_id`.
- The contract test is a reusable function any future dataset adapter calls: checks shapes, dtypes, id uniqueness, label validity, antenna_ids length.

**Never:**
- Do not modify `data/`; no resampling, dropping or reordering of packets; no splitter, `describe`, CLI or config code (stories 1.3, 1.4); no `git add`/commit, no code review run (user instruction).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Full load | the 19 files | 19 `Recording`s, csi `(4, n, 234)` complex64, antenna_ids `(0,1,2,3)` | N/A |
| Name variants | `r2_sit_1`, `r2_walk_2`, `r1_empty` | sitting/r2, walking/r2, empty/r1 | N/A |
| Downsample | `r2_standing_2`, `r1_sitting_3`, `r1_empty` | 5, 2, None | N/A |
| Bad name | `r3_dance_1.npy` | — | `DataError` naming file |
| Bad shape | array not 3-D complex with 4 antennas | — | `DataError` naming file |
| Missing root | nonexistent directory | — | `DataError` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/registry.py` -- `register(stage, name)` needs `Params` + int `VERSION`; reuse
- `src/csi_bench/core/errors.py` -- `DataError`; `core/hashing.py` -- add no new hash logic (file sha256 is a plain byte hash)
- `src/csi_bench/adapters/` -- empty package; new subpackage `datasets/` (import-linter: adapters import only core; independence over `adapters.*.*`)
- `src/csi_bench/runner/discover.py` -- walks `csi_bench.adapters`; tests clear the registry, so test must call `discover()` or import the module after reset
- `tests/test_discovery.py` -- pattern for registry reset fixture
- `data/csi_proc_python/saved_csi/*.npy` -- 19 files, `(4, n, 234)` complex64, ~452 MB total; leave unchanged

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/contracts.py` -- frozen `RecordingMeta`, `Recording` -- AD-4
- [x] `src/csi_bench/core/ports.py` -- `DatasetSource` Protocol: `classes: tuple[str, ...]`, `load() -> list[Recording]`
- [x] `src/csi_bench/adapters/datasets/{__init__,tor_vergata}.py` -- registered source with name parsing, downsample table, sha256 -- AC 1-3
- [x] `tests/contracts/dataset_source.py` (+ `__init__.py` if needed) -- reusable `check_dataset_source(source)` contract
- [x] `tests/test_tor_vergata.py` -- real-data load (skip if data absent), parsing/downsample/error cases on synthetic tmp files, runs the contract

**Acceptance Criteria:**
- Given the real folder, when loaded, then 19 recordings with correct labels, rooms, ids, meta and the contract passes.
- Given an unrecognised or malformed file, when loaded, then `DataError` names the file.
- Given `registry.reset()` then `discover()`, then `get("dataset", "tor_vergata")` resolves.
- `ruff check`, `ruff format --check`, `ty check`, `lint-imports`, `pytest` all pass.

## Implementation Notes

- Implemented directly (no subagent). Code review and git add/commit skipped per user instruction.
- `DatasetSource.classes` is a read-only property in the Protocol so plain class attributes satisfy `ty`.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run pytest` -- expected: all pass

### Review Findings

- [x] [Review][Patch] `Recording.csi` stays writable inside a frozen dataclass — a stage can mutate shared data in place; set `csi.setflags(write=False)` after load and test it [src/csi_bench/adapters/datasets/tor_vergata.py:66]
- [x] [Review][Patch] Corrupt/non-npy/object-array file raises raw `ValueError`/`OSError` outside the `DataError` contract, and each file is read twice (`np.load` + `read_bytes`); read bytes once, `np.load(BytesIO, allow_pickle=False)`, wrap errors naming the file [src/csi_bench/adapters/datasets/tor_vergata.py:66-74]
- [x] [Review][Patch] Root with no `*.npy` files returns `[]` silently; raise `DataError` and add an empty-directory test [src/csi_bench/adapters/datasets/tor_vergata.py:63-64]
- [x] [Review][Patch] `test_registered_via_discovery` resets the registry only at the end; a failing assert leaks registry state into later tests — use a fixture with teardown [tests/test_tor_vergata.py:86-93]
- [x] [Review][Patch] `test_real_dataset` pins only `r1_empty`=0 and `r2_standing_2`; assert sitting/standing/walking label indices and the `r1_sitting_*` and `r2_standing_1` downsample factors on the real files [tests/test_tor_vergata.py:25]
- [x] [Review][Defer] `Params.root` default is relative to CWD, so loading fails when run outside the repo root — deferred: spec mandates this default string; resolve against project root when the CLI/config stories (1.3/1.4) decide path handling

#### Rejected

- Duplicate recording ids / case variants (false): ids come from distinct filenames in one directory, and the regex only accepts lowercase stems.
- Unknown file name aborts whole load (false): spec requires `DataError` naming the file.
- r1 sitting rule looser than `r1_sitting_*` (false): `r1_sit_*` is the same recording class per spec aliases; matches intent.
- Heterogeneous subcarrier counts, zero packets, NaN/inf, directory named `*.npy` (low): no reachable case shown; fix adds guards.
- Contract test lacks label-type/prefix/determinism checks, `__post_init__` validation, `classes` ordering enforcement, known-digest check, `VERSION`/`Params` tests (low): all spec-listed checks are covered; fixes add complexity or speculative surface.
- `sys.modules` monkeypatch may spuriously pass (maybe-false, low): nothing shows it misbehaves.
- Real-data test hardcodes 19/234 (false): spec defines exactly this dataset.
