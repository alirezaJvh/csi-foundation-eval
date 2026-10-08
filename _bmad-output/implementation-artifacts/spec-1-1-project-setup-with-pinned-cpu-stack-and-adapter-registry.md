---
title: 'Project setup with the pinned CPU stack and an adapter registry'
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

**Problem:** The repo has no Python project, so no component can be added, linted, tested, or kept inside the architecture rules (AD-2, AD-3).

**Approach:** Create the uv project `csi_bench` with the Stack-table pins on CPU-only torch, ruff/ty/import-linter/pytest config, and `core/` with errors, hashing and the self-registering adapter registry plus runner discovery.

## Boundaries & Constraints

**Always:**
- Python 3.12; pins exactly per the architecture Stack table; torch/torchvision from the PyTorch CPU index via `tool.uv.sources`; `pandas==3.0.6` via `override-dependencies`; `uv.lock` committed (not git-ignored).
- `_vendor/` folders excluded from ruff and ty.
- Hashes: SHA-256 of canonical JSON (sorted keys, no whitespace), first 12 hex chars.
- Registry: `@register("<stage>", "<name>")`; adapter declares pydantic `Params` and int `VERSION`; lookup by `(stage, name)`; duplicate raises `ConfigError`. Discovery uses `pkgutil.walk_packages` over `csi_bench.adapters`.
- import-linter contracts encode AD-2: core imports nothing internal; adapters import only core; adapters never import each other; only runner imports adapters.
- `.cache/` and `runs/` git-ignored; stdlib logging, no `print` in library code.

**Never:**
- Do not modify `data/csi_proc_python/`; no `nvidia-*` packages; no contracts, ports, splitters, datasets or CLI beyond what is listed (later stories).
- No `git add`/commit, no code review run (user instruction).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Register + lookup | adapter decorated `("splitter","x")`, discovery run | `get("splitter","x")` returns it | N/A |
| Duplicate | same `(stage,name)` registered twice | — | `ConfigError` |
| Unknown lookup | `(stage,name)` not registered | — | `ConfigError` |
| Hash | dicts equal up to key order | identical 12-char hex | N/A |
| Layering break | test module importing across layers | — | `lint-imports` fails |

</frozen-after-approval>

## Code Map

- `_bmad-output/planning-artifacts/architecture/architecture-master-2026-10-05/ARCHITECTURE-SPINE.md` -- Stack table, AD-2, AD-3, conventions, Structural Seed
- `.gitignore` -- already ignores `runs/`, data `.npy`; add `.cache/`; must not ignore `uv.lock`
- `data/csi_proc_python/` -- professor scripts and data, leave unchanged
- No existing Python package or pyproject.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml` -- uv project, src layout, pins, CPU index sources, pandas override, ruff/ty/import-linter/pytest config, dev group -- AC 1, 2
- [x] `.gitignore` -- add `.cache/` -- AC last
- [x] `src/csi_bench/{__init__,core/__init__,adapters/__init__}.py` -- package skeleton (plus empty `config/`, `runner/`, `results_api/` packages) -- structural seed
- [x] `src/csi_bench/core/errors.py` -- `ConfigError`, `DataError`, `ExtractorError`
- [x] `src/csi_bench/core/hashing.py` -- canonical-JSON SHA-256, 12-hex helper
- [x] `src/csi_bench/core/registry.py` -- `register`, lookup, duplicate/unknown -> ConfigError, reset helper for tests
- [x] `src/csi_bench/runner/discover.py` -- import `csi_bench.adapters` via `pkgutil.walk_packages`
- [x] `tests/` -- unit tests for errors, hashing, registry, discovery (with a fixture adapter), and a layering test that a violating module makes `lint-imports` fail

**Acceptance Criteria:**
- Given a fresh clone, when `uv sync` runs, then all Stack pins install, torch/torchvision are `+cpu`, no `nvidia-*` is installed, pandas is 3.0.6, and `uv.lock` is tracked.
- Given the project, when `ruff check`, `ruff format --check`, `ty check`, `pytest` and `lint-imports` run, then all pass.
- Given an adapter decorated with `@register`, when discovery imports `csi_bench.adapters`, then it is found by `(stage, name)` without being named elsewhere.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv sync` -- expected: success, no nvidia packages (`uv pip list | grep -i nvidia` empty)
- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest && uv run lint-imports` -- expected: all pass

### Review Findings

- [x] [Review][Patch] Independence contract breaks lint-imports once stage packages have children — `modules = ["csi_bench.adapters.*", "csi_bench.adapters.*.*"]` mixes parent and child; verified in a scratch copy with `adapters/splitters/x.py` that `lint-imports` errors with "Modules have shared descendants". Today it is vacuous (empty adapters/). Use leaf-level `csi_bench.adapters.*.*` only. [pyproject.toml:104-107]
- [x] [Review][Patch] Layering tests prove only one contract can fail — `test_violating_module_fails` asserts only `returncode != 0` (any config error passes), covers only core→adapters (hit by two contracts), leaves an unused `adapters/a.py`, and copies `__pycache__`. Parametrize over adapter→adapter, adapter→config/runner, config→adapters; assert the broken contract's name in stdout. [tests/test_layering.py:26-33]
- [x] [Review][Patch] `Params` not checked as a pydantic model — `register` only does `hasattr(cls, "Params")`; `Params = 5` registers, contradicting the spec constraint. Require `isinstance(Params, type) and issubclass(Params, BaseModel)`. [src/csi_bench/core/registry.py:19]
- [x] [Review][Patch] Registry tests miss `names()`, bool/str `VERSION`, non-pydantic `Params` — mutating `names()` (flip filter, drop sorted) or removing the bool guard leaves every test green. [tests/test_core.py]
- [x] [Review][Patch] `canonical_json` emits non-standard `NaN`/`Infinity` — add `allow_nan=False` so such configs fail loudly instead of hashing non-canonical JSON. [src/csi_bench/core/hashing.py:10]

#### Rejected
- `false` — AC 1 CPU stack unverified / nvidia pulled in: `uv.lock` pins `torch 2.14.1+cpu` and `torchvision 0.29.1+cpu` from the CPU index (darwin gets plain CPU wheels), and contains 0 `nvidia` entries. `uv.lock` is untracked only because the story is uncommitted; it is not git-ignored.
- `false` — `override-dependencies` duplicating the pandas pin: the spec mandates it.
- `false` — `ensure_ascii` unset: default is deterministic, so hashes are stable.
- `false` — 1 vs 1.0 / bare TypeError on non-JSON types: type-sensitive hashing is intended, and failing loudly is correct.
- `low` — `discover()` raw ImportError, no `onerror`, plain-module `__path__`, and no re-register after `reset()`: the traceback already names the module, and reset is test-only; guards add complexity for unlikely cases.
- `low` — `test_discovery_of_real_package_runs` asserts only a list: asserting `== []` would break when the first adapter lands.
- `low` — registry guards (non-class, empty stage/name, VERSION < 1), thread-safety, public `reset()`, no `CsiBenchError` base, stage-name typos: speculative, not shown reachable.
- `low` — `required-version = ">=0.11"` loose, `requires-python` strict, redundant pytest `pythonpath`, missing `py.typed`/CI/readme: negligible and outside the spec.
- `low` — `environments` platform restriction: the lock already resolves darwin and non-darwin.
- `defer` candidate dropped (`low`) — a `tests/test_env.py` CPU-torch smoke test: `uv.lock` already pins `+cpu`, so the benefit is small.
