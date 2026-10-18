---
title: 'Embed every packet with frozen LWM v1.1'
type: 'feature'
created: '2026-10-16'
status: 'done'
baseline_commit: '0b15d752a8ce1572174d977cfcabc29ce0e3f686'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-5-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** There is no way to get LWM v1.1 embeddings for our CSI, so LWM can't be evaluated. Epic 5 starts by building the frozen per-packet extractor.

**Approach:** Add a `lwm` extractor adapter (`granularity: per_packet`) with a vendored model excerpt pinned to HF `wi-lab/lwm-v1.1` @ `1b68512dbe97de56841301206eacb780fcd76011`, and a `PacketFeatures` contract. It turns each packet into 4×4 complex patches, prepends CLS, and returns a CLS or channel-mean embedding `(n_packets, 128)` plus input amplitude statistics. Pooling, caching and scoring belong to 5.3.

## Boundaries & Constraints

**Always:** `_vendor/` holds `lwm_model.py` copied byte-for-byte, a `patch_maker.py` excerpt (only `patch_maker` plus `import numpy as np`), an empty `__init__.py`, and `SOURCE.md` (repo URL, full sha, local edits, constructor args `lwm()` defaults: element_length 32, d_model 128, n_layers 12, max_len 513, n_heads 8, dropout 0.1). Nothing in it imports DeepMIMO. Weights come from `models/model.pth` at the pinned sha via `huggingface_hub` with `cache_dir=".cache/weights"`. They load with `torch.load(map_location="cpu")`, `module.` stripped, and `load_state_dict(strict=True)`. CLS token = `0.2 * ones(32)` (the repo's tokenizer constant), re-implemented in the adapter. The model is in `eval()` and every forward runs under `torch.inference_mode()`, in batches (param `batch_size`, default 512, `affects_result=False`). `channel_mean` = mean over output tokens `1:`. Input: complex64 `(antennas, packets, subcarriers)` with 1–4 antennas; patch rows are padded to 4 by `patch_maker`. The adapter exposes `VENDOR_HASH` (sha256 of the `_vendor/` files) and `WEIGHTS_REVISION` (the sha). Input stats are `{median_abs, mean_abs, p5_abs, p95_abs}`. If `median_abs` is outside [0.1, 10], log a warning naming the value.

**Never:** Call `make_sample`, `from_pretrained`, `tokenizer` or `patch_gen`. Touch global RNGs. Fine-tune. Add a pooler, cache path, preprocessing steps or experiment yaml (5.2/5.3). Change any existing `run_id`. Run `git add`/commit.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| 4 antennas | `(4, n, 234)` order-1 CSI | `PacketFeatures` `(n, 128)` float32, finite, non-constant, names `lwm.cls.0..127` | — |
| 2 antennas | `(2, n, 234)` | same shape (59 tokens + CLS) | — |
| channel_mean | `embedding: channel_mean` | `(n, 128)`, names `lwm.channel_mean.*`, differs from cls | — |
| Raw scale | median abs ~1e12 | embeddings returned, stats report the median | warning logged |
| Bad input | not 3-D, 0 packets, >4 antennas, or not complex | — | `ExtractorError` |
| Config use | experiment with `extractors: [{name: lwm}]` | rejected until 5.3 adds pooling | `ConfigError` naming `extractors.<i>` |

</frozen-after-approval>

## Code Map

- `src/csi_bench/core/contracts.py` -- add frozen `PacketFeatures(values: float32 (n_packets, d), feature_names, input_stats: dict[str, float])`, row i = packet i.
- `src/csi_bench/core/ports.py:26` -- add a comment to `FeatureExtractor`: a `per_packet` extractor implements `extract_packets(csi) -> PacketFeatures` in place of `extract`.
- `src/csi_bench/adapters/extractors/amplitude_stats.py` -- adapter shape to copy (`@register`, `Params` with `extra="forbid"`, `VERSION`, `requires`, `min_window_len`). LWM: `requires = {"phase": True, "n_antennas": "any"}`, `min_window_len = 1`, `Params.embedding: Literal["cls","channel_mean"] = "cls"`.
- `src/csi_bench/adapters/extractors/lwm/{__init__.py,adapter.py,_vendor/}` (new). Load the model lazily on the first `extract_packets` call and keep it on the instance. `discover()` walks packages, so `_vendor` gets imported too (torch only, which is fine).
- Upstream source: the pinned `lwm_model.py`, `input_preprocess.py:259` `patch_maker(original_ch, patch_rows, patch_cols)`, which expects `(n, 1, rows, cols)`; call it as `patch_maker(x[:, None], 4, 4)` with x = `csi.transpose(1, 0, 2)`. Note that `patch_gen(manual_data)` calls it without the row/col args (an upstream bug, so don't use it).
- `src/csi_bench/config/experiment.py:185` `validate_registry` -- reject `granularity == "per_packet"` extractors with a `ConfigError` (`extractors.<i>.name: lwm is per_packet; pooling arrives in Story 5.3`).
- `src/csi_bench/config/experiment.py:240` run_id `extractors` body -- add `vendor_hash`/`weights_revision` keys **only when the class defines them**, so existing run_ids stay the same (the cache key at :303 already includes them).
- `tests/contracts/extractor.py` -- add `check_packet_extractor(extractor, csi)`: shape `(n_packets, d)`, float32, finite, no constant column, unique dotted names, one-packet calls match batch rows within `atol=1e-5` (batch matmul is not bit-exact).
- `tests/test_paper_fixture.py:95` -- pattern for the `slow` full-data test (skip if data is missing). `TorVergata().load()` reads `data/csi_proc_python/saved_csi`. Rescale as `global_rescale` does (fit on all recordings is fine for a timing check).
- `pyproject.toml` already excludes `**/_vendor` from ruff and ty, and already pins `huggingface-hub` and `torch`. `.cache/` is gitignored.

## Tasks & Acceptance

**Execution:**
- [x] `src/csi_bench/core/contracts.py`, `core/ports.py` -- `PacketFeatures` + per-packet port note -- contract for 5.3
- [x] `src/csi_bench/adapters/extractors/lwm/_vendor/*` -- vendored excerpt + `SOURCE.md` -- pinned, auditable model code
- [x] `src/csi_bench/adapters/extractors/lwm/adapter.py` (+ empty `__init__.py`) -- `lwm` adapter as specified -- the story's extractor
- [x] `src/csi_bench/config/experiment.py` -- per_packet guard + conditional vendor/weights in run_id -- safe until 5.3, provenance-ready
- [x] `tests/contracts/extractor.py`, `tests/test_lwm.py` (new) -- vendor-folder contents/no DeepMIMO/SOURCE.md sha; strict load (no missing/unexpected keys after stripping `module.`); 4 and 2 antennas × both embeddings pass `check_packet_extractor`; parameter checksum unchanged and `model.training is False` after extraction; global `np.random`/`torch` RNG state unchanged; amplitude warning (caplog) on raw scale and none on order 1; bad inputs; config guard; existing run_id unchanged; `slow` full-dataset embedding that logs elapsed seconds -- coverage

**Acceptance Criteria:**
- Given the weights are not cached, when the adapter first embeds, then `models/model.pth` is fetched at the pinned sha into `.cache/weights/` and later runs reuse it.
- Given the real dataset, when `uv run pytest -m slow tests/test_lwm.py` runs, then every packet of every recording gets a `(n_packets, 128)` embedding on CPU, and the elapsed time and amplitude stats are logged.
- Given the whole suite, when `pytest`, `ruff` and `ty` run, then they pass and `lint-imports` contracts hold.

### Review Findings

- [x] [Review][Decision] Default test suite needs network access and the 2.4M-param weights — The module `model` fixture and `test_global_rngs_are_untouched` call `hf_hub_download` and are not marked, so plain `pytest` on a fresh/offline checkout (cold `.cache/weights/`) errors instead of skipping. Options: skip when the weights are not cached and the Hub is unreachable; add a `weights` marker excluded by `addopts`; or accept as documented in Implementation Notes. [tests/test_lwm.py:46] — resolved 2026-10-16: accepted as-is (Jj). The weights are ~9.5 MB, downloaded once into `.cache/weights/` and loaded offline afterwards; a skip would hide a broken cache.
- [x] [Review][Patch] `vendor_hash()` is only ever compared to itself — add a test that copies `_vendor/` to `tmp_path`, monkeypatches `VENDOR_DIR`, flips a byte in `patch_maker.py` and asserts the hash changes; dropping the bytes from the digest currently passes every test, which would let stale LWM cache entries survive vendored-code edits [tests/test_lwm.py:121]
- [x] [Review][Patch] Token construction and `channel_mean` are not pinned — no test checks that the CLS row is 0.2, that patch tokens equal `patch_maker(x[:, None], 4, 4)`, or that `channel_mean` excludes CLS; `CLS_VALUE = 0` or `output.mean(dim=1)` would pass every test. Capture the forward input and compare against an independent computation [src/csi_bench/adapters/extractors/lwm/adapter.py:139]
- [x] [Review][Patch] Bad-input paths beyond the matrix are untested — add 0 subcarriers, >2048 subcarriers and an `inf` entry to `test_bad_input_raises`, and assert the model was never loaded (the comment claims it) [tests/test_lwm.py:240]
- [x] [Review][Patch] Amplitude warning below 0.1 is untested — add a `_csi(scale=1e-3)` caplog case; changing the check to `median > high` currently passes [tests/test_lwm.py:207]
- [x] [Review][Patch] `model` fixture loads weights from the process cwd but tests assert `ROOT/.cache/weights` — `chdir(ROOT)` in the fixture so pytest from another directory neither re-downloads nor fails `test_weights_are_cached_at_the_pinned_revision` / `test_strict_load_...` [tests/test_lwm.py:46]
- [x] [Review][Patch] Slow full-dataset test under-asserts — `caplog` is unused; assert no amplitude warning fires after `global_rescale` and that `total` equals the summed `n_packets` [tests/test_lwm.py:274]
- [x] [Review][Defer] AC3 "`ty` passes" is not met as written [src/csi_bench/runner/run.py] — deferred: pre-existing ty errors in `runner/intervals.py`, `runner/run.py`, `tests/test_run.py`, `tests/test_sweep.py`; every file this story touches is clean
- [x] [Review][Defer] `requires = {"n_antennas": "any"}` while `_validate` rejects >4 antennas [src/csi_bench/adapters/extractors/lwm/adapter.py:115] — deferred: spec-mandated and every current dataset has ≤4 antennas; revisit when an Epic 6 dataset with more antennas arrives (config validation would pass and extraction would fail)
- [x] [Review][Defer] Vendored LWM code has no licence/attribution record [src/csi_bench/adapters/extractors/lwm/_vendor/SOURCE.md] — deferred (unverified, would be medium): check the licence on the `wi-lab/lwm-v1.1` model card at the pinned sha and record it in `SOURCE.md`

**Rejected:**
- false — `torch.load` without `weights_only=True` unpickles arbitrary code: torch is pinned at 2.14.1, where `weights_only=True` is the default.
- false — model forward output misread: with no `masked_pos`, the vendored `lwm.forward` returns `(output, attention_maps)`, which matches the adapter's unpacking.
- false — concurrent lazy load needs a lock: nothing calls `extract_packets` concurrently on one instance.
- false — `resolved_body` adds null provenance via `hasattr`: no extractor defines `VENDOR_HASH`/`WEIGHTS_REVISION` as `None`; existing run_ids are asserted unchanged.
- false — `check_packet_extractor` fails on duplicate packets: that is a precondition on test input, and every caller passes random CSI.
- false — batch-size test never exercises multi-batch: `batch_size=3` on 7 packets runs as 3+3+1.
- false — diff was missing files (Blind Hunter, first pass): a reviewer-setup error, corrected before the final reports.
- low — `WEIGHTS_CACHE` is cwd-relative in the adapter: the spec mandates `cache_dir=".cache/weights"`, matching the `.cache/features` convention in `runner/run.py`; the test-side effect is the fixture patch above.
- low — raw `huggingface_hub`/torch exceptions escape on a failed download: the error is already self-explanatory, and wrapping it adds a guard.
- low — `VENDOR_HASH` is computed at import and would crash `discover()` if a vendor file were missing: hatchling ships every file under `src/csi_bench`, and the test pins the folder contents.
- low — finite complex128 above float32 range overflows on the cast, and `np.abs` overflows near float32 max: CSI never comes near 1e38.
- low — no sha256 check on `model.pth`: the HF revision sha pins the blob, and hub downloads are etag-verified.
- low — `batch_size` has no upper bound: the user sets it, and memory use scales visibly.
- low — `SOURCE.md` is part of the vendor hash, so doc edits change run ids: the spec defines the hash over the `_vendor/` files.
- low — `channel_mean` is a misnomer, and zero-padded antenna rows go into 1–2-antenna inputs: the name and the padding behaviour are spec/epic decisions.
- low — `batch_size` is marked `affects_result=False` but differs at 1e-5: spec-mandated, and within float tolerance.
- low — the `ConfigError` text mentions "Story 5.3": the message wording is spec-mandated and temporary until 5.3.
- low — per-packet extractor has no typed Protocol: the spec chose a port comment; the next story can formalise it when the runner consumes it.
- low — the amplitude warning repeats per recording: per-call warnings carry per-recording medians, which is the intended provenance.
- low — unknown `granularity` values pass validation: this predates the story.
- low — no timing budget in the slow test, and the "about 2 min" in Verification is stale: AC2 only requires the time to be logged; updating the Verification text would edit the spec.
- low — RNG test runs on a warm cache only: a download performs no tensor initialisation, and construction on the meta device draws no RNG.

## Implementation Notes

- 2026-10-16: full-dataset embedding (63,272 packets, 19 recordings, `global_rescale` fitted on all) took 203–206 s on CPU, not the ~2 min estimate; after rescale the per-recording median |CSI| is about 0.78–1.11, so no amplitude warning fires.
- The model is built on the `meta` device and filled with `load_state_dict(strict=True, assign=True)`, so constructing it never draws from the global torch RNG.
- Extra input checks beyond the matrix: non-finite CSI and subcarrier counts giving more than 512 patches raise `ExtractorError`. Embeddings are also checked for non-finite values.
- Tests that load the model need network on first use (weights cached afterwards in `.cache/weights/`).
- `uv run ty check` still reports errors in `runner/intervals.py`, `runner/run.py`, `tests/test_run.py` and `tests/test_sweep.py`. Those errors predate this story; every file it touches is clean.
- Code review skipped at the user's request; nothing staged or committed.

## Spec Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run pytest -m slow tests/test_lwm.py -s --log-cli-level=INFO` -- expected: pass, elapsed time logged (about 2 min)
- `uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check` -- expected: clean
