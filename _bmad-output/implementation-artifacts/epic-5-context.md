# Epic 5 Context: LWM evaluation

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Let the researcher embed every packet with the frozen LWM v1.1 foundation model and score those embeddings with the shared windows, classifiers, metrics and confidence intervals, so LWM gets rows in the same comparison table as the classical baselines. The epic covers both the within-room (`time_block`) and cross-room (`room_holdout`) protocols. Experiment variables are 2 or 4 antennas, the preprocessing preset, the embedding type, the pooler and the window length. Within-room classical scores already sit near the ceiling (about 98.5–99.9% test accuracy), so the cross-room section with intervals is where LWM can show a real difference. LWM was pretrained on simulated, noise-free outdoor channels of order-1 magnitude. This data is real indoor CSI with amplitudes around 1e12 and hardware gain and phase offsets, so measuring and documenting that domain gap is part of the result.

## Stories

- Story 5.1: Embed every packet with frozen LWM v1.1
- Story 5.2: LWM preprocessing variants
- Story 5.3: Pool and score LWM in the comparison

## Requirements & Constraints

- **Frozen model.** No fine-tuning in this phase. Run in `eval()` under `torch.inference_mode()`. A test proves that parameter checksums are unchanged after extraction.
- **Per-packet output.** The extractor returns `PacketFeatures` of shape `(n_packets, 128)` for every packet, for both the 4-antenna and 2-antenna (`[0, 1]`) inputs, with `embedding: cls | channel_mean`. Outputs must pass the runner's finite and non-constant check. Otherwise it raises `ExtractorError`.
- **Domain-gap reporting.** The adapter reports input amplitude statistics alongside the embeddings, and warns if the median magnitude is not of order 1. These statistics go into run provenance. Recordings are used at their saved rate, with no resampling.
- **Preprocessing variants** (experiment variables, not fixed choices). There are 4 presets: (1) global rescale, (2) per-packet power normalisation to unit average power, (3) either of those followed by phase sanitisation, which subtracts each packet's linear phase fit across subcarriers, and (4) amplitude-only, with phase set to 0. Rescaling is mandatory for LWM. Phase sanitisation must remove an added linear slope on synthetic CSI to within tolerance.
- **Windows.** `window_len ∈ {1, 10, 30, 40, 50}` (40 = the paper's W2). `window_len: 1` is valid for LWM (per-packet) and rejected for per-window extractors. Changing the window length must never re-run LWM.
- `feature_selection` defaults to `null` for LWM embeddings.
- **No leakage.** Same splits, windows and metrics as every other method. Fitted preprocessing state comes from the selected train blocks only. Test is scored once.
- **Compute.** Embedding the full dataset (about 63k packets) on CPU is expected to take about 2 minutes, and the time is logged. Work in batches, because RAM is 11 GB.
- **Success.** `experiments/lwm.yaml` sweeps antennas × preprocessing preset × embedding × pooler × window length × classifier × rooms, under `time_block` and both `room_holdout` plans. `report` shows LWM rows in the within-room and cross-room sections, with confidence intervals.

## Technical Decisions

- **Adapter layout.** The adapter lives in `adapters/extractors/lwm/` (`adapter.py` plus `_vendor/`) with registry name `lwm`. It declares `granularity: per_packet`, plus a `requires` entry for phase and antenna count, which the runner checks at config validation. Only this adapter imports its `_vendor/`.
- **Vendoring.** Copy only `lwm_model.py` and a `patch_maker` excerpt, with no training, data or DeepMIMO code. `_vendor/SOURCE.md` records the repo URL, the full sha `1b68512dbe97de56841301206eacb780fcd76011`, every local edit and the constructor arguments. `_vendor/` is excluded from ruff and ty, and its hash feeds `run_id` and the cache key.
- **Weights.** Fetch from HF `wi-lab/lwm-v1.1` at the pinned sha with `huggingface_hub` into `.cache/weights/`. Load with `map_location="cpu"`, strip the `module.` prefix in the adapter, and use strict key matching. A strict-load test asserts that no keys are missing and none are unexpected.
- **Never call `make_sample`.** Vendored code must not touch global RNGs. The runner owns device, threads and determinism flags.
- **Model facts.** 12 layers, d_model 128, about 2.4M parameters, 4×4 complex patches whose tokens carry real and imaginary parts. One packet (4 antennas × 234 subcarriers) is zero-padded to 4×236, which gives 59 patch tokens plus CLS. The 2-antenna input is padded to 4 rows by `patch_maker`, with the same token count. The 4×4 patching stays inside the extractor and is not a preprocessing step.
- **Preprocessing steps** are `PreprocessStep` adapters (`per_packet_power`, `phase_sanitize`, `amplitude_only`) with `fit(train, plan) → state` and a pure `transform`. Steps work per packet or within one split block. Presets go in `config/presets.py`, and the chain hash covers step order. Each step must pass the `PreprocessStep` contract test.
- **Pooling.** The `mean` and `mean_std` poolers have no trainable weights and reduce each window's packet rows to one vector. Per-packet embeddings are cached once per recording, and the cache key has no `window_len` or `stride`. Pooling happens per window after the cache read. Only the runner touches the cache, and writes are atomic.
- **Heads and report.** LWM uses the shared head family (`logreg`, `svm`, `knn`, `random_forest`, `mlp`), with hyperparameters chosen on val. Report cells show the variant with the best validation macro-F1, with its test score and bootstrap interval.

## Cross-Story Dependencies

- 5.1 needs only Epic 1 (dataset, split). It uses the existing `antennas` and `global_rescale` steps from Epic 2.
- 5.2 is independent of 5.1, but its presets are the preprocessing axis of the 5.3 sweep.
- 5.3 depends on 5.1 and 5.2, and on Epic 2 (windowing, heads, cache, sweep, report). It also depends on Epic 4 (readable run layout, per-`(dataset, splitter)` report sections, `room_holdout` plans under `splits/tor_vergata/`, and confidence intervals).
- Downstream: Story 7.4 compares LWM against WavesFM, Mantis and ImageNet in every `(dataset, protocol)` section. Epic 6 datasets reuse the same LWM configs with only a `dataset:` change.
