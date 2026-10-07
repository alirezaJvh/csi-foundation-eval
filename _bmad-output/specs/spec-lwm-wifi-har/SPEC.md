---
id: SPEC-lwm-wifi-har
companions:
  - datasets.md
  - lwm-models.md
  - preprocessing.md
  - candidate-models.md
  - ../../planning-artifacts/architecture/architecture-master-2026-10-05/ARCHITECTURE-SPINE.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Pretrained Deep Models vs Classical ML for WiFi CSI Human Sensing (MSc Thesis)

## Why

Opportunity plus research mandate. Pretrained deep models (wireless foundation models such as LWM and WavesFM, general time-series and vision models) promise strong results from small labeled datasets. The thesis tests whether they beat classical ML on real WiFi CSI human sensing, where labeled data is scarce, and which one fits best; no single model is assumed to be the right one. The researcher has a strong CS/AI background and a weak wireless background, so the work adapts existing models to a sensing task rather than designing wireless signal processing. The same harness serves the supervisor's classical-ML work (feature and classifier sweeps) on every dataset. This is research: model choice, features, preprocessing and pipeline structure are expected to change as results come in.

## Capabilities

- **CAP-1**
  - **intent:** Researcher can load the group's own CSI recordings as labeled samples (activity class, room, recording) with fixed, documented train/test splits.
  - **success:** All 19 recordings in `data/csi_proc_python/saved_csi/` load with class, room and recording labels. Per-class and per-room counts are reported, and the splits are reproducible from a split file.
- **CAP-2**
  - **intent:** Researcher can run pretrained LWM on the group's CSI format and obtain CLS and channel embeddings.
  - **success:** Every packet produces embeddings of the documented shape for both the 2-antenna and 4-antenna variants, and a sanity check confirms they are not constant or NaN.
- **CAP-3**
  - **intent:** Researcher can measure how well LWM embeddings perform at activity recognition as-is (released weights, no extra pretraining).
  - **success:** Accuracy, macro-F1 and a confusion matrix are reported for room 1, room 2 and both rooms mixed, for every preprocessing, antenna and window variant run.
- **CAP-4**
  - **intent:** Researcher and supervisor can run classical ML over a library of per-window signal features (the paper's beamforming features, amplitude statistics such as mean power, standard deviation and flatness), choose feature subsets manually or by automatic selection, and train SVM, KNN, Random Forest or logistic regression.
  - **success:** One sweep command runs every listed combination of window length, feature set or selector, and classifier, and reports the CAP-3 metrics on the same splits, plus the features selected in each run and how stable that choice is across rooms and seeds.
- **CAP-5**
  - **intent:** Researcher and supervisor can compare the pretrained deep models (CAP-3, CAP-7) and classical ML (CAP-4) side by side.
  - **success:** One results table lists every method per room setting on identical splits, each with the variant chosen by validation macro-F1 and its test score; an all-variants view covers the rest. The dataset paper's method, recomputed from `saved_csi`, appears twice: under the time-block split as a comparison row, and under a random split as a reference row next to its published numbers.
- **CAP-6**
  - **intent:** Researcher can repeat CAP-3 to CAP-5 on public datasets once the supervisor verifies them.
  - **success:** Each verified dataset gets a loader and a results table with no change to the evaluation code.
- **CAP-7**
  - **intent:** Researcher can run other pretrained models frozen on the same data, splits and classifier heads as LWM: WavesFM, Mantis-8M, and an ImageNet-pretrained vision model. Watch-list models join once their weights are obtained (`candidate-models.md`).
  - **success:** Each model reports the CAP-3 metrics for every room and setting, and adding a model needs only a new model adapter, not pipeline changes.

## Constraints

- LWM v1.1 takes a single complex channel matrix (N antennas × SC subcarriers, N·SC ≤ 8196) split into 4×4 patches. A packet here is 4 (or 2) antennas × 234 subcarriers; LWM's own `patch_maker` zero-pads both axes. Details are in `lwm-models.md`.
- The raw CSI must be rescaled before LWM: its amplitudes are around 10¹³, while LWM was trained on channels of order 1. Which rescaling to use is an experiment variable (see `preprocessing.md`).
- LWM was pretrained only on simulated outdoor DeepMIMO channels, while this data is real indoor 802.11ac CSI. The domain gap is part of the finding and is reported, not hidden.
- Every pretrained model is used as-is: frozen released weights with only a classifier head trained, and the same head family across models. No pretraining, fine-tuning or LoRA in this phase.
- Window-based models (WavesFM, Mantis, ImageNet) take time × subcarrier amplitude windows. LWM takes per-packet matrices whose embeddings are pooled over the same windows, so every model is scored on identical windows.
- Recordings are used at their saved sample rate, with no resampling. To guard against the sample rate standing in for the activity, report each recording's packet count. If the professor confirms the rates differ, rerun with the rates equalized as a check.
- **Primary split:** contiguous train, validation and test blocks within each recording, separated by a gap longer than any window, so no window spans two blocks. Hyperparameters and variants are chosen on validation; test is scored once. A random split is used only for the paper-reproduction reference row.
- All methods use identical splits, windowing and metrics.
- Feature selection uses train and validation only, never test. It is off by default for pretrained-model embeddings.
- Build order: the classical harness comes first (it exercises split, windows, classifiers, sweeps and report on cheap features, and the supervisor needs it now); LWM embeddings may start right after data loading.
- Variants run as separate experiments, not hard-coded choices: 2 vs 4 antennas, preprocessing variants, and window length (per-packet and several pooled lengths).
- Model, preprocessing, windowing, classifier head and dataset must be switchable per experiment without rewriting the pipeline, and every reported number must trace back to its exact configuration.
- Runs on the local CPU (Ryzen 7 6800-series, 11 GB RAM). No GPU dependency.
- Public datasets are used only after the supervisor verifies them.

## Non-goals

- Pretraining or fine-tuning any model (partial, full or LoRA); fine-tuning is the next phase.
- Large language model approaches (e.g. WirelessSenseLLM, Wi-Chat), and LWM-Spectro, which needs I/Q signals.
- People counting, deferred until a verified public dataset with count labels exists.
- Collecting new data or re-decoding raw pcaps, which are not available.
- Real-time or deployed sensing system.
- GPU or ROCm setup.

## Success signal

- The supervisor receives a reproducible table comparing the pretrained deep models (LWM, WavesFM, Mantis-8M, an ImageNet model) with the classical ML pipeline on the own dataset (room 1, room 2, mixed), using the time-block split, next to the paper's random-split reference numbers. Each row can be regenerated with one command from its saved configuration.

## Assumptions

- The own dataset's task is 4-class activity recognition (empty, sitting, standing, walking) in two rooms, per its paper and the file labels.
- Large language model approaches stay out of this phase (CPU only; the candidates need a 7B fine-tune). Reopen if you want an LLM row.
- Data cleaning is limited to the existing preprocessing variants until the supervisor specifies cleaning steps; each one becomes a preprocessing option.
- The split protocol and window strategy were chosen by Claude because you deferred them. Confirm both with your supervisor.
- The gap between split blocks is 64 packets (~13 s at 5 Hz), long enough for windows up to 50 packets and the paper method's ~55-packet span. Confirm with your supervisor.

## Open Questions

- Which data-cleaning steps (e.g. outlier or noisy-packet removal) does Paolo want? Not blocking.
- Will the authors of the watch-list models share their weights? You're contacting them and will check manually (`candidate-models.md`).
- Did the saved recordings end up at one common rate (~5 Hz) after downsampling? Ask the professor; the original pcap timestamps would settle it.
