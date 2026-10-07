# CSI preprocessing — findings and variants

## Measured on `saved_csi` (2026-10-03)

| Observation | Value | Consequence |
|---|---|---|
| Amplitude scale | ~7×10¹² mean; range 10⁹–10¹³·⁸ | LWM was trained on channels of order 1 (DeepMIMO ×1e6), so rescaling is **mandatory** for LWM |
| Packet power jitter | 10–15% std even in an empty room (receiver gain control) | Per-packet normalization may remove noise that isn't caused by motion |
| Phase slope across subcarriers | Random per packet: mean −0.35, std 0.17 rad/subcarrier (timing offset) | Raw phase is mostly hardware artifact, and LWM consumes real and imaginary parts, so phase sanitization is likely needed |

## Variants to run (experiment variables, not fixed choices)

LWM input:
1. Global rescale only (divide by the dataset's mean amplitude).
2. Per-packet power normalization (unit average power per packet).
3. Option 1 or 2 plus phase sanitization (subtract the per-packet linear phase fit across subcarriers).
4. Amplitude-only, with phase set to 0. This matches what the classical pipeline effectively uses.

Paper method: `norm = 0` and `norm = 1`. In `processing.py`, `norm = 1` divides by the mean amplitude of the whole recording, test part included. Here it is the `recording_mean_norm` preprocessing step, fitted on each recording's train block; the whole-recording version is allowed only for the paper-reproduction reference rows (architecture AD-9, AD-16).

Windows: per-packet classification, plus mean-pooled LWM embeddings over several window lengths, including 40 to match the paper's W2.
