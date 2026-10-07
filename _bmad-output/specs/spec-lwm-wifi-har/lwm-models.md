# LWM model family

Papers are in `papers/LWM/`. All models come from wi-lab (ASU, Alkhateeb group): https://huggingface.co/wi-lab

| Model | Input | Pretraining data | Status here |
|---|---|---|---|
| LWM v1.0 (arXiv 2411.08872) | one channel matrix, real and imaginary parts as 1D patches (32×32 → 128 patches of 16) | DeepMIMO ray-traced, simulated | Superseded by v1.1 |
| **LWM v1.1** (https://huggingface.co/wi-lab/lwm-v1.1) | one complex channel N×SC, N·SC ≤ 8196, 4×4 2D patches, max sequence 512 | 1.05M samples, 140 DeepMIMO city scenarios | **In use (candidate)** |
| LWM-Temporal (arXiv 2603.10024) | channel sequences in the angle-delay-time domain, sparse spatio-temporal attention | Ray-traced mobility trajectories | Later candidate: fits the temporal nature of activities |
| LWM-Spectro (arXiv 2601.08780) | I/Q baseband spectrograms; mixture of experts over WiFi, LTE and 5G | 9.2M synthetic spectrograms | Probably not applicable: the own data is CSI, not I/Q |

## LWM v1.1 facts relevant here

- Architecture: 12 layers, d_model 128, 8 heads, about 2.4M parameters, 10 MB of weights (`models/model.pth`). Token length is 32 (a 4×4 patch × real and imaginary parts).
- `input_preprocess.patch_gen(manual_data=...)` accepts custom channels; `patch_maker` zero-pads to multiples of 4.
- Outputs a CLS embedding of shape (n, 128) and channel embeddings of shape (n, n_patches, 128).
- Released modes: no fine-tuning, partial fine-tuning (last 4 layers), full fine-tuning.
- Repo scripts: `downstream.py` for inference and fine-tuning, `main.py` for pretraining. Depends on PyTorch, transformers and DeepMIMOv3.
- Tasks shown by the authors: LoS/NLoS classification and sub-6 GHz to mmWave beam prediction. No real or measured WiFi CSI is used anywhere.

## Fit with the own data

- One packet is 4 antennas × 234 subcarriers = 936 complex values, under the 8196 limit. It becomes 59 tokens: 1 × 59 patches after padding to 4×236. The 2-antenna variant pads rows to 4, which gives the same token count.
- Domain gap: the model was trained on simulated, noise-free outdoor channels, while this data is real indoor CSI with hardware gain and phase offsets. See `preprocessing.md`.

## Compute (measured on this machine)

- CPU: Ryzen 7 6800-series, 8 cores / 16 threads, 11 GB RAM. GPU: integrated Radeon 680M with shared memory; no CUDA, and ROCm is not installed and not officially supported.
- An LWM-sized transformer on this CPU embeds the whole dataset (~63k packets) in about 2 minutes. Full fine-tuning costs about 9 minutes per epoch for 50k samples.
- Verdict: the CPU is enough for this phase. Use a cheap cloud GPU (e.g. Colab T4) only if fine-tuning sweeps become a bottleneck.
