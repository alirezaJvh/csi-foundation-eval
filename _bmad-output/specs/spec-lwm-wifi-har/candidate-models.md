# Candidate pretrained models (surveyed 2026-10-05)

## Comparison set for this phase (all frozen, plus a classifier head)

| Model | Kind | Weights | Size | Input used here | Notes |
|---|---|---|---|---|---|
| LWM v1.1 | Channel model, simulated data | [wi-lab/lwm-v1.1](https://huggingface.co/wi-lab/lwm-v1.1) | 2.4M | Per-packet antennas × subcarriers, pooled over the window | One candidate among peers; see `lwm-models.md` |
| WavesFM | Wireless model; ViT with masked-autoencoder pretraining | [ahmedaboulfo/wavesfm](https://huggingface.co/ahmedaboulfo/wavesfm) (`wavesfm-v1p0.pth`), Apache-2.0 | 38M | Time × subcarrier amplitude window resized to 224×224 | Built-in WiFi CSI activity task (802.11n, 114 subcarriers × 3 antennas, 500 Hz, 6 classes). Its WiFi pretraining set was small (840 samples), and training from scratch beat it in its own paper (98.3% vs 96.2%). [paper](https://arxiv.org/html/2504.14100) · [code](https://github.com/AhmedTarek62/wavesfm) · [site](https://wavesfm.waveslab.ai/) |
| Mantis-8M | General time-series model; contrastively pretrained ViT | [paris-noah/Mantis-8M](https://huggingface.co/paris-noah/Mantis-8M), Apache-2.0; `pip install mantis-tsfm` | 8M | Subcarrier amplitude series over the window, treated as multiple channels | Pretraining includes activity-recognition data; has adapters that reduce the number of channels. [paper](https://arxiv.org/html/2502.15637v2) |
| ImageNet ViT or ResNet | General vision transfer | torchvision / timm | 5–90M | Amplitude window as an image | Classic transfer-learning baseline |

## Watch list: highly relevant, no public weights

The user is contacting the authors and will check these manually. A model joins CAP-7 only once its weights are obtained.

| Model | Paper | Why relevant |
|---|---|---|
| AM-FM | [arXiv 2602.11200](https://arxiv.org/abs/2602.11200) | Closest match: 9.2M real CSI samples from 20 device types including Broadcom, supports 80 MHz, amplitude input 500×112, 5M parameters; tasks include activity recognition and occupancy |
| CSI-JEPA | [arXiv 2605.14171](https://arxiv.org/abs/2605.14171) | Amplitude windows (232 subcarriers × 500); evaluated frozen with an MLP probe; up to 98% fewer labels needed |
| Scale What Counts, Mask What Matters | [arXiv 2511.18792](https://arxiv.org/abs/2511.18792) | Masked-autoencoder pretraining on 1.3M samples from 14 real CSI datasets (20–160 MHz); finds data scale matters more than model size |
| WiLLM | [arXiv 2607.09727](https://arxiv.org/abs/2607.09727) · [repo](https://github.com/cjychenjiayi/WiLLM) | Cross-device CSI model; the repo releases data loaders only |
| WirelessGPT | [arXiv 2502.06877](https://arxiv.org/pdf/2502.06877) | Multi-task channel and sensing model that shows WiFi activity recognition; no weights found |

## Not used in this phase

| Model | Reason |
|---|---|
| WirelessSenseLLM ([arXiv 2605.14070](https://arxiv.org/html/2605.14070)) | Fine-tunes a 7B language model and needs video recorded in sync with the CSI; can't run on this CPU |
| Wi-Chat ([arXiv 2502.12421](https://arxiv.org/pdf/2502.12421)) | Prompt-based language model; doesn't fit the frozen-embedding comparison |
| MOMENT ([arXiv 2402.03885](https://arxiv.org/pdf/2402.03885)) | 385M parameters, slow on CPU; Mantis beats it on classification |
| LWM-Spectro | Needs I/Q signals, not CSI |

## Further reading

[Wireless foundation models survey](https://arxiv.org/html/2609.04707) · [LAM4PHY paper list](https://github.com/AI4Wireless/LAM4PHY_6G/) · [CSI-Bench](https://proceedings.neurips.cc/paper_files/paper/2025/file/f7c68372b3d39e8d7a093eb2edaaad87-Paper-Datasets_and_Benchmarks_Track.pdf) · [SSLCSI code](https://github.com/JJJinx/SSLCSI)
