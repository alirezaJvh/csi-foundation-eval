# Datasets

## Own dataset (Tor Vergata) — primary

Source: De Sanctis et al., "WiFi Sensing Through Digital Receive Beamforming and CSI", IEEE PIMRC 2025 (`papers/WiFi_Sensing_Through_Digital_Receive_Beamforming_and_CSI.pdf`).
Location: `data/csi_proc_python/`.

| Property | Value |
|---|---|
| Hardware | Asus RT-AC86U monitor router, Nexmon CSI tool, IEEE 802.11ac VHT, 80 MHz |
| Saved format | `saved_csi/<room>_<activity>_<k>.npy`, complex64, shape (4 antennas, n_packets, 234 subcarriers) |
| Subcarriers | 234 = 256 minus nulls/DC minus 8 pilots (the paper's "242" counts the pilots) |
| Rate | ~5 packets/s nominal; some files downsampled before saving (see below) |
| Classes | empty, sitting, standing, walking (several positions and paths share one label) |
| Raw pcaps | Not included; `csi_decoder.py` needs them plus the `nexcsi` library |

### Recordings (packets per antenna)

| Room | empty | sitting | standing | walking |
|---|---|---|---|---|
| r1 | 4962 (1 file) | 2947 / 3051 / 3020 | 4992 / 4848 / 5169 | 3345 / 2501 / 2462 / 2675 |
| r2 | 3427 / 2121 | 2657 / 2846 | 3549 / 3240 | 2770 / 2690 |

- `csi_saver.py` comments show that some files were downsampled: r2_standing_2 by 5, r2_standing_1 by 3, r1 sitting by 2.
- Room 1 "empty" has only one recording, so leave-one-recording-out is impossible for that class. The time-block split works for every class.

### Precomputed classical features

`saved_dataset/dataset_{r1,r2,r1r2}_w2_{10,30,40,50}.npy`: 6 beamforming features plus a label column (0 empty, 1 sitting, 2 standing, 3 walking). Example: r1r2_w2_40 has 62,227 rows. The features are computed from antennas 0 and 1 only.

### Professor scripts

| File | What it does | Use here |
|---|---|---|
| `csi_decoder.py` | Nexmon pcap → complex CSI; drops nulls and pilots; splits packets by antenna (`css` field) | Reference only (no pcaps) |
| `csi_saver.py` | Runs the decoder and saves `.npy` (downsampling lines are commented out) | Reference for the downsampling history |
| `plot_csi.py` | Live amplitude plot per antenna | Optional for exploration |
| `processing.py` | The paper's method: combines antennas 0 and 1 with 40 phase weights; per window computes Doppler integral (W=8), CSI distance (W=16) and flatness (W=1); keeps the best weight; second window W2 gives 6 features | Not CAP-4 (that waits for Paolo). Recomputed as the reference method under both splits (architecture AD-16) |
| `processing_single_method.py` | Exploration and plotting variant for one method on 4 room-1 files | Not needed |
| `classification.py` | StandardScaler + random forest, **random 50/50 split**, optional MRMR feature selection, confusion-matrix heatmap | Reproduce the paper's numbers |

Known issues:
- **Leakage:** windows slide by 1 packet, so with W2 = 40 neighbouring windows share 39 of 40 inputs. A random split puts near-copies in train and test, so the paper's 97–99% is likely optimistic.
- **CSI distance:** the distance loop compares each packet with all packets from index 1 onward, not only consecutive pairs as in the paper's Eq. 10.

Paper's reference results (random split): room 1 99.41%, room 2 99.59%, mixed rooms 97.37% (4 features, W2 = 40).

## Public datasets — verified by supervisor (2026-10-08)

The supervisor accepted all candidates and split them into two use cases. Activity recognition comes first and is in scope for CAP-6. People counting comes later and is out of scope.

| Dataset | Use case | Status | Links | Note |
|---|---|---|---|---|
| Padova 80 MHz CSI (Meneghello et al.) | activity recognition | **in scope, first candidate** | data: https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels · paper: https://arxiv.org/abs/2305.03170 · code: https://github.com/francescamen/SHARP | Same hardware family as the own data (Asus RT-AC86U, Nexmon, 802.11ac 80 MHz), so probably the easiest import. Likely recorded across several environments and days, a natural second generalisation test. Story 6.1 confirms the format, classes and splits. |
| CSI-Bench (NeurIPS 2025 Datasets & Benchmarks) | activity recognition (plus other tasks) | **in scope, second** | paper: https://proceedings.neurips.cc/paper_files/paper/2025/file/f7c68372b3d39e8d7a093eb2edaaad87-Paper-Datasets_and_Benchmarks_Track.pdf | Large and multi-device. Its paper reports Transformer baselines, which are context for the spec's "Next phase". Story 6.2 confirms the format (possibly amplitude-only), classes and splits. |
| WiMANS | people counting (multi-user) | later use case, out of scope | https://github.com/huangshk/WiMANS · paper: https://arxiv.org/abs/2402.09430 | Kept for the people-counting phase. |
| EHUNAM | — | not characterised | https://doi.org/10.6084/m9.figshare.28541225.v1 | Unchanged. |

Each dataset declares its own class list (AD-4). No label mapping across datasets is done in this phase; zero-shot transfer across datasets belongs to the spec's "Next phase (conditional)".
