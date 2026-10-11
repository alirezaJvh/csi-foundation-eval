# Bridging RF Signal Processing and Machine Learning in Wireless Sensing

**Document Scope:** Methodological Guide for Master's Thesis Research  
**Focus:** Handcrafted Feature Engineering vs. End-to-End Latent Representations in CSI-Based Human Sensing  
**Target Reader:** AI/ML Master's Researcher transitioning into RF Physical Layer & ISAC  

---

## 1. The Core Paradigm Divide: Feature Engineering vs. Representation Learning

In Channel State Information (CSI) sensing, the machine learning workflow fundamentally bifurcates into two opposing philosophies: **Classical Handcrafted Feature Engineering** and **End-to-End Representation Learning (Deep Learning / Foundation Models)**.

Understanding this conceptual divide is essential to analyzing why models succeed or fail when moving from in-room validation to cross-environment generalization.

```
═════════════════════════════════════════════════════════════════════════════════════
PARADIGM A: CLASSICAL SIGNAL PROCESSING + MACHINE LEARNING
═════════════════════════════════════════════════════════════════════════════════════
Raw CSI Matrix             Sanitization & Filters       Handcrafted Extraction        Classifier
[Subcarriers x Time x Ant] ──► [Unwrap, Detrend, Diff] ──► [Collapse to 6-12 Scalars] ──► [Random Forest / SVM]
  (e.g., 242 x 40 x 4)         (Remove CFO / Clutter)       (Doppler energy, Flatness)      (Decision Boundary)
                                                            ▲
                                                            └── THE HUMAN BOTTLENECK:
                                                                An engineer decides what matters.

═════════════════════════════════════════════════════════════════════════════════════
PARADIGM B: DEEP LEARNING, WORLD MODELS & FOUNDATION MODELS (END-TO-END)
═════════════════════════════════════════════════════════════════════════════════════
Raw CSI Matrix             Minimal Filtering            Spatio-Temporal Backbone      Latent Space z
[Subcarriers x Time x Ant] ──► [Diff-CSI / DC-Nulling]  ──► [ConvNet / Vision Transformer] ─► [Compact Vector z in R^D]
  (e.g., 242 x 40 x 4)         (Global Scale Preserved)     (Trainable Filter Banks)        (D = 64 / 128)
                                                            ▲
                                                            └── DATA-DRIVEN INVARIANCE:
                                                                Filters learn kinematic wave invariants.
```

---

### A. Paradigm A: Classical Handcrafted Feature Engineering

In the classical workflow (such as the pipeline employed by De Sanctis et al., PIMRC 2025):
* **The Problem:** A single CSI window contains tens of thousands of complex numbers (e.g., $242\text{ subcarriers} \times 40\text{ packets} \times 4\text{ antennas} \approx 38{,}720\text{ numbers}$). Feeding this directly into classical classifiers (SVM, Random Forest, GBDT) causes immediate overfitting and the curse of dimensionality.
* **The Engineering Solution:** The researcher acts as a manual dimensionality reducer. Using domain-specific signal processing intuition (Fourier transforms, statistics, spatial antenna projection), the entire spatio-temporal volume is collapsed into a compact vector of handcrafted scalar metrics (e.g., $\mathbf{x} \in \mathbb{R}^6$).
* **Advantage:** Low computational footprint, high interpretability, and nearly $100\%$ accuracy when training and testing within the exact same room.
* **Fundamental Flaw:** Handcrafted scalars aggregate global energy and discard phase relationships across subcarriers and antennas. If the physical room changes (different wall distances, new multipath reflections), the scalar values shift drastically, causing out-of-distribution (OOD) accuracy to collapse.

---

### B. Paradigm B: End-to-End Latent Representation Learning (DNNs, SSL, LWM)

In modern representation learning:
* **The Philosophy:** Avoid discarding spatio-temporal phase coherency. Treat the CSI time-frequency grid as a structured, continuous tensor (similar to a multi-channel image or video).
* **The Mechanism:** 
  * Convolutional layers act as **parameterized, learnable 2D/3D filter banks** (learning optimal matched filtering and spatial beamforming directly from data).
  * Self-Attention layers in Vision Transformers (ViT) capture **long-range multipath correlations** across non-adjacent subcarriers and distant temporal frames.
  * Self-Supervised pre-training (e.g., RF-JEPA, RF-BYOL, SimCLR, or Large Wireless Models) trains the backbone to predict masked blocks or invariant views, forcing the latent representation $\mathbf{z} \in \mathbb{R}^D$ to preserve intrinsic human kinematics while discarding static wall clutter.
* **Advantage:** Preserves spatial and temporal structures necessary for Out-of-Distribution (OOD) robustness across rooms.
* **Trade-off:** Requires GPU computation, larger training datasets, and careful pre-training or regularization to avoid dimensional collapse.

---

## 2. Bridging Signal Processing Concepts to Machine Learning Primitives

For an ML researcher unfamiliar with radio physical layers, the following conceptual translation maps RF phenomena directly to familiar data structures:

| RF & Signal Processing Concept | Physical Definition in Wireless Channel | ML / Computer Vision Equivalent | Why It Matters for Human Activity (HAR) |
| :--- | :--- | :--- | :--- |
| **Subcarriers ($N_{\text{sub}}$)** | Discrete orthogonal frequency bins in an OFDM transmission (e.g., 242 subchannels). | **Image Height / Spatial Resolution** | Different frequencies experience distinct multipath fading. Moving scatterers modulate subcarriers non-uniformly. |
| **Packet Sequence / Time ($T$)** | Consecutive CSI measurements sampled at a given rate (e.g., $5\text{--}100\text{ Hz}$). | **Temporal Axis / Video Frames** | Captures the progression of human motion across observation windows. |
| **Antennas ($N_{\text{ant}}$)** | Physical reception elements spaced by distance $d \approx \frac{\lambda}{2}$. | **Color Channels / Multi-View Feeds** | Inter-antenna phase differences provide angular/spatial directionality (Angle of Arrival). |
| **Doppler Shift ($f_D$)** | Frequency shift $f_D = \frac{v}{\lambda}\cos\theta$ caused by relative body displacement. | **Optical Flow / Temporal Velocity Vectors** | Directly separates walking ($f_D > 10\text{ Hz}$) from static sitting/standing ($f_D \approx 0\text{ Hz}$). |
| **Static Clutter (DC Component)** | Constant reflection paths off static concrete walls, ceilings, and desks. | **Static Background Image / Spurious Bias** | Accounts for over $90\%$ of total RF energy; must be removed to isolate the human target. |
| **Carrier Frequency Offset (CFO)** | Hardware clock drift between Tx and Rx oscillators, causing random phase rotation. | **High-Frequency Stochastic Sensor Noise** | Scrambles raw CSI phase if not sanitized by linear detrending or relative phase processing. |

---

## 3. Taxonomy of Classical Handcrafted Features in RF Sensing

In the literature (e.g., PIMRC 2025, WiMANS, CSI-Bench, SHARP), classical features extracted from CSI are broadly categorized into four families:

```
                        [ HANDCRAFTED RF FEATURES ]
                                     │
    ┌─────────────────┬──────────────┴──────────────┬─────────────────┐
    ▼                 ▼                             ▼                 ▼
[ Temporal &      [ Spectral &                  [ Spatial &       [ Statistical &
 Statistical ]    Doppler Domain ]              Phase/Array ]     Information ]
 • Window Energy   • Mean Doppler Spectrum       • Virtual BF Opt. • CSI Flatness
 • Euclidean Dist  • Doppler Spread              • AoA Variance    • Spectral Entropy
 • Temp. Variance  • Spectrogram Integral        • Inter-Ant Phase • Kurtosis / Skew
```

---

### Family 1: Temporal Dynamics & Distance Metrics

* **Purpose:** Quantify the speed and magnitude of channel state transitions over consecutive packets.
* **Key Metrics:**
  * **Mean Consecutive Euclidean Distance:** Computes the step-by-step Euclidean distance between consecutive channel vectors $d(\mathbf{h}(t), \mathbf{h}(t-1))$.
    * *Behavior:* Near zero for an empty room or a motionless subject; high during walking or postural transitions.
  * **Temporal Variance:** The variance of the signal amplitude over a sliding window along the time dimension for each subcarrier.
  * **Peak-to-Average Power Ratio (PAPR) / Dynamic Range:** Measures the contrast between peak motion spikes and baseline noise floor.

---

### Family 2: Spectral & Doppler Domain Metrics

* **Purpose:** Isolate velocity distributions and dynamic limb kinematics by converting time-domain CSI variations into frequency shifts via the Fast Fourier Transform (FFT).
* **Key Metrics:**
  * **Mean Doppler Spectrum Integral:** Computes the FFT along time for each subcarrier, averages the resulting spectra across all subcarriers, and integrates the total spectral power outside the zero-Doppler bin.
    * *Behavior:* Directly correlates with kinetic energy. High values for `walking`, moderate values for limb gestures, near zero for `sitting` and `empty`.
  * **Doppler Spread / Spectral Centroid:** The center-of-mass frequency of the Doppler power spectrum. Reflects the dominant speed of the human subject.
  * **Spectral Entropy:** Measures the randomness of frequency distribution. Narrow Doppler peaks (coherent limb stride) yield low entropy; diffused noise yields high entropy.

---

### Family 3: Spatial & Antenna Array Metrics

* **Purpose:** Leverage multiple physical antennas to estimate spatial direction of arrival and isolate directional motion from ambient reflections.
* **Key Metrics:**
  * **Virtual Receive Beamforming Metric ($\theta^*$ Optima):** Applies complex phase steering weights $w_1 = 1, w_2 = e^{j\theta}$ across physical receive antennas and performs a virtual angular sweep $\theta \in [0, 2\pi)$.
    * *Behavior:* The angle $\theta^*$ that maximizes Doppler response or minimizes multipath distortion changes dynamically when a human moves across angular sectors.
  * **Inter-Antenna Phase Difference (IAPD):** The sanitized phase difference $\Delta\phi(t) = \angle h_1(t) - \angle h_2(t)$. Eliminates common-mode CFO errors introduced by shared local oscillators.
  * **Angle of Arrival (AoA) Variance:** Tracks the dispersion of propagation angles extracted via subspace algorithms (e.g., MUSIC / ESPRIT).

---

### Family 4: Channel Flatness & Statistical Distribution

* **Purpose:** Measure how multipath interference distorts the frequency spectrum across OFDM subcarriers.
* **Key Metrics:**
  * **CSI Frequency Flatness:** The ratio of the geometric mean to the arithmetic mean of subcarrier power (or the cross-subcarrier amplitude variance).
    * *Behavior:* In free space with no clutter, the channel is flat. Heavy multipath scattering introduces frequency-selective fading notches, reducing flatness.
  * **Amplitude Skewness & Kurtosis:** Higher-order statistical moments measuring the non-Gaussianity of CSI amplitude fluctuations induced by biological scatterers.

---

## 4. Why Classical Features Suffer Under Out-of-Distribution (OOD) Shifts

While handcrafted features achieve $>98\%$ accuracy when training and testing within the same physical room, they systematically fail when tested on unseen environments (e.g., training on Room 1 and testing on Room 2). 

The primary physical failure modes include:

1. **Multipath Entanglement:**  
   Features such as *CSI Flatness* or *Euclidean Distance* conflate environment dimensions with human actions. An empty large room with reflective glass walls can exhibit an identical flatness metric to a small furnished room occupied by a person.
2. **Loss of Phase Coherency:**  
   Aggregating an entire time-frequency matrix into a single scalar (e.g., mean Doppler energy) destroys the precise phase progression and temporal sequencing needed to distinguish subtle activities (e.g., `sitting` vs. `standing`).
3. **Arbitrary Thresholding:**  
   A decision boundary placed by a Random Forest on a scalar feature (e.g., `"Doppler Integral > 14.5"`) depends on the transmitter-receiver distance and wall reflection coefficients of the training room. When moved to an adjacent room with higher attenuation, the threshold fails.

---

## 5. Practical Guidance for Master's Thesis Implementation

To build a rigorous experimental comparison across datasets:

1. **Implement Both Pathways in Parallel:**
   * **Path 1 (Classical Baseline):** Extract the 6 key features (Doppler Integral, CSI Distance, CSI Flatness, and their beamforming phase variations) and evaluate using Random Forest / SVM.
   * **Path 2 (Latent Representation / SSL):** Feed the uncompressed, sanitized time-frequency tensor $[56 \times 56 \times 3]$ into modern encoders (RF-JEPA, RF-BYOL, SimCLR, and frozen LWM embeddings).
2. **Benchmark Both Under Strict OOD Protocols:**
   * Evaluate both paths on:
     * *In-Room Split (Random vs. Time-Purged)* to quantify data leakage.
     * *Strict OOD (Train on Room 1 $\rightarrow$ Test on Room 2)* to measure true physical generalization.
3. **Dataset Staging:**
   * **Stage 1 (Primary):** The Tor Vergata dataset (De Sanctis et al., 2-room Nexmon setup) for baseline and OOD validation.
   * **Stage 2 (Cross-Dataset Transfer):** Ingest WiMANS or CSI-Bench to evaluate zero-shot transfer capabilities of pre-trained wireless foundation models.