"""Paper beamforming feature family (per-window, AD-8): port of `processing.py`."""

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.errors import ExtractorError
from csi_bench.core.registry import register

NAMES = (
    "paper_beamforming.doppler_theta_var",
    "paper_beamforming.doppler_integral_mean",
    "paper_beamforming.distance_theta_var",
    "paper_beamforming.distance_mean",
    "paper_beamforming.flatness_theta_var",
    "paper_beamforming.flatness_mean",
)
N_WEIGHTS = 40
DOPPLER_WINDOW = 8
DISTANCE_WINDOW = 16
FLATNESS_WINDOW = 1
START = 1  # `processing.py` starts its inner windows at packet 1
MIN_WINDOW_LEN = DISTANCE_WINDOW + 2  # two inner windows, so the variances are not constant


def _weights() -> tuple[np.ndarray, np.ndarray]:
    """Phase weights and their (-pi, pi] labels, accumulated exactly as the script does."""
    steps = np.cumsum(np.full(N_WEIGHTS, 2 * np.pi / N_WEIGHTS))
    return steps, np.where(steps > np.pi, steps - 2 * np.pi, steps)


def _combined_amplitude(ch1: np.ndarray, ch2: np.ndarray, size: int, steps: np.ndarray):
    """Amplitudes of every inner window under every weight: `(windows, weights, size, subc)`."""
    last = ch1.shape[0]
    w1 = sliding_window_view(ch1[START:], size, axis=0).transpose(0, 2, 1)[
        : last - START - size + 1
    ]
    w2 = sliding_window_view(ch2[START:], size, axis=0).transpose(0, 2, 1)[
        : last - START - size + 1
    ]
    rotation = np.exp(steps * 1j)[None, :, None, None]
    return np.abs(w1[:, None] + w2[:, None] * rotation)


def _distance(amp: np.ndarray, fix_distance_loop: bool) -> np.ndarray:
    size = amp.shape[2]
    total = np.zeros(amp.shape[:2])
    count = 0
    for sample in range(size - 1):
        first = sample + 1 if fix_distance_loop else 1
        diff = amp[:, :, first:] - amp[:, :, sample : sample + 1]
        total += np.sqrt((diff**2).sum(axis=-1)).sum(axis=-1)
        count += size - first
    return total / count


def _features(csi: np.ndarray, fix_distance_loop: bool) -> list[float]:
    ch1 = csi[0].astype(np.complex128)
    ch2 = csi[1].astype(np.complex128)
    steps, labels = _weights()
    n_metric = ch1.shape[0] - START - DISTANCE_WINDOW + 1

    amp = _combined_amplitude(ch1, ch2, DOPPLER_WINDOW, steps)
    integral = np.abs(np.fft.fft(amp, axis=2)).mean(axis=-1).sum(axis=-1)
    best = integral.argmax(axis=1)
    doppler_theta = labels[best][:n_metric]
    doppler_max = integral.max(axis=1)[:n_metric]

    amp = _combined_amplitude(ch1, ch2, DISTANCE_WINDOW, steps)
    distance = _distance(amp, fix_distance_loop)
    distance_theta = labels[distance.argmax(axis=1)][:n_metric]
    distance_max = distance.max(axis=1)[:n_metric]

    amp = _combined_amplitude(ch1, ch2, FLATNESS_WINDOW, steps)
    variance = amp.var(axis=-1).mean(axis=-1)
    flatness_theta = labels[variance.argmin(axis=1)][:n_metric]
    flatness_min = variance.min(axis=1)[:n_metric]

    return [
        float(np.var(doppler_theta)),
        float(np.mean(doppler_max)),
        float(np.var(distance_theta)),
        float(np.mean(distance_max)),
        float(np.var(flatness_theta)),
        float(np.mean(flatness_min)),
    ]


@register("extractor", "paper_beamforming")
class PaperBeamforming:
    VERSION = 1
    granularity = "per_window"
    requires = {"phase": True, "n_antennas": 2}
    min_window_len = MIN_WINDOW_LEN

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        # False keeps the script's pair loop (self, duplicate and reversed pairs are counted);
        # True averages over unique pairs only.
        fix_distance_loop: bool = False

    def __init__(self, params: "PaperBeamforming.Params | None" = None) -> None:
        self.params = params or self.Params()

    def extract(self, window_slices: Sequence[np.ndarray]) -> FeatureMatrix:
        if not window_slices:
            raise ExtractorError("paper_beamforming received no windows")
        rows = []
        for csi in window_slices:
            if csi.shape[0] != 2 or csi.shape[1] < MIN_WINDOW_LEN:
                raise ExtractorError(
                    f"paper_beamforming needs 2 antennas and >= {MIN_WINDOW_LEN} packets, "
                    f"got shape {csi.shape}"
                )
            rows.append(_features(csi, self.params.fix_distance_loop))
        return FeatureMatrix(values=np.asarray(rows, dtype=np.float32), feature_names=NAMES)
