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
POSITION_CHUNK = 32  # inner starts per batch; bounds the `(positions, 40, size, subc)` temporaries
MIN_WINDOW_LEN = DISTANCE_WINDOW + 2  # two inner windows, so the variances are not constant


def _weights() -> tuple[np.ndarray, np.ndarray]:
    """Phase weights and their (-pi, pi] labels, accumulated exactly as the script does."""
    steps = np.cumsum(np.full(N_WEIGHTS, 2 * np.pi / N_WEIGHTS))
    return steps, np.where(steps > np.pi, steps - 2 * np.pi, steps)


def _combined_amplitude(ch1: np.ndarray, ch2: np.ndarray, size: int, steps: np.ndarray):
    """Amplitudes of every inner start under every weight: `(positions, weights, size, subc)`."""
    w1 = sliding_window_view(ch1, size, axis=0).transpose(0, 2, 1)
    w2 = sliding_window_view(ch2, size, axis=0).transpose(0, 2, 1)
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


def _position_metrics(
    ch1: np.ndarray, ch2: np.ndarray, fix_distance_loop: bool, first: int, count: int
) -> np.ndarray:
    """Six metrics for the inner starts `first .. first+count-1` (block-relative, before START).

    Returns `(6, count)`: doppler theta/max, distance theta/max, flatness theta/min. Each
    position reads only its own inner window's packets, so any chunking gives the same values.
    """
    steps, labels = _weights()
    out = np.empty((6, count))
    for lo in range(0, count, POSITION_CHUNK):
        n = min(POSITION_CHUNK, count - lo)
        base = START + first + lo
        c1, c2 = ch1[base:], ch2[base:]

        amp = _combined_amplitude(
            c1[: n + DOPPLER_WINDOW - 1], c2[: n + DOPPLER_WINDOW - 1], DOPPLER_WINDOW, steps
        )
        integral = np.abs(np.fft.fft(amp, axis=2)).mean(axis=-1).sum(axis=-1)
        out[0, lo : lo + n] = labels[integral.argmax(axis=1)]
        out[1, lo : lo + n] = integral.max(axis=1)

        amp = _combined_amplitude(
            c1[: n + DISTANCE_WINDOW - 1], c2[: n + DISTANCE_WINDOW - 1], DISTANCE_WINDOW, steps
        )
        distance = _distance(amp, fix_distance_loop)
        out[2, lo : lo + n] = labels[distance.argmax(axis=1)]
        out[3, lo : lo + n] = distance.max(axis=1)

        amp = _combined_amplitude(c1[:n], c2[:n], FLATNESS_WINDOW, steps)
        variance = amp.var(axis=-1).mean(axis=-1)
        out[4, lo : lo + n] = labels[variance.argmin(axis=1)]
        out[5, lo : lo + n] = variance.min(axis=1)
    return out


def _reduce(metrics: np.ndarray, first: int, n_metric: int) -> list[float]:
    """The six features of one window from the per-position metrics at `first .. +n_metric`."""
    m = metrics[:, first : first + n_metric]
    return [
        float(np.var(m[0])),
        float(np.mean(m[1])),
        float(np.var(m[2])),
        float(np.mean(m[3])),
        float(np.var(m[4])),
        float(np.mean(m[5])),
    ]


def _features(csi: np.ndarray, fix_distance_loop: bool) -> list[float]:
    ch1 = csi[0].astype(np.complex128)
    ch2 = csi[1].astype(np.complex128)
    n_metric = ch1.shape[0] - START - DISTANCE_WINDOW + 1
    metrics = _position_metrics(ch1, ch2, fix_distance_loop, 0, n_metric)
    return _reduce(metrics, 0, n_metric)


@register("extractor", "paper_beamforming")
class PaperBeamforming:
    VERSION = 1
    granularity = "per_window"
    requires = {"phase": True, "n_antennas": 2}
    min_window_len = MIN_WINDOW_LEN
    uses_context = True

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

    def compute_context(self, block: np.ndarray) -> np.ndarray:
        """Per-position metrics `(6, B - 16)` of one whole split block.

        They depend on the block alone, so every window length, stride and room set can share
        them.
        """
        if block.shape[0] != 2 or block.shape[1] < MIN_WINDOW_LEN:
            raise ExtractorError(
                f"paper_beamforming needs 2 antennas and >= {MIN_WINDOW_LEN} packets, "
                f"got block {block.shape}"
            )
        ch1 = block[0].astype(np.complex128)
        ch2 = block[1].astype(np.complex128)
        count = block.shape[1] - START - DISTANCE_WINDOW + 1
        return _position_metrics(ch1, ch2, self.params.fix_distance_loop, 0, count)

    def extract_blocks(
        self,
        block: np.ndarray,
        offsets: Sequence[int],
        window_len: int,
        context: np.ndarray | None = None,
    ) -> FeatureMatrix:
        """Features of the windows `[o, o + window_len)` inside one split block.

        Inner-window metrics are computed once per packet position and shared by the windows
        that contain them; the output equals `extract` on the same slices exactly. `context`
        is `compute_context(block)` for the whole block, possibly cached; offsets are relative
        to the block start either way.
        """
        if not offsets:
            raise ExtractorError("paper_beamforming received no windows")
        if block.shape[0] != 2 or window_len < MIN_WINDOW_LEN:
            raise ExtractorError(
                f"paper_beamforming needs 2 antennas and window_len >= {MIN_WINDOW_LEN}, "
                f"got block {block.shape} and window_len {window_len}"
            )
        n_metric = window_len - START - DISTANCE_WINDOW + 1
        first, last = min(offsets), max(offsets)
        if first < 0 or last + window_len > block.shape[1]:
            raise ExtractorError(f"window offsets {first}..{last} fall outside the block")
        if context is None:
            ch1 = block[0].astype(np.complex128)
            ch2 = block[1].astype(np.complex128)
            metrics = _position_metrics(
                ch1, ch2, self.params.fix_distance_loop, first, last - first + n_metric
            )
            base = first
        else:
            if context.shape != (6, block.shape[1] - START - DISTANCE_WINDOW + 1):
                raise ExtractorError(f"context shape {context.shape} does not match the block")
            metrics = context
            base = 0
        rows = [_reduce(metrics, o - base, n_metric) for o in offsets]
        return FeatureMatrix(values=np.asarray(rows, dtype=np.float32), feature_names=NAMES)
