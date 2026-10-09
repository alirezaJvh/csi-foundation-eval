"""Amplitude statistics feature family (per-window, AD-8)."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.errors import ExtractorError
from csi_bench.core.registry import register

NAMES = (
    "amplitude_stats.mean_power",
    "amplitude_stats.std",
    "amplitude_stats.flatness",
)


def _features(csi: np.ndarray) -> tuple[float, float, float]:
    amplitude = np.abs(csi).astype(np.float64)
    power = amplitude**2
    mean_power = float(power.mean())
    std = float(amplitude.std())
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness = float(np.exp(np.log(power).mean()) / mean_power)
    return mean_power, std, flatness


@register("extractor", "amplitude_stats")
class AmplitudeStats:
    VERSION = 1
    granularity = "per_window"
    requires = {"phase": False, "n_antennas": "any"}
    min_window_len = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "AmplitudeStats.Params | None" = None) -> None:
        self.params = params or self.Params()

    def extract(self, window_slices: Sequence[np.ndarray]) -> FeatureMatrix:
        if not window_slices:
            raise ExtractorError("amplitude_stats received no windows")
        rows = [_features(csi) for csi in window_slices]
        return FeatureMatrix(values=np.asarray(rows, dtype=np.float32), feature_names=NAMES)
