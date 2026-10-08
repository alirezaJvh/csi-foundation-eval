import numpy as np
import pytest

from csi_bench.adapters.extractors.amplitude_stats import AmplitudeStats
from csi_bench.core.errors import ExtractorError
from tests.contracts.extractor import check_feature_extractor


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_columns_and_dtype():
    windows = [_rng(i).normal(size=(4, 40, 3)).astype(np.complex64) for i in range(5)]
    matrix = AmplitudeStats().extract(windows)
    assert matrix.feature_names == (
        "amplitude_stats.mean_power",
        "amplitude_stats.std",
        "amplitude_stats.flatness",
    )
    assert matrix.values.dtype == np.float32
    assert matrix.values.shape == (5, 3)


def test_batch_equals_single():
    windows = [_rng(i).normal(size=(2, 30, 3)).astype(np.complex64) for i in range(4)]
    check_feature_extractor(AmplitudeStats(), windows)


def test_values_are_pooled_statistics():
    csi = np.array([[[3 + 4j, 0 + 0j]], [[0 + 0j, 0 + 0j]]], dtype=np.complex64)
    mean_power, std, flatness = AmplitudeStats().extract([csi]).values[0]
    amplitude = np.abs(csi)
    assert mean_power == pytest.approx(float((amplitude**2).mean()))
    assert std == pytest.approx(float(amplitude.std()))
    with np.errstate(divide="ignore"):
        geometric = float(np.exp(np.log(amplitude**2).mean()))
    assert flatness == pytest.approx(geometric / float((amplitude**2).mean()))


def test_empty_raises():
    with pytest.raises(ExtractorError):
        AmplitudeStats().extract([])


def test_metadata():
    assert AmplitudeStats.VERSION == 1
    assert AmplitudeStats.granularity == "per_window"
    assert AmplitudeStats.requires == {"phase": False, "n_antennas": "any"}
