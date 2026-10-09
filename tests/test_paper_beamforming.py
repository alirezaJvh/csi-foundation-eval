import numpy as np
import pytest

from csi_bench.adapters.extractors.paper_beamforming import NAMES, PaperBeamforming
from csi_bench.core.errors import ExtractorError
from tests.contracts.extractor import check_feature_extractor
from tests.oracles.paper_processing import paper_features


def _windows(count: int, packets: int = 30, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (2, packets, 6)
    return [
        (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128)
        for _ in range(count)
    ]


@pytest.mark.parametrize("fix", [False, True])
def test_matches_oracle(fix):
    windows = _windows(3)
    extractor = PaperBeamforming(PaperBeamforming.Params(fix_distance_loop=fix))
    matrix = extractor.extract(windows)
    expected = np.array([paper_features(w, fix) for w in windows])
    np.testing.assert_allclose(matrix.values, expected, rtol=1e-4, atol=1e-6)


def test_fix_changes_only_distance_columns():
    window = _windows(1)
    plain = PaperBeamforming().extract(window).values[0]
    fixed = (
        PaperBeamforming(PaperBeamforming.Params(fix_distance_loop=True)).extract(window).values[0]
    )
    assert plain[3] != fixed[3]
    np.testing.assert_allclose(np.delete(plain, [2, 3]), np.delete(fixed, [2, 3]))


def test_contract_batch_equals_single():
    check_feature_extractor(PaperBeamforming(), _windows(3))


def test_depends_only_on_own_window():
    a, b = _windows(2)
    alone = PaperBeamforming().extract([a]).values
    together = PaperBeamforming().extract([b, a]).values[1:]
    np.testing.assert_allclose(alone, together, rtol=0, atol=0)


def test_too_short_or_wrong_antennas_raise():
    with pytest.raises(ExtractorError):
        PaperBeamforming().extract(_windows(1, packets=PaperBeamforming.min_window_len - 1))
    with pytest.raises(ExtractorError):
        PaperBeamforming().extract([np.ones((4, 30, 6), dtype=np.complex64)])
    with pytest.raises(ExtractorError):
        PaperBeamforming().extract([])


def test_metadata():
    assert NAMES[0].startswith("paper_beamforming.") and len(set(NAMES)) == 6
    assert PaperBeamforming.granularity == "per_window"
    assert PaperBeamforming.requires == {"phase": True, "n_antennas": 2}
    assert PaperBeamforming.min_window_len == 18


def test_minimum_window_len_runs_on_complex64():
    rng = np.random.default_rng(3)
    shape = (2, PaperBeamforming.min_window_len, 6)
    window = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    values = PaperBeamforming().extract([window]).values
    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        values, [paper_features(window.astype(np.complex128))], rtol=1e-4, atol=1e-6
    )
