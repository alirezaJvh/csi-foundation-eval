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
    assert PaperBeamforming.uses_context is True


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


def _block(packets: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (2, packets, 6)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


@pytest.mark.parametrize("fix", [False, True])
@pytest.mark.parametrize(("window_len", "stride"), [(18, 1), (30, 1), (25, 7), (40, 3)])
def test_extract_blocks_equals_slice_path_exactly(fix, window_len, stride):
    extractor = PaperBeamforming(PaperBeamforming.Params(fix_distance_loop=fix))
    block = _block(120)
    offsets = list(range(0, block.shape[1] - window_len + 1, stride))
    blocked = extractor.extract_blocks(block, offsets, window_len)
    sliced = extractor.extract([block[:, o : o + window_len] for o in offsets])
    assert blocked.feature_names == sliced.feature_names
    np.testing.assert_array_equal(blocked.values, sliced.values)


def test_extract_blocks_crosses_position_chunks_identically(monkeypatch):
    from csi_bench.adapters.extractors import paper_beamforming as module

    block = _block(90)
    offsets = list(range(0, 61))
    reference = PaperBeamforming().extract_blocks(block, offsets, 30).values
    monkeypatch.setattr(module, "POSITION_CHUNK", 5)
    np.testing.assert_array_equal(
        PaperBeamforming().extract_blocks(block, offsets, 30).values, reference
    )


def test_extract_blocks_rejects_bad_input():
    extractor = PaperBeamforming()
    with pytest.raises(ExtractorError):
        extractor.extract_blocks(_block(40), [], 20)
    with pytest.raises(ExtractorError):
        extractor.extract_blocks(_block(40), [0], PaperBeamforming.min_window_len - 1)
    with pytest.raises(ExtractorError):
        extractor.extract_blocks(_block(40), [30], 20)
    with pytest.raises(ExtractorError):
        extractor.extract_blocks(np.ones((4, 40, 6), dtype=np.complex64), [0], 20)


@pytest.mark.parametrize("fix", [False, True])
def test_shared_context_equals_slice_path_for_any_length_and_stride(fix):
    rng = np.random.default_rng(3)
    shape = (2, 90, 5)
    block = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    extractor = PaperBeamforming(PaperBeamforming.Params(fix_distance_loop=fix))
    context = extractor.compute_context(block)
    assert context.shape == (6, 90 - 16)
    for window_len, stride in [(18, 1), (20, 1), (30, 4), (50, 7), (90, 1)]:
        offsets = list(range(0, 90 - window_len + 1, stride))
        shared = extractor.extract_blocks(block, offsets, window_len, context=context)
        alone = extractor.extract_blocks(block, offsets, window_len)
        sliced = extractor.extract([block[:, o : o + window_len] for o in offsets])
        np.testing.assert_array_equal(shared.values, sliced.values)
        np.testing.assert_array_equal(alone.values, sliced.values)


def test_context_shape_mismatch_raises():
    block = _windows(1, packets=40)[0]
    extractor = PaperBeamforming()
    with pytest.raises(ExtractorError):
        extractor.extract_blocks(block, [0], 20, context=np.zeros((6, 5)))
