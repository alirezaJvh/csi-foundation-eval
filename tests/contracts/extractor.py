"""Reusable FeatureExtractor port contract; every extractor's test calls it."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from csi_bench.core.ports import FeatureExtractor


def check_feature_extractor(extractor: FeatureExtractor, windows: Sequence[np.ndarray]) -> None:
    batch = extractor.extract(windows)
    assert batch.values.dtype == np.float32
    assert batch.values.shape == (len(windows), len(batch.feature_names))
    assert len(set(batch.feature_names)) == len(batch.feature_names)
    assert all(name and "." in name for name in batch.feature_names)
    for i, window in enumerate(windows):
        single = extractor.extract([window])
        assert single.feature_names == batch.feature_names
        np.testing.assert_allclose(single.values, batch.values[i : i + 1], rtol=0, atol=0)
    if getattr(extractor, "uses_context", False):
        _check_block_path(extractor, windows)


def _check_block_path(extractor: FeatureExtractor, windows: Sequence[np.ndarray]) -> None:
    """Block-aware output equals slice-based output exactly, for overlapping windows."""
    window_len = windows[0].shape[1]
    block = np.concatenate([windows[0], windows[1][:, : window_len // 2]], axis=1)
    offsets = list(range(block.shape[1] - window_len + 1))
    blocked = cast(Any, extractor).extract_blocks(block, offsets, window_len)
    sliced = extractor.extract([block[:, o : o + window_len] for o in offsets])
    assert blocked.feature_names == sliced.feature_names
    np.testing.assert_array_equal(blocked.values, sliced.values)
    if hasattr(extractor, "compute_context"):
        context = cast(Any, extractor).compute_context(block)
        shared = cast(Any, extractor).extract_blocks(block, offsets, window_len, context=context)
        np.testing.assert_array_equal(shared.values, sliced.values)
