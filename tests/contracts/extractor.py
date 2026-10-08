"""Reusable FeatureExtractor port contract; every extractor's test calls it."""

from collections.abc import Sequence

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
