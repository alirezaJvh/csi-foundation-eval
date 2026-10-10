"""Reusable FeatureSelector port contract; every selector's test calls it."""

import numpy as np

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.ports import FeatureSelector


def check_selector(
    selector: FeatureSelector,
    train: FeatureMatrix,
    y_train: np.ndarray,
    val: FeatureMatrix,
    y_val: np.ndarray,
    test: FeatureMatrix,
) -> None:
    selector.validate(train.feature_names)
    selector.fit(train, y_train, val, y_val)
    assert selector.candidates
    for index, names in enumerate(selector.candidates):
        selector.choose(index)
        assert selector.selected_names == names
        assert len(set(names)) == len(names)
        assert set(names) <= set(train.feature_names)
        for matrix in (train, val, test):
            out = selector.transform(matrix)
            assert out.feature_names == names
            assert out.values.shape == (matrix.values.shape[0], len(names))
            columns = [matrix.feature_names.index(n) for n in names]
            np.testing.assert_array_equal(out.values, matrix.values[:, columns])
