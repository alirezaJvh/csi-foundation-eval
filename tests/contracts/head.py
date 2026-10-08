"""Reusable Head port contract; every head's test calls it."""

import numpy as np

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.ports import Head


def check_head(
    head: Head,
    train: FeatureMatrix,
    y_train: np.ndarray,
    val: FeatureMatrix,
    y_val: np.ndarray,
    test: FeatureMatrix,
) -> None:
    head.fit(train, y_train, val, y_val)
    predicted = head.predict(test)
    assert predicted.shape == (test.values.shape[0],)
    assert set(np.unique(predicted)) <= set(np.unique(np.concatenate([y_train, y_val])))
