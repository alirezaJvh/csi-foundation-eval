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
    assert 0.0 <= head.val_score <= 1.0
    predicted = head.predict(test)
    assert predicted.shape == (test.values.shape[0],)
    assert set(np.unique(predicted)) <= set(np.unique(np.concatenate([y_train, y_val])))


def check_seed_determinism(make_head, train, y_train, val, y_val, test) -> None:
    first, second = make_head(7), make_head(7)
    first.fit(train, y_train, val, y_val)
    second.fit(train, y_train, val, y_val)
    np.testing.assert_array_equal(first.predict(test), second.predict(test))


def check_scaler_inside(make_head, part, scale: float = 1e4) -> None:
    """Features on a huge scale still classify well, so scaling happens inside the head."""
    (train, y_train), (val, y_val), (test, y_test) = (part(60, 0), part(30, 1), part(30, 2))
    big = [
        FeatureMatrix(values=m.values * scale, feature_names=m.feature_names)
        for m in (train, val, test)
    ]
    head = make_head(0)
    head.fit(big[0], y_train, big[1], y_val)
    assert float((head.predict(big[2]) == y_test).mean()) > 0.9
