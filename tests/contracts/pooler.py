"""Reusable Pooler port contract; every registered pooler's test calls it."""

import numpy as np

from csi_bench.core.contracts import PacketFeatures
from csi_bench.core.ports import Pooler

REDUCERS = {"mean": np.mean, "std": np.std}


def check_pooler(pooler: Pooler, packets: PacketFeatures, window_len: int) -> None:
    """Shape, dtype and names; equals naive per-window numpy; order- and batch-independent."""
    n = len(packets.values)
    starts = list(range(0, n - window_len + 1, 3))
    pooled = pooler.pool(packets, starts, window_len)
    assert pooled.values.dtype == np.float32
    assert pooled.values.shape == (len(starts), len(pooled.feature_names))
    assert len(set(pooled.feature_names)) == len(pooled.feature_names)
    expected_columns = []
    for name in pooled.feature_names:
        source, stat = name.rsplit(".", 1)
        column = packets.feature_names.index(source)
        expected_columns.append(
            [REDUCERS[stat](packets.values[s : s + window_len, column]) for s in starts]
        )
    expected = np.array(expected_columns, dtype=np.float64).T
    np.testing.assert_allclose(pooled.values, expected, rtol=0, atol=1e-5)
    reversed_starts = starts[::-1]
    np.testing.assert_array_equal(
        pooler.pool(packets, reversed_starts, window_len).values, pooled.values[::-1]
    )
    single = pooler.pool(packets, starts[-1:], window_len)
    np.testing.assert_allclose(single.values, pooled.values[-1:], rtol=0, atol=1e-6)
