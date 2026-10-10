"""Column picking shared by feature selectors."""

from collections.abc import Sequence

import numpy as np

from csi_bench.core.contracts import FeatureMatrix


def take_columns(matrix: FeatureMatrix, names: Sequence[str]) -> FeatureMatrix:
    """The matrix restricted to `names`, in the order given."""
    position = {name: i for i, name in enumerate(matrix.feature_names)}
    keep = [position[name] for name in names]
    return FeatureMatrix(
        values=np.ascontiguousarray(matrix.values[:, keep]),
        feature_names=tuple(names),
    )
