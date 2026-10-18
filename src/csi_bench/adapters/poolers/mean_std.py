"""Mean and standard deviation of each window's per-packet rows."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import FeatureMatrix, PacketFeatures
from csi_bench.core.registry import register


@register("pooler", "mean_std")
class MeanStd:
    VERSION = 1
    min_window_len = 2  # the std of one packet is always 0

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "MeanStd.Params | None" = None) -> None:
        self.params = params or self.Params()

    def pool(
        self, packets: PacketFeatures, starts: Sequence[int], window_len: int
    ) -> FeatureMatrix:
        # centre each column first so E[x²] - E[x]² does not cancel catastrophically
        rows = packets.values.astype(np.float64)
        centre = rows.mean(axis=0)
        rows = rows - centre
        first = np.asarray(starts, dtype=np.int64)
        last = first + window_len
        total = np.zeros((len(rows) + 1, rows.shape[1]))
        np.cumsum(rows, axis=0, out=total[1:])
        shifted = (total[last] - total[first]) / window_len
        np.cumsum(rows * rows, axis=0, out=total[1:])
        square = (total[last] - total[first]) / window_len
        std = np.sqrt(np.maximum(square - shifted * shifted, 0.0))
        mean = shifted + centre
        names = packets.feature_names
        return FeatureMatrix(
            values=np.hstack([mean, std]).astype(np.float32),
            feature_names=tuple(f"{n}.mean" for n in names) + tuple(f"{n}.std" for n in names),
        )
