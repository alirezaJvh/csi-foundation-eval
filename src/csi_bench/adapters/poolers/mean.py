"""Mean of each window's per-packet rows."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import FeatureMatrix, PacketFeatures
from csi_bench.core.registry import register


@register("pooler", "mean")
class Mean:
    VERSION = 1
    min_window_len = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "Mean.Params | None" = None) -> None:
        self.params = params or self.Params()

    def pool(
        self, packets: PacketFeatures, starts: Sequence[int], window_len: int
    ) -> FeatureMatrix:
        rows = packets.values.astype(np.float64)
        first = np.asarray(starts, dtype=np.int64)
        total = np.zeros((len(rows) + 1, rows.shape[1]))
        np.cumsum(rows, axis=0, out=total[1:])
        mean = (total[first + window_len] - total[first]) / window_len
        return FeatureMatrix(
            values=mean.astype(np.float32),
            feature_names=tuple(f"{name}.mean" for name in packets.feature_names),
        )
