"""Divide all CSI by the mean amplitude over the selected train blocks."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import Recording
from csi_bench.core.errors import DataError
from csi_bench.core.registry import register


@register("preprocess", "global_rescale")
class GlobalRescale:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "GlobalRescale.Params | None" = None) -> None:
        self.params = params or self.Params()

    def fit(self, recordings: Sequence[Recording]) -> dict:
        total = 0.0
        count = 0
        for rec in recordings:
            total += float(np.abs(rec.csi).sum(dtype=np.float64))
            count += rec.csi.size
        if count == 0:
            raise DataError("global_rescale: no train data to fit on")
        mean = total / count
        if not np.isfinite(mean) or mean <= 0:
            raise DataError(f"global_rescale: invalid mean amplitude {mean}")
        return {"mean_amplitude": mean}

    def transform(self, recording: Recording, state: dict) -> Recording:
        csi = (recording.csi / np.float32(state["mean_amplitude"])).astype(np.complex64)
        return replace(recording, csi=csi)
