"""Drop the phase: replace CSI by its magnitude (phase 0)."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import Recording
from csi_bench.core.registry import register


@register("preprocess", "amplitude_only")
class AmplitudeOnly:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "AmplitudeOnly.Params | None" = None) -> None:
        self.params = params or self.Params()

    def fit(self, recordings: Sequence[Recording]) -> dict:
        return {}

    def transform(self, recording: Recording, state: dict) -> Recording:
        return replace(recording, csi=np.abs(recording.csi).astype(np.complex64))
