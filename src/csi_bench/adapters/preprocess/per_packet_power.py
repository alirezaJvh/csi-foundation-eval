"""Scale each packet to unit average power over its antennas and subcarriers."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import Recording
from csi_bench.core.registry import register


@register("preprocess", "per_packet_power")
class PerPacketPower:
    """Divide each packet by `sqrt(mean |h|^2)`; keeps power ratios between antennas.

    Stateless and per packet, so nothing crosses split blocks. A zero-power packet
    is left unchanged.
    """

    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "PerPacketPower.Params | None" = None) -> None:
        self.params = params or self.Params()

    def fit(self, recordings: Sequence[Recording]) -> dict:
        return {}

    def transform(self, recording: Recording, state: dict) -> Recording:
        csi = recording.csi.astype(np.complex128)
        power = (np.abs(csi) ** 2).mean(axis=(0, 2))  # (packets,)
        scale = np.ones_like(power)
        np.divide(1.0, np.sqrt(power), out=scale, where=power > 0)
        out = (csi * scale[None, :, None]).astype(np.complex64)
        return replace(recording, csi=out)
