"""Remove each packet's linear phase trend across subcarriers."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import Recording
from csi_bench.core.registry import register


@register("preprocess", "phase_sanitize")
class PhaseSanitize:
    """Subtract one least-squares phase line `a*k + b` per packet, shared by its antennas.

    The phase is unwrapped across subcarriers, `k = 0..n_sub-1`. The fit is
    `phi_r(k) = a*k + b_r` over all antennas r (one slope; per-antenna intercepts only
    absorb unwrap offsets), and the removed intercept is the circular mean of `b_r`.
    Every antenna of a packet gets the same correction, so magnitudes and the relative
    phase between antennas are kept. Stateless and per packet.
    """

    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, params: "PhaseSanitize.Params | None" = None) -> None:
        self.params = params or self.Params()

    def fit(self, recordings: Sequence[Recording]) -> dict:
        return {}

    def transform(self, recording: Recording, state: dict) -> Recording:
        csi = recording.csi.astype(np.complex128)  # (antennas, packets, subcarriers)
        n_ant, _, n_sub = csi.shape
        phase = np.unwrap(np.angle(csi), axis=2)
        k = np.arange(n_sub, dtype=np.float64)
        kc = k - k.mean()
        denom = n_ant * float((kc**2).sum())
        if denom > 0:
            slope = (phase * kc).sum(axis=(0, 2)) / denom  # (packets,)
        else:
            slope = np.zeros(csi.shape[1])
        intercepts = phase.mean(axis=2) - slope[None, :] * k.mean()  # (antennas, packets)
        offset = np.angle(np.exp(1j * intercepts).sum(axis=0))  # (packets,)
        correction = np.exp(-1j * (slope[:, None] * k[None, :] + offset[:, None]))
        out = (csi * correction[None, :, :]).astype(np.complex64)
        return replace(recording, csi=out)
