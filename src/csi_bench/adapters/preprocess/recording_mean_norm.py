"""Divide each antenna of a recording by its own mean amplitude."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from csi_bench.core.contracts import Recording
from csi_bench.core.errors import DataError
from csi_bench.core.registry import register


@register("preprocess", "recording_mean_norm")
class RecordingMeanNorm:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        fit_scope: Literal["train", "whole_recording"] = "train"

    def __init__(self, params: "RecordingMeanNorm.Params | None" = None) -> None:
        self.params = params or self.Params()
        self.fit_scope = self.params.fit_scope

    def fit(self, recordings: Sequence[Recording]) -> dict:
        means: dict[str, list[float]] = {}
        for rec in recordings:
            if rec.csi.shape[1] == 0:
                raise DataError(f"recording_mean_norm: no data in {rec.recording_id}")
            per_antenna = np.abs(rec.csi).mean(axis=(1, 2), dtype=np.float64)
            if not np.isfinite(per_antenna).all() or (per_antenna <= 0).any():
                raise DataError(
                    f"recording_mean_norm: invalid mean amplitude in {rec.recording_id}"
                )
            means[rec.recording_id] = [float(m) for m in per_antenna]
        return {"means": means}

    def transform(self, recording: Recording, state: dict) -> Recording:
        means = np.asarray(state["means"][recording.recording_id], dtype=np.float32)
        csi = (recording.csi / means[:, None, None]).astype(np.complex64)
        return replace(recording, csi=csi)
