"""Antenna selection: keep the listed original antenna ids, in listed order."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from csi_bench.core.contracts import Recording
from csi_bench.core.errors import ConfigError
from csi_bench.core.registry import register


def _rows(recording: Recording, ids: Sequence[int]) -> list[int]:
    missing = [i for i in ids if i not in recording.antenna_ids]
    if missing:
        raise ConfigError(
            f"antennas: ids {missing} not in recording {recording.recording_id} "
            f"(has {list(recording.antenna_ids)})"
        )
    return [recording.antenna_ids.index(i) for i in ids]


@register("preprocess", "antennas")
class Antennas:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        ids: list[int] = Field(min_length=1)

        @field_validator("ids")
        @classmethod
        def _unique(cls, ids: list[int]) -> list[int]:
            if len(set(ids)) != len(ids):
                raise ValueError("ids must be unique")
            return ids

    def __init__(self, params: "Antennas.Params") -> None:
        self.params = params

    def fit(self, recordings: Sequence[Recording]) -> dict:
        for rec in recordings:
            _rows(rec, self.params.ids)
        return {"ids": list(self.params.ids)}

    def transform(self, recording: Recording, state: dict) -> Recording:
        ids = state["ids"]
        rows = _rows(recording, ids)
        return replace(
            recording, csi=np.ascontiguousarray(recording.csi[rows]), antenna_ids=tuple(ids)
        )
