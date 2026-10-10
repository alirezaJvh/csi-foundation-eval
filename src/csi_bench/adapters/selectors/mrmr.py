"""MRMR feature selection (`mrmr-selection`, `mrmr_classif`), fitted on train rows only."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from csi_bench.core.columns import take_columns
from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.errors import ConfigError
from csi_bench.core.registry import register


@register("selector", "mrmr")
class Mrmr:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        k: list[PositiveInt] = Field(min_length=1)

        @field_validator("k")
        @classmethod
        def _unique(cls, k: list[int]) -> list[int]:
            if len(set(k)) != len(k):
                raise ValueError("k values must be unique")
            return k

    def __init__(self, params: "Mrmr.Params") -> None:
        self.params = params
        self._ranking: tuple[str, ...] = ()
        self._selected: tuple[str, ...] = ()
        self._candidates: tuple[tuple[str, ...], ...] = ()

    @property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        return self._candidates

    @property
    def selected_names(self) -> tuple[str, ...]:
        return self._selected

    def validate(self, feature_names: Sequence[str]) -> None:
        if max(self.params.k) > len(feature_names):
            raise ConfigError(
                f"feature_selection.mrmr.k: {max(self.params.k)} exceeds the "
                f"{len(feature_names)} available columns: {list(feature_names)}"
            )

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        """Rank on train only; every top-k is a prefix of the one greedy ranking."""
        self.validate(train.feature_names)
        frame = pd.DataFrame(train.values, columns=list(train.feature_names))
        ranking = mrmr_classif(
            X=frame,
            y=pd.Series(y_train),
            K=max(self.params.k),
            n_jobs=1,
            show_progress=False,
        )
        self._ranking = tuple(ranking)
        self._candidates = tuple(self._ranking[:k] for k in self.params.k)
        self._selected = self._candidates[0]

    def choose(self, index: int) -> None:
        self._selected = self._candidates[index]

    def transform(self, matrix: FeatureMatrix) -> FeatureMatrix:
        return take_columns(matrix, self._selected)
