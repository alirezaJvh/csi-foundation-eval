"""Manual feature selection by column name or whole family (AD-18)."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from csi_bench.core.columns import take_columns
from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.errors import ConfigError
from csi_bench.core.registry import register


@register("selector", "manual")
class Manual:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        columns: list[str] = Field(min_length=1)

    def __init__(self, params: "Manual.Params") -> None:
        self.params = params
        self._names: tuple[str, ...] = ()

    @property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        return (self._names,)

    @property
    def selected_names(self) -> tuple[str, ...]:
        return self._names

    def validate(self, feature_names: Sequence[str]) -> None:
        """Resolve entries (a column name or a family prefix) against the matrix columns."""
        chosen: set[str] = set()
        for entry in self.params.columns:
            matched = [n for n in feature_names if n == entry or n.startswith(f"{entry}.")]
            if not matched:
                raise ConfigError(
                    f"feature_selection: {entry!r} matches no column or family; "
                    f"available columns: {list(feature_names)}"
                )
            chosen.update(matched)
        self._names = tuple(n for n in feature_names if n in chosen)

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        self.validate(train.feature_names)

    def choose(self, index: int) -> None:
        if index != 0:
            raise IndexError(index)

    def transform(self, matrix: FeatureMatrix) -> FeatureMatrix:
        return take_columns(matrix, self._names)
