"""Random-forest head, tuned on val and refit on train ∪ val (AD-11)."""

import itertools

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.head_search import tune_and_refit
from csi_bench.core.registry import register


@register("head", "random_forest")
class RandomForest:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        n_estimators_grid: tuple[PositiveInt, ...] = Field(default=(100, 300), min_length=1)
        max_depth_grid: tuple[PositiveInt | None, ...] = Field(default=(None, 10), min_length=1)

    def __init__(self, params: "RandomForest.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: RandomForestClassifier | None = None

    def _make(self, n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=self.seed
        )

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        grid = itertools.product(self.params.n_estimators_grid, self.params.max_depth_grid)
        candidates = [lambda n=n, d=d: self._make(n, d) for n, d in grid]
        self._scaler, self._model, _ = tune_and_refit(candidates, train, y_train, val, y_val)

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._model is not None, "fit before predict"
        return self._model.predict(self._scaler.transform(features.values))
