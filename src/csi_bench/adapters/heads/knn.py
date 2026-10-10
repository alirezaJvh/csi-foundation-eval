"""k-nearest-neighbours head, tuned on val and refit on train ∪ val (AD-11)."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.head_search import tune_and_refit
from csi_bench.core.registry import register


@register("head", "knn")
class Knn:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        k_grid: tuple[PositiveInt, ...] = Field(default=(1, 3, 5, 11), min_length=1)

    def __init__(self, params: "Knn.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed  # kNN has no randomness; kept for the shared constructor
        self._scaler: StandardScaler | None = None
        self._model: KNeighborsClassifier | None = None
        self.val_score = 0.0

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        candidates = [lambda k=k: KNeighborsClassifier(n_neighbors=k) for k in self.params.k_grid]
        self._scaler, self._model, _, self.val_score = tune_and_refit(
            candidates, train, y_train, val, y_val
        )

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._model is not None, "fit before predict"
        return self._model.predict(self._scaler.transform(features.values))
