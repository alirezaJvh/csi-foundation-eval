"""Support-vector-machine head, tuned on val and refit on train ∪ val (AD-11)."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.head_search import tune_and_refit
from csi_bench.core.registry import register

# libsvm kernel cache in MB (default 200). Speed only, never the result; kept modest because
# candidates are fitted in parallel.
KERNEL_CACHE_MB = 500


@register("head", "svm")
class Svm:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        C_grid: tuple[PositiveFloat, ...] = Field(default=(0.1, 1.0, 10.0), min_length=1)
        kernel: Literal["rbf", "linear"] = "rbf"

    def __init__(self, params: "Svm.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: SVC | None = None
        self.val_score = 0.0

    def _make(self, c: float) -> SVC:
        return SVC(
            C=c, kernel=self.params.kernel, random_state=self.seed, cache_size=KERNEL_CACHE_MB
        )

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        candidates = [lambda c=c: self._make(c) for c in self.params.C_grid]
        self._scaler, self._model, _, self.val_score = tune_and_refit(
            candidates, train, y_train, val, y_val
        )

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._model is not None, "fit before predict"
        return self._model.predict(self._scaler.transform(features.values))
