"""Logistic-regression head, tuned on val and refit on train ∪ val (AD-11)."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.head_search import tune_and_refit
from csi_bench.core.registry import register


@register("head", "logreg")
class LogReg:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        C_grid: tuple[PositiveFloat, ...] = Field(default=(0.01, 0.1, 1.0, 10.0), min_length=1)
        max_iter: int = Field(default=1000, ge=1)

    def __init__(self, params: "LogReg.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self.val_score = 0.0

    def _make(self, c: float) -> LogisticRegression:
        return LogisticRegression(C=c, max_iter=self.params.max_iter, random_state=self.seed)

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
