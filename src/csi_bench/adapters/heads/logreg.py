"""Logistic-regression head, tuned on val and refit on train ∪ val (AD-11)."""

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.registry import register


@register("head", "logreg")
class LogReg:
    VERSION = 1

    class Params(BaseModel):
        C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
        max_iter: int = 1000

    def __init__(self, params: "LogReg.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None

    def _make(self, c: float) -> LogisticRegression:
        return LogisticRegression(C=c, max_iter=self.params.max_iter, random_state=self.seed)

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        scaler = StandardScaler().fit(train.values)
        x_train = scaler.transform(train.values)
        x_val = scaler.transform(val.values)
        best_c, best_f1 = self.params.C_grid[0], -1.0
        for c in self.params.C_grid:
            model = self._make(c).fit(x_train, y_train)
            score = float(f1_score(y_val, model.predict(x_val), average="macro", zero_division=0))
            if score > best_f1:
                best_c, best_f1 = c, score
        self._scaler = scaler
        x_all = np.vstack([x_train, x_val])
        y_all = np.concatenate([y_train, y_val])
        self._model = self._make(best_c).fit(x_all, y_all)

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._model is not None, "fit before predict"
        return self._model.predict(self._scaler.transform(features.values))
