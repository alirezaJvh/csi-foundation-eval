"""Multi-layer-perceptron head (scikit-learn), tuned on val and refit on train ∪ val (AD-11)."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.head_search import tune_and_refit
from csi_bench.core.registry import register


@register("head", "mlp")
class Mlp:
    VERSION = 1

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        hidden_layer_sizes: tuple[PositiveInt, ...] = Field(default=(64,), min_length=1)
        alpha_grid: tuple[PositiveFloat, ...] = Field(default=(1e-4, 1e-2), min_length=1)
        max_iter: int = Field(default=300, ge=1)

    def __init__(self, params: "Mlp.Params | None" = None, seed: int = 0) -> None:
        self.params = params or self.Params()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: MLPClassifier | None = None

    def _make(self, alpha: float) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.params.hidden_layer_sizes,
            alpha=alpha,
            max_iter=self.params.max_iter,
            random_state=self.seed,
        )

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        candidates = [lambda a=a: self._make(a) for a in self.params.alpha_grid]
        self._scaler, self._model, _ = tune_and_refit(candidates, train, y_train, val, y_val)

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._model is not None, "fit before predict"
        return self._model.predict(self._scaler.transform(features.values))
