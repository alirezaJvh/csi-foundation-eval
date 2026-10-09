"""Shared tune-on-val, refit-on-train∪val routine for classifier heads (AD-11)."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from csi_bench.core.contracts import FeatureMatrix


def tune_and_refit(
    candidates: Sequence[Callable[[], Any]],
    train: FeatureMatrix,
    y_train: np.ndarray,
    val: FeatureMatrix,
    y_val: np.ndarray,
) -> tuple[StandardScaler, Any, int]:
    """Pick the candidate with the best val macro-F1, then refit it on train ∪ val.

    Each candidate is a factory returning a fresh unfitted model. One scaler is fitted
    on train and every candidate is fitted and scored in that scaled space; the first
    best wins ties. The scaler and the winning model are then refitted on train ∪ val.
    Returns the final scaler, the final model and the chosen candidate index.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    scaler = StandardScaler().fit(train.values)
    x_train = scaler.transform(train.values)
    x_val = scaler.transform(val.values)
    best_index, best_f1 = 0, -1.0
    for index, make in enumerate(candidates):
        model = make().fit(x_train, y_train)
        score = float(f1_score(y_val, model.predict(x_val), average="macro", zero_division=0))
        if score > best_f1:
            best_index, best_f1 = index, score
    # The final model sees train ∪ val, so its scaler is fitted on the same rows.
    x_raw = np.vstack([train.values, val.values])
    y_all = np.concatenate([y_train, y_val])
    final_scaler = StandardScaler().fit(x_raw)
    final_model = candidates[best_index]().fit(final_scaler.transform(x_raw), y_all)
    return final_scaler, final_model, best_index
