"""Stage ports (typing.Protocol) implemented by adapters."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from csi_bench.core.contracts import FeatureMatrix, Recording, SplitPlan


class DatasetSource(Protocol):
    @property
    def classes(self) -> tuple[str, ...]: ...

    def load(self) -> list[Recording]:
        """Return all recordings, sorted by recording_id."""
        ...


class Splitter(Protocol):
    def split(self, recordings: list[Recording], dataset: str) -> SplitPlan:
        """Return the train/val/test packet blocks for every recording."""
        ...


class FeatureExtractor(Protocol):
    """Pure per-window extractor; batched output equals single-window output (AD-8)."""

    VERSION: int
    granularity: str
    requires: dict
    min_window_len: int

    def extract(self, window_slices: Sequence[np.ndarray]) -> FeatureMatrix:
        """Return features for the given complex64 `(antennas, packets, subcarriers)` slices."""
        ...


class Head(Protocol):
    """Classifier that tunes on val, refits on train ∪ val and predicts once (AD-11)."""

    VERSION: int

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        """Tune hyperparameters on val, then refit on train ∪ val."""
        ...

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        """Return predicted integer labels, row-aligned to the features."""
        ...
