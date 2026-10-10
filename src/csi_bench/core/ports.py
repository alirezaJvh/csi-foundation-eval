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
    val_score: float  # best val macro-F1 reached while tuning, set by `fit`

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


class FeatureSelector(Protocol):
    """Column selector fitted on train and val rows only; test is only ever transformed."""

    VERSION: int

    @property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        """Candidate column subsets after `fit`, in matrix column order (first wins ties)."""
        ...

    @property
    def selected_names(self) -> tuple[str, ...]:
        """Columns kept by the chosen candidate (the first one until `choose` is called)."""
        ...

    def validate(self, feature_names: Sequence[str]) -> None:
        """Raise `ConfigError` if the selector cannot apply to these columns."""
        ...

    def fit(
        self,
        train: FeatureMatrix,
        y_train: np.ndarray,
        val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> None:
        """Compute the candidate subsets from train rows (val may be used to validate)."""
        ...

    def choose(self, index: int) -> None:
        """Make `candidates[index]` the selection used by `transform`."""
        ...

    def transform(self, matrix: FeatureMatrix) -> FeatureMatrix:
        """Return the matrix restricted to `selected_names`, in column order."""
        ...


class FeatureCache(Protocol):
    """Runner-owned store of one extractor's matrix over all windows (AD-10)."""

    def get(self, key: str, window_ids: Sequence[str] | None = None) -> FeatureMatrix | None:
        """Return the cached matrix; a corrupt entry or one whose windows differ is a miss."""
        ...

    def put(self, key: str, matrix: FeatureMatrix, window_ids: Sequence[str]) -> None:
        """Store atomically; a failed write leaves no entry."""
        ...


class PreprocessStep(Protocol):
    """Fitted preprocessing step: `fit` sees train blocks only, `transform` is pure."""

    VERSION: int

    def fit(self, recordings: Sequence[Recording]) -> dict:
        """Return JSON-able fitted state from (train-block views of) the recordings."""
        ...

    def transform(self, recording: Recording, state: dict) -> Recording:
        """Return a new `Recording` using the fitted state; never mutate the input."""
        ...
