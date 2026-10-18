"""Stage ports (typing.Protocol) implemented by adapters."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from csi_bench.core.contracts import FeatureMatrix, PacketFeatures, Recording, SplitPlan


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

    # A `granularity = "per_packet"` extractor implements
    #     extract_packets(csi) -> PacketFeatures
    # in place of `extract`: `csi` is one complex64 `(antennas, packets, subcarriers)` array and
    # row i of the result is packet i. The runner caches it per recording and a `Pooler`
    # reduces each window's rows to one vector.

    VERSION: int
    granularity: str
    requires: dict
    min_window_len: int

    def extract(self, window_slices: Sequence[np.ndarray]) -> FeatureMatrix:
        """Return features for the given complex64 `(antennas, packets, subcarriers)` slices."""
        ...

    # Optional. An extractor with `uses_context = True` also implements
    #     extract_blocks(block, offsets, window_len) -> FeatureMatrix
    # where `block` is one split block `(antennas, packets, subcarriers)` and the windows are
    # `[o, o + window_len)` for each block-relative offset `o`. It may share per-packet
    # intermediates across overlapping windows; its output must equal `extract` on the same
    # slices exactly (contract test). The runner calls it per block, never across a split gap.
    # It may also offer `compute_context(block) -> np.ndarray` (intermediates of the whole block,
    # independent of window length, stride and room set); the runner then caches them per block
    # and passes them back as `extract_blocks(..., context=...)`, which must give the same output.


class Pooler(Protocol):
    """Weight-free reduction of a window's per-packet rows to one vector (AD-8)."""

    VERSION: int
    min_window_len: int

    def pool(
        self, packets: PacketFeatures, starts: Sequence[int], window_len: int
    ) -> FeatureMatrix:
        """Row j pools packet rows `[starts[j], starts[j] + window_len)` of one recording."""
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


class ContextCache(Protocol):
    """Runner-owned store of one block's extractor intermediates (AD-10)."""

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached array; a corrupt entry is a miss."""
        ...

    def put(self, key: str, values: np.ndarray) -> None:
        """Store atomically; a failed write leaves no entry."""
        ...


class PacketCache(Protocol):
    """Runner-owned store of one recording's per-packet features (AD-10)."""

    def get(self, key: str) -> PacketFeatures | None:
        """Return the cached features; a corrupt entry is a miss."""
        ...

    def put(self, key: str, features: PacketFeatures) -> None:
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
