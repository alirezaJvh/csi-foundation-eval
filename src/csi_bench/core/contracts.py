"""Immutable data contracts exchanged between stages (AD-4)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecordingMeta:
    n_packets: int
    source_sha256: str
    downsample_factor: int | None = None
    sample_rate_hz: float | None = None


@dataclass(frozen=True)
class Recording:
    """CSI of one recording: complex64, axes (antennas, packets, subcarriers)."""

    csi: np.ndarray
    antenna_ids: tuple[int, ...]
    recording_id: str
    room: str
    label: int
    meta: RecordingMeta


@dataclass(frozen=True)
class SplitPlan:
    """Committed train/val/test packet blocks per recording (0-based, end-exclusive)."""

    dataset: str
    splitter: str
    params: dict
    gap_packets: int
    blocks: dict[str, dict[str, tuple[int, int]]]
    source_sha256: dict[str, str]
    split_hash: str
    unit: str = "block"  # "window": assignment derived per window from params (reference only)


@dataclass(frozen=True)
class Window:
    """One window: a packet span `[start, end)` inside one block of one split part."""

    window_id: str
    recording_id: str
    start: int
    end: int
    split: str


@dataclass(frozen=True)
class WindowSet:
    """Ordered windows (by `(recording_id, start)`) that every stage sees."""

    windows: tuple[Window, ...]

    def __iter__(self):
        return iter(self.windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Window:
        return self.windows[index]


@dataclass(frozen=True)
class FeatureMatrix:
    """float32 `(n_windows, d)` row-aligned to the WindowSet, with unique column names."""

    values: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class Predictions:
    """Predicted labels for one split, row-aligned to the WindowSet."""

    window_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    split: str


@dataclass(frozen=True)
class Metrics:
    """Scores for one split, keyed by class name where applicable."""

    split: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: tuple[tuple[int, ...], ...]
    class_names: tuple[str, ...]
