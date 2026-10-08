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
