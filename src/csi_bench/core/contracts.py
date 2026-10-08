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
