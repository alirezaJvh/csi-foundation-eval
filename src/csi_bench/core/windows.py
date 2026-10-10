"""The Windower: the sole owner of window boundaries (AD-7).

A core function, not an adapter: every model sees the same windows for a given
`(split, selection, window_len, stride)`.
"""

import zlib
from collections.abc import Sequence

import numpy as np

from csi_bench.core.contracts import Recording, SplitPlan, Window, WindowSet
from csi_bench.core.errors import ConfigError


def build_windows(
    plan: SplitPlan,
    recordings: Sequence[Recording],
    rooms: Sequence[str],
    window_len: int,
    stride: int,
) -> WindowSet:
    """Return selected-room windows fully inside a single split block, in AD-4 order."""
    if window_len < 1:
        raise ConfigError(f"window_len must be >= 1, got {window_len}")
    if stride < 1:
        raise ConfigError(f"stride must be >= 1, got {stride}")
    if plan.unit != "window" and window_len > plan.gap_packets:
        raise ConfigError(f"window_len {window_len} exceeds plan.gap_packets {plan.gap_packets}")
    wanted = set(rooms)
    windows: list[Window] = []
    for rec in recordings:
        if rec.room not in wanted:
            continue
        if plan.unit == "window":
            windows.extend(_random_windows(plan, rec, window_len, stride))
            continue
        for part, (block_start, block_end) in plan.blocks[rec.recording_id].items():
            start = block_start
            while start + window_len <= block_end:
                windows.append(
                    Window(
                        window_id=f"{rec.recording_id}@{start}:{start + window_len}",
                        recording_id=rec.recording_id,
                        start=start,
                        end=start + window_len,
                        split=part,
                    )
                )
                start += stride
    windows.sort(key=lambda w: (w.recording_id, w.start))
    return WindowSet(windows=tuple(windows))


def _random_windows(plan: SplitPlan, rec: Recording, window_len: int, stride: int) -> list[Window]:
    """Assign every window of one recording to a part by a seeded permutation (reference only)."""
    n = rec.meta.n_packets
    starts = np.arange(0, n - window_len + 1, stride)
    if len(starts) == 0:
        return []
    seed = int(plan.params["seed"])
    _, f_val, f_test = plan.params["fractions"]
    order = np.random.default_rng([seed, zlib.crc32(rec.recording_id.encode())]).permutation(
        len(starts)
    )
    n_val = int(f_val * len(starts))
    n_test = int(f_test * len(starts))
    parts = {
        "val": order[:n_val],
        "test": order[n_val : n_val + n_test],
        "train": order[n_val + n_test :],
    }
    return [
        Window(
            window_id=f"{rec.recording_id}@{start}:{start + window_len}",
            recording_id=rec.recording_id,
            start=int(start),
            end=int(start) + window_len,
            split=part,
        )
        for part, idx in parts.items()
        for start in starts[idx]
    ]
