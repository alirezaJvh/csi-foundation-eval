"""Run an ordered preprocessing chain, fitting every step on train blocks only."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from csi_bench.core.contracts import Recording
from csi_bench.core.ports import PreprocessStep


def train_view(
    recordings: Sequence[Recording], blocks: Mapping[str, Mapping[str, tuple[int, int]]]
) -> list[Recording]:
    """Recordings cut down to their train block; val and test packets are not included."""
    views = []
    for rec in recordings:
        start, end = blocks[rec.recording_id]["train"]
        views.append(replace(rec, csi=rec.csi[:, start:end, :]))
    return views


def run_chain(
    steps: Sequence[PreprocessStep],
    recordings: Sequence[Recording],
    blocks: Mapping[str, Mapping[str, tuple[int, int]]],
) -> tuple[list[Recording], list[dict]]:
    """Fit step i on step i-1's output (train blocks), apply it to whole recordings.

    A step with `fit_scope == "whole_recording"` (reference runs only) is fitted on
    the full recordings instead. Returns the processed recordings and per-step state.
    """
    current = list(recordings)
    states: list[dict] = []
    for step in steps:
        whole = getattr(step, "fit_scope", "train") == "whole_recording"
        state = step.fit(current if whole else train_view(current, blocks))
        current = [step.transform(rec, state) for rec in current]
        states.append(state)
    return current, states
