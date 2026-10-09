"""Reusable PreprocessStep port contract; every registered step's test calls it."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from csi_bench.core.contracts import Recording
from csi_bench.core.ports import PreprocessStep
from csi_bench.core.preprocess import train_view


def check_preprocess_step(
    step: PreprocessStep,
    recordings: Sequence[Recording],
    blocks: Mapping[str, Mapping[str, tuple[int, int]]],
) -> None:
    views = train_view(recordings, blocks)
    state = step.fit(views)
    json.dumps(state)

    # transform is pure: input unchanged, output new and repeatable
    rec = recordings[0]
    before = rec.csi.copy()
    out = step.transform(rec, state)
    assert out is not rec
    np.testing.assert_array_equal(rec.csi, before)
    assert out.csi.dtype == np.complex64
    assert out.csi.shape[0] == len(out.antenna_ids)
    np.testing.assert_array_equal(step.transform(rec, state).csi, out.csi)

    # fit never sees val/test: perturbing them leaves the state unchanged
    perturbed = []
    for r in recordings:
        csi = r.csi.copy()
        t0, t1 = blocks[r.recording_id]["train"]
        mask = np.ones(csi.shape[1], dtype=bool)
        mask[t0:t1] = False
        csi[:, mask, :] = csi[:, mask, :] * 7 + 3
        perturbed.append(replace(r, csi=csi))
    assert step.fit(train_view(perturbed, blocks)) == state
