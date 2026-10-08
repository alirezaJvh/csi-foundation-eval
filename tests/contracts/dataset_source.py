"""Reusable DatasetSource port contract; every dataset adapter's test calls it."""

import numpy as np

from csi_bench.core.ports import DatasetSource


def check_dataset_source(source: DatasetSource) -> None:
    recs = source.load()
    assert recs
    ids = [r.recording_id for r in recs]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert len(set(source.classes)) == len(source.classes)
    for r in recs:
        assert r.csi.dtype == np.complex64
        assert r.csi.ndim == 3
        assert len(r.antenna_ids) == r.csi.shape[0]
        assert r.meta.n_packets == r.csi.shape[1]
        assert 0 <= r.label < len(source.classes)
        assert r.room
        assert len(r.meta.source_sha256) == 64
        assert "/" in r.recording_id
