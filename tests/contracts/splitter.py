"""Reusable Splitter port contract; every splitter's test calls it."""

from csi_bench.core.contracts import Recording
from csi_bench.core.ports import Splitter


def check_splitter(splitter: Splitter, recordings: list[Recording], dataset: str) -> None:
    plan = splitter.split(recordings, dataset)
    assert plan == splitter.split(recordings, dataset)
    assert plan.split_hash == splitter.split(recordings, dataset).split_hash
    assert plan.dataset == dataset
    assert set(plan.blocks) == {r.recording_id for r in recordings}
    for rec in recordings:
        assert plan.source_sha256[rec.recording_id] == rec.meta.source_sha256
        b = plan.blocks[rec.recording_id]
        spans = [b["train"], b["val"], b["test"]]
        for start, end in spans:
            assert 0 <= start < end <= rec.meta.n_packets
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert next_start - prev_end >= plan.gap_packets
