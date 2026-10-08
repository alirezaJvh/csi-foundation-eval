import numpy as np
import pytest

from csi_bench.core.contracts import Recording, RecordingMeta, SplitPlan
from csi_bench.core.errors import ConfigError
from csi_bench.core.windows import build_windows

GAP = 64


def _rec(rid: str, room: str, n: int = 700, label: int = 0) -> Recording:
    return Recording(
        csi=np.zeros((4, n, 3), dtype=np.complex64),
        antenna_ids=(0, 1, 2, 3),
        recording_id=rid,
        room=room,
        label=label,
        meta=RecordingMeta(n_packets=n, source_sha256=rid.ljust(64, "0")),
    )


def _plan(recs: list[Recording], blocks: dict[str, dict[str, tuple[int, int]]]) -> SplitPlan:
    return SplitPlan(
        dataset="d",
        splitter="time_block",
        params={},
        gap_packets=GAP,
        blocks=blocks,
        source_sha256={r.recording_id: r.meta.source_sha256 for r in recs},
        split_hash="x" * 12,
    )


def _simple() -> tuple[list[Recording], SplitPlan]:
    recs = [_rec("d/a", "r1"), _rec("d/b", "r2")]
    blocks = {
        "d/a": {"train": (0, 100), "val": (200, 300), "test": (400, 500)},
        "d/b": {"train": (0, 100), "val": (200, 300), "test": (400, 500)},
    }
    return recs, _plan(recs, blocks)


def test_ids_order_and_confinement():
    recs, plan = _simple()
    ws = build_windows(plan, recs, ["r1"], window_len=40, stride=40)
    ids = [w.window_id for w in ws]
    assert ids == sorted(ids, key=lambda i: (i.split("@")[0], int(i.split("@")[1].split(":")[0])))
    assert all(w.recording_id == "d/a" for w in ws)
    assert [w.split for w in ws] == ["train", "train", "val", "val", "test", "test"]
    assert [w.start for w in ws] == [0, 40, 200, 240, 400, 440]
    for w in ws:
        block = plan.blocks[w.recording_id][w.split]
        assert block[0] <= w.start < w.end <= block[1]
        assert w.window_id == f"{w.recording_id}@{w.start}:{w.end}"


def test_stride_and_parts():
    recs, plan = _simple()
    ws = build_windows(plan, recs, ["r1", "r2"], window_len=40, stride=10)
    # train: 0..60 step 10 -> 7; val: 200..260 -> 7; test: 400..460 -> 7; two recordings
    assert len(ws) == 2 * 3 * 7
    assert {w.recording_id for w in ws} == {"d/a", "d/b"}


def test_room_selection():
    recs, plan = _simple()
    assert {w.recording_id for w in build_windows(plan, recs, ["r2"], 40, 1)} == {"d/b"}
    assert len(build_windows(plan, recs, ["nope"], 40, 1)) == 0


def test_window_longer_than_gap_raises():
    recs, plan = _simple()
    with pytest.raises(ConfigError, match="gap_packets"):
        build_windows(plan, recs, ["r1"], GAP + 1, 1)


def test_bad_window_len_and_stride():
    recs, plan = _simple()
    with pytest.raises(ConfigError, match="window_len"):
        build_windows(plan, recs, ["r1"], 0, 1)
    with pytest.raises(ConfigError, match="stride"):
        build_windows(plan, recs, ["r1"], 10, 0)
