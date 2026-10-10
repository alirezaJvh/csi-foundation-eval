import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from csi_bench.adapters.splitters.random_window import RandomWindow
from csi_bench.adapters.splitters.time_block import TimeBlock
from csi_bench.config.experiment import config_from_dict
from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.splitplan import (
    check_split_role,
    compute_split_hash,
    load_split_plan,
    write_split_plan,
)
from csi_bench.core.windows import build_windows
from tests.contracts.splitter import check_splitter

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "splits/tor_vergata"


def _rec(rid: str, n: int = 500, room: str = "r1", label: int = 0) -> Recording:
    return Recording(
        csi=np.zeros((1, n, 1), dtype=np.complex64),
        antenna_ids=(0,),
        recording_id=rid,
        room=room,
        label=label,
        meta=RecordingMeta(n_packets=n, source_sha256=rid.ljust(64, "0")),
    )


def _recs() -> list[Recording]:
    return [_rec("d/a"), _rec("d/b", room="r2")]


def _split(seed: int = 0, **kw):
    return RandomWindow(RandomWindow.Params(seed=seed, **kw)).split(_recs(), "d")


def _assignment(plan, window_len=10, stride=5, rooms=("r1", "r2")):
    ws = build_windows(plan, _recs(), list(rooms), window_len, stride)
    return {w.window_id: w.split for w in ws}


def test_splitter_contract():
    check_splitter(RandomWindow(), _recs(), "d")


def test_plan_is_window_unit_and_differs_from_time_block():
    plan = _split()
    assert plan.unit == "window"
    assert plan.split_hash == compute_split_hash(plan)
    assert plan.split_hash != TimeBlock().split(_recs(), "d").split_hash


def test_same_seed_same_assignment_other_seed_differs():
    assert _split().split_hash == _split().split_hash
    assert _assignment(_split(0)) == _assignment(_split(0))
    assert _split(0).split_hash != _split(1).split_hash
    assert _assignment(_split(0)) != _assignment(_split(1))


def test_exact_fractions_per_recording_and_room_independence():
    plan = _split()
    both = _assignment(plan)
    only_r2 = _assignment(plan, rooms=("r2",))
    assert {k: v for k, v in both.items() if k.startswith("d/b")} == only_r2
    mine = [v for k, v in both.items() if k.startswith("d/a")]
    n = len(mine)  # (500 - 10) // 5 + 1 = 99
    assert n == 99
    assert mine.count("val") == int(0.2 * n)
    assert mine.count("test") == int(0.2 * n)
    assert mine.count("train") == n - 2 * int(0.2 * n)


def test_parts_are_disjoint_and_cover_every_start():
    for window_len, stride in ((10, 5), (40, 1), (7, 13)):
        ws = build_windows(_split(), _recs(), ["r1", "r2"], window_len, stride)
        ids = [w.window_id for w in ws]
        assert len(ids) == len(set(ids))
        for rec in _recs():
            starts = sorted(w.start for w in ws if w.recording_id == rec.recording_id)
            assert starts == list(range(0, rec.meta.n_packets - window_len + 1, stride))


def test_tiny_recording_class_without_windows_in_a_part():
    recs = [_rec("d/a"), _rec("d/tiny", n=4, label=1)]
    with pytest.raises(DataError, match="class 1 has no val windows"):
        RandomWindow().split(recs, "d")


def test_other_window_length_and_stride_still_assigned():
    plan = _split()
    ws = build_windows(plan, _recs(), ["r1"], 40, 1)
    assert len(ws) == 461
    assert {w.split for w in ws} == {"train", "val", "test"}
    assert all(w.end - w.start == 40 for w in ws)


def test_bad_fractions():
    with pytest.raises(ConfigError):
        RandomWindow(RandomWindow.Params(fractions=(0.5, 0.5, 0.5)))


def test_role_gate():
    check_split_role("window", "reference")
    check_split_role("block", "comparison")
    with pytest.raises(ConfigError):
        check_split_role("window", "comparison")


def _raw(role: str) -> dict:
    return {
        "dataset": "tor_vergata",
        "selection": {"rooms": ["r1"]},
        "window_len": 40,
        "extractors": [{"name": "amplitude_stats"}],
        "head": {"name": "logreg"},
        "role": role,
    }


def test_config_validation_rejects_comparison_on_window_plan(tmp_path, monkeypatch):
    from csi_bench.runner.discover import discover

    discover()
    plan = dataclasses.replace(_split(), dataset="tor_vergata")
    write_split_plan(
        dataclasses.replace(plan, split_hash=compute_split_hash(plan)), tmp_path / "splits"
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="role: reference"):
        config_from_dict(_raw("comparison"))
    config_from_dict(_raw("reference"))


def test_missing_plan_passes_validation(tmp_path, monkeypatch):
    from csi_bench.runner.discover import discover

    discover()
    monkeypatch.chdir(tmp_path)
    config_from_dict(_raw("comparison"))


def test_window_plan_roundtrip(tmp_path):
    plan = _split()
    path = write_split_plan(plan, tmp_path)
    assert load_split_plan(path, _recs()) == plan


def test_committed_block_plans_keep_hash_and_load():
    paths = sorted(COMMITTED.glob("*.json"))
    assert paths
    for path in paths:
        raw = json.loads(path.read_text())
        assert "unit" not in raw
        recs = [
            Recording(
                csi=np.zeros((1, 1, 1), dtype=np.complex64),
                antenna_ids=(0,),
                recording_id=rid,
                room="r",
                label=0,
                meta=RecordingMeta(
                    n_packets=max(end for _, end in raw["blocks"][rid].values()),
                    source_sha256=sha,
                ),
            )
            for rid, sha in raw["source_sha256"].items()
        ]
        plan = load_split_plan(path, recs)
        assert plan.unit == "block"
        assert compute_split_hash(plan) == plan.split_hash == path.stem
