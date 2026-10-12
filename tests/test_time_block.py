import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from csi_bench.adapters.datasets.tor_vergata import TorVergata
from csi_bench.adapters.splitters.time_block import TimeBlock
from csi_bench.core import registry
from csi_bench.core.contracts import Recording, RecordingMeta, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.splitplan import (
    ROOM_HOLDOUT,
    compute_split_hash,
    load_split_plan,
    write_split_plan,
)
from csi_bench.runner.discover import discover
from tests.contracts.splitter import check_splitter

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/csi_proc_python/saved_csi"
COMMITTED = ROOT / "splits/tor_vergata"


def _rec(rid: str, n: int, label: int = 0, sha: str | None = None) -> Recording:
    return Recording(
        csi=np.zeros((1, 1, 1), dtype=np.complex64),
        antenna_ids=(0,),
        recording_id=rid,
        room="r1",
        label=label,
        meta=RecordingMeta(n_packets=n, source_sha256=sha or rid.ljust(64, "0")),
    )


def _recs(n: int = 1000) -> list[Recording]:
    return [_rec(f"d/a{i}", n, label=i % 2) for i in range(4)]


def test_defaults():
    plan = TimeBlock().split([_rec("d/a", 1000)], "d")
    assert plan.blocks["d/a"] == {"train": (0, 524), "val": (588, 762), "test": (826, 1000)}


def test_contract_and_repeat():
    check_splitter(TimeBlock(), _recs(), "d")
    a = TimeBlock().split(_recs(), "d")
    b = TimeBlock().split(list(reversed(_recs())), "d")
    assert a == b


def test_hash_changes_with_params():
    a = TimeBlock().split(_recs(), "d")
    b = TimeBlock(TimeBlock.Params(gap_packets=10)).split(_recs(), "d")
    assert a.split_hash != b.split_hash


@pytest.mark.parametrize("n", [128, 100, 0])
def test_too_short(n):
    with pytest.raises(DataError, match="d/a"):
        TimeBlock().split([_rec("d/a", n)], "d")


def test_empty_part_names_class_and_part():
    # usable = 10 -> val = test = 2 for 0.2, but 0.05 floors to 0
    p = TimeBlock.Params(gap_packets=0, fractions=(0.9, 0.05, 0.05))
    with pytest.raises(DataError, match=r"class 3.*val"):
        TimeBlock(p).split([_rec("d/a", 10, label=3)], "d")


def test_class_check_sums_over_recordings():
    p = TimeBlock.Params(gap_packets=0, fractions=(0.6, 0.2, 0.2))
    # 4 packets -> val = test = 0 alone; with a 10-packet sibling the class is fine
    TimeBlock(p).split([_rec("d/a", 4), _rec("d/b", 10)], "d")


@pytest.mark.parametrize("fr", [(0.5, 0.2, 0.2), (1.0, 0.0, 0.0), (0.6, 0.4, 0.0)])
def test_bad_fractions(fr):
    with pytest.raises(ConfigError):
        TimeBlock(TimeBlock.Params(fractions=fr))


def test_write_load_round_trip(tmp_path):
    recs = _recs()
    plan = TimeBlock().split(recs, "d")
    path = write_split_plan(plan, tmp_path)
    assert path == tmp_path / "d" / f"{plan.split_hash}.json"
    assert load_split_plan(path, recs) == plan
    assert write_split_plan(plan, tmp_path).read_bytes() == path.read_bytes()


def test_changed_sha_and_recording_set(tmp_path):
    recs = _recs()
    path = write_split_plan(TimeBlock().split(recs, "d"), tmp_path)
    changed = [
        dataclasses.replace(
            recs[0], meta=dataclasses.replace(recs[0].meta, source_sha256="x" * 64)
        ),
        *recs[1:],
    ]
    with pytest.raises(DataError, match="d/a0"):
        load_split_plan(path, changed)
    with pytest.raises(DataError, match="differ"):
        load_split_plan(path, recs[1:])


def test_tampered_plan(tmp_path):
    recs = _recs()
    path = write_split_plan(TimeBlock().split(recs, "d"), tmp_path)
    path.write_text(path.read_text().replace('"gap_packets":64', '"gap_packets":65'))
    with pytest.raises(DataError):
        load_split_plan(path, recs)


def _break(path: Path, how: str) -> None:
    text = path.read_text()
    if how == "missing":
        path.unlink()
    elif how == "not_json":
        path.write_text(text[:20])
    elif how == "missing_key":
        path.write_text(text.replace('"dataset"', '"datasett"'))
    elif how == "short_block":
        path.write_text(text.replace('"train":[0,', '"train":[', 1))
    elif how == "blocks_list":
        path.write_text(json.dumps({**json.loads(text), "blocks": []}))
    elif how == "root_list":
        path.write_text("[]")


@pytest.mark.parametrize(
    "how", ["missing", "not_json", "missing_key", "short_block", "blocks_list", "root_list"]
)
def test_unreadable_plan(tmp_path, how):
    recs = _recs()
    path = write_split_plan(TimeBlock().split(recs, "d"), tmp_path)
    _break(path, how)
    with pytest.raises(DataError, match="unreadable"):
        load_split_plan(path, recs)


def test_committed_plans_are_consistent():
    paths = sorted(COMMITTED.glob("*.json"))
    assert paths
    for path in paths:
        raw = json.loads(path.read_text())
        if raw.get("unit") == "window":
            continue  # random_window plans are checked in test_random_window.py
        plan = SplitPlan(
            dataset=raw["dataset"],
            splitter=raw["splitter"],
            params=raw["params"],
            gap_packets=raw["gap_packets"],
            blocks={r: {k: (v[0], v[1]) for k, v in b.items()} for r, b in raw["blocks"].items()},
            source_sha256=raw["source_sha256"],
            split_hash=raw["split_hash"],
        )
        assert path.stem == plan.split_hash == compute_split_hash(plan)
        assert set(plan.blocks) == set(plan.source_sha256)
        for b in plan.blocks.values():
            spans = [b["train"], b["val"], b["test"]]
            assert all(0 <= s <= e for s, e in spans)
            if plan.splitter != ROOM_HOLDOUT:
                assert all(s < e for s, e in spans)  # only room_holdout has empty parts
            filled = [span for span in spans if span[0] < span[1]]
            assert filled
            for (_, prev_end), (next_start, _) in zip(filled, filled[1:], strict=False):
                assert next_start - prev_end >= plan.gap_packets


@pytest.fixture
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def test_registered_via_discovery(monkeypatch, clean_registry):
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters.splitters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    assert registry.get("splitter", "time_block").__name__ == "TimeBlock"


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_data_plan_and_committed_file():
    recs = TorVergata(TorVergata.Params(root=str(REAL))).load()
    assert len(recs) == 19
    check_splitter(TimeBlock(), recs, "tor_vergata")
    plan = TimeBlock().split(recs, "tor_vergata")
    committed = COMMITTED / f"{plan.split_hash}.json"
    assert committed.is_file(), f"generate with write_split_plan: {committed}"
    assert load_split_plan(committed, recs) == plan
