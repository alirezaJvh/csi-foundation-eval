import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from csi_bench.core import registry
from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.splitplan import (
    ROOM_HOLDOUT,
    check_split_rooms,
    find_split,
    load_split_plan,
    read_plan_params,
    read_plan_splitter,
    write_split_plan,
)
from csi_bench.runner.cli import main
from csi_bench.runner.discover import discover
from tests.contracts.splitter import check_splitter

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/csi_proc_python/saved_csi"
COMMITTED = ROOT / "splits/tor_vergata"
R1_TO_R2 = "1e8688472a1f"
R2_TO_R1 = "c4faaf799ad5"


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _splitter(train=("r1",), test=("r2",), **params):
    from csi_bench.adapters.splitters.room_holdout import RoomHoldout

    return RoomHoldout(RoomHoldout.Params(train_rooms=list(train), test_rooms=list(test), **params))


def _rec(rid: str, room: str, label: int, n: int = 1000) -> Recording:
    return Recording(
        csi=np.zeros((1, 1, 1), dtype=np.complex64),
        antenna_ids=(0,),
        recording_id=rid,
        room=room,
        label=label,
        meta=RecordingMeta(n_packets=n, source_sha256=rid.ljust(64, "0")),
    )


def _recs() -> list[Recording]:
    return [_rec(f"d/{room}_{i}", room, i % 2, 1000 + i) for room in ("r1", "r2") for i in range(4)]


def test_blocks():
    plan = _splitter().split([_rec("d/a", "r1", 0), _rec("d/b", "r2", 0, 900)], "d")
    # usable = 1000 - 64 = 936, val = floor(0.25 * 936) = 234, train = 702
    assert plan.blocks["d/a"] == {"train": (0, 702), "val": (766, 1000), "test": (0, 0)}
    assert plan.blocks["d/b"] == {"train": (0, 0), "val": (0, 0), "test": (0, 900)}
    assert plan.splitter == ROOM_HOLDOUT
    assert plan.unit == "block"
    assert plan.params == {
        "train_rooms": ["r1"],
        "test_rooms": ["r2"],
        "fractions": [0.75, 0.25],
        "gap_packets": 64,
    }


def test_contract_and_order():
    check_splitter(_splitter(), _recs(), "d")
    assert _splitter().split(_recs(), "d") == _splitter().split(list(reversed(_recs())), "d")


def test_recording_in_neither_room_set_is_empty():
    recs = [*_recs(), _rec("d/r3_0", "r3", 0)]
    plan = _splitter().split(recs, "d")
    assert plan.blocks["d/r3_0"] == {"train": (0, 0), "val": (0, 0), "test": (0, 0)}
    check_splitter(_splitter(), recs, "d")


def test_directions_have_distinct_hashes():
    a = _splitter().split(_recs(), "d")
    b = _splitter(train=("r2",), test=("r1",)).split(_recs(), "d")
    assert a.split_hash != b.split_hash


@pytest.mark.parametrize(
    ("train", "test", "params"),
    [
        (("r1",), ("r1", "r2"), {}),
        ((), ("r2",), {}),
        (("r1",), (), {}),
        (("r1",), ("r2",), {"fractions": (0.5, 0.6)}),
        (("r1",), ("r2",), {"fractions": (1.0, 0.0)}),
        (("r1",), ("r2",), {"gap_packets": -1}),
    ],
)
def test_bad_params(train, test, params):
    with pytest.raises(ConfigError):
        _splitter(train, test, **params)


def test_listed_room_without_recordings():
    with pytest.raises(ConfigError, match="r3"):
        _splitter(test=("r2", "r3")).split(_recs(), "d")


def test_class_missing_from_the_test_room():
    recs = [r for r in _recs() if not (r.room == "r2" and r.label == 1)]
    with pytest.raises(DataError, match="class 1 has no test"):
        _splitter().split(recs, "d")


@pytest.mark.parametrize("n", [64, 65, 10])
def test_too_short(n):
    with pytest.raises(DataError, match="d/a"):
        _splitter().split([_rec("d/a", "r1", 0, n), _rec("d/b", "r2", 0)], "d")


def test_implicit_find_split_ignores_holdout_plans(tmp_path):
    from csi_bench.adapters.splitters.time_block import TimeBlock

    recs = _recs()
    block = write_split_plan(TimeBlock().split(recs, "d"), tmp_path)
    for train, test in ((("r1",), ("r2",)), (("r2",), ("r1",))):
        write_split_plan(_splitter(train, test).split(recs, "d"), tmp_path)
    assert len(list((tmp_path / "d").glob("*.json"))) == 3
    assert find_split("d", root=tmp_path) == block


def test_block_plan_files_omit_unit(tmp_path):
    from csi_bench.adapters.splitters.time_block import TimeBlock

    for plan in (TimeBlock().split(_recs(), "d"), _splitter().split(_recs(), "d")):
        assert "unit" not in json.loads(write_split_plan(plan, tmp_path).read_text())


def test_check_split_rooms():
    params = {"train_rooms": ["r1"], "test_rooms": ["r2"]}
    check_split_rooms(ROOM_HOLDOUT, params, ["r2", "r1"], "comparison")
    check_split_rooms("time_block", {}, ["r1"], "reference")
    with pytest.raises(ConfigError, match=r"\['r1'\].*\['r1', 'r2'\]"):
        check_split_rooms(ROOM_HOLDOUT, params, ["r1"], "comparison")
    with pytest.raises(ConfigError, match="role: comparison"):
        check_split_rooms(ROOM_HOLDOUT, params, ["r1", "r2"], "reference")


def test_direction_and_slug():
    assert rs.room_direction(["r1"], ["r2"]) == "r1→r2"
    config = {
        "extractors": [{"name": "amplitude_stats"}],
        "head": {"name": "logreg"},
        "selection": {"rooms": ["r1", "r2"]},
        "window_len": 40,
    }
    assert rs.run_slug(config) == "amplitude_stats-none-logreg-mixed-w40"
    assert rs.run_slug(config, "r2→r1") == "amplitude_stats-none-logreg-r2_to_r1-w40"


def test_committed_plans_are_the_two_directions():
    trains = {
        h: read_plan_params(COMMITTED / f"{h}.json")["train_rooms"] for h in (R1_TO_R2, R2_TO_R1)
    }
    assert trains == {R1_TO_R2: ["r1"], R2_TO_R1: ["r2"]}
    holdout = {p.stem for p in COMMITTED.glob("*.json") if read_plan_splitter(p) == ROOM_HOLDOUT}
    assert holdout == {R1_TO_R2, R2_TO_R1}


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_data_plans_and_committed_files():
    from csi_bench.adapters.datasets.tor_vergata import TorVergata

    recs = TorVergata(TorVergata.Params(root=str(REAL))).load()
    assert len(recs) == 19
    for (train, test), name in (((("r1",), ("r2",)), R1_TO_R2), ((("r2",), ("r1",)), R2_TO_R1)):
        check_splitter(_splitter(train, test), recs, "tor_vergata")
        plan = _splitter(train, test).split(recs, "tor_vergata")
        assert plan.split_hash == name
        assert load_split_plan(COMMITTED / f"{name}.json", recs) == plan
        for rec in recs:
            b = plan.blocks[rec.recording_id]
            if rec.room == test[0]:
                assert b == {"train": (0, 0), "val": (0, 0), "test": (0, rec.meta.n_packets)}
            else:
                assert b["test"] == (0, 0) and b["train"][1] > 0 and b["val"][1] > b["val"][0]


# --- end to end on a tiny fixture ----------------------------------------------------------

NAMES = [
    "r1_empty",
    "r1_sitting_1",
    "r1_standing_1",
    "r1_walking_1",
    "r2_empty_1",
    "r2_sit_1",
    "r2_standing_1",
    "r2_walk_1",
]
CONFIG = {
    "dataset": "tor_vergata",
    "selection": {"rooms": ["r1", "r2"]},
    "window_len": 40,
    "stride": 1,
    "extractors": [{"name": "amplitude_stats"}],
    "head": {"name": "logreg"},
    "seeds": [0],
    "role": "comparison",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("CSI_BENCH_WORKERS", "1")
    data = tmp_path / "data/csi_proc_python/saved_csi"
    data.mkdir(parents=True)
    for i, name in enumerate(NAMES):
        rng = np.random.default_rng(i)
        csi = (1 + rng.normal(0, 0.1, (4, 800, 3)) + 1j * rng.normal(0, 0.1, (4, 800, 3))).astype(
            np.complex64
        )
        np.save(data / f"{name}.npy", csi)
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.time_block import TimeBlock

    recs = TorVergata(TorVergata.Params(root=str(data))).load()
    write_split_plan(TimeBlock().split(recs, "tor_vergata"), tmp_path / "splits")
    hashes = {}
    for train, test in (("r1", "r2"), ("r2", "r1")):
        plan = _splitter((train,), (test,)).split(recs, "tor_vergata")
        write_split_plan(plan, tmp_path / "splits")
        hashes[f"{train}→{test}"] = plan.split_hash
    monkeypatch.chdir(tmp_path)
    return tmp_path, hashes


def _config_file(tmp_path: Path, **changes) -> str:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, **changes}, sort_keys=True), encoding="utf-8")
    return str(path)


def test_run_scores_val_on_train_room_and_test_on_test_room(project, capsys):
    tmp_path, hashes = project
    assert main(["run", _config_file(tmp_path, split=hashes["r1→r2"])]) == 0
    run_id = capsys.readouterr().out.strip()
    root = tmp_path / "runs"
    directory = rs.run_dir(run_id, root)
    assert directory.parent == root / "tor_vergata" / ROOM_HOLDOUT
    assert "-r1_to_r2-w40--" in directory.name
    metrics = json.loads(rs.metrics_path(run_id, root).read_text())
    for split in ("val", "test"):
        assert metrics["seeds"]["0"][split]["accuracy"] is not None
    rows = pq.read_table(rs.predictions_path(run_id, 0, root)).to_pylist()
    val = [r["recording_id"] for r in rows if r["split"] == "val"]
    test = [r["recording_id"] for r in rows if r["split"] == "test"]
    assert val and test
    assert all("/r1_" in rid for rid in val)
    assert all("/r2_" in rid for rid in test)


def test_rooms_must_match_the_plan(project, capsys):
    tmp_path, hashes = project
    code = main(["run", _config_file(tmp_path, split=hashes["r1→r2"], selection={"rooms": ["r1"]})])
    assert code != 0
    err = capsys.readouterr().err
    assert "['r1']" in err and "['r1', 'r2']" in err
    assert not rs.iter_run_dirs(tmp_path / "runs")


def test_config_without_split_keeps_the_time_block_plan(project, capsys):
    tmp_path, _ = project
    assert main(["run", _config_file(tmp_path, selection={"rooms": ["r1"]})]) == 0
    run_id = capsys.readouterr().out.strip()
    assert rs.run_dir(run_id, tmp_path / "runs").parent.name == "time_block"
