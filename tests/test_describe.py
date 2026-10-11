import sys
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from csi_bench.core import registry
from csi_bench.core.results_schema import DATASET_TABLE_SCHEMA, dataset_table_path
from csi_bench.core.splitplan import write_split_plan
from csi_bench.runner.cli import main

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/csi_proc_python/saved_csi"
NAMES = ["r1_empty", "r1_sitting_1", "r2_walk_1", "r2_standing_1"]


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    yield
    registry.reset()


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/csi_proc_python/saved_csi"
    data.mkdir(parents=True)
    for i, n in enumerate(NAMES):
        np.save(data / f"{n}.npy", np.full((4, 1000 + i, 3), i, dtype=np.complex64))
    return tmp_path


def _plan(gap: int = 64):
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.time_block import TimeBlock

    recs = TorVergata().load()
    plan = TimeBlock(TimeBlock.Params(gap_packets=gap)).split(recs, "tor_vergata")
    return write_split_plan(plan)


def _rows(path):
    return pq.read_table(path).to_pylist()


def test_describe(project, capsys):
    _plan()
    assert main(["describe", "tor_vergata"]) == 0
    out = capsys.readouterr().out
    assert "r1_sitting_1" in out
    table = pq.read_table(dataset_table_path("tor_vergata"))
    assert table.schema.equals(DATASET_TABLE_SCHEMA)
    rows = table.to_pylist()
    rec = [r for r in rows if r["scope"] == "recording"]
    assert [r["recording_id"] for r in rec] == sorted(r["recording_id"] for r in rec)
    assert len(rec) == 4
    assert len([r for r in rows if r["scope"] == "class"]) == 4
    assert len([r for r in rows if r["scope"] == "room"]) == 2
    (total,) = [r for r in rows if r["scope"] == "total"]
    for col in ("n_packets", "train", "val", "test"):
        assert total[col] == sum(r[col] for r in rec)
    for r in rec:
        assert r["train"] + r["val"] + r["test"] <= r["n_packets"]
    sitting = next(r for r in rows if r["scope"] == "class" and r["class"] == "sitting")
    assert sitting["n_packets"] == 1001
    assert sitting["recording_id"] is None
    assert next(r for r in rec if "sitting" in r["recording_id"])["downsample_factor"] == 2
    assert len(out.splitlines()) == len(rows) + 1


def test_rerun_identical(project):
    _plan()
    assert main(["describe", "tor_vergata"]) == 0
    first = dataset_table_path("tor_vergata").read_bytes()
    assert main(["describe", "tor_vergata"]) == 0
    assert dataset_table_path("tor_vergata").read_bytes() == first


def test_unknown_dataset(project, capsys):
    assert main(["describe", "nope"]) == 2
    assert "tor_vergata" in capsys.readouterr().err
    assert not Path("runs").exists()


def test_no_split(project, capsys):
    assert main(["describe", "tor_vergata"]) == 2
    assert "no split plan" in capsys.readouterr().err
    assert not Path("runs").exists()


def test_several_splits(project, capsys):
    a = _plan(64)
    b = _plan(32)
    assert main(["describe", "tor_vergata"]) == 2
    err = capsys.readouterr().err
    assert a.stem in err and b.stem in err
    assert not Path("runs").exists()
    assert main(["describe", "tor_vergata", "--split", b.stem]) == 0
    assert _rows(dataset_table_path("tor_vergata"))[0]["split_hash"] == b.stem
    assert main(["describe", "tor_vergata", "--split", "bogus"]) == 2


def test_stale_split(project, capsys):
    _plan()
    np.save(
        project / "data/csi_proc_python/saved_csi/r1_empty.npy",
        np.ones((4, 1000, 3), dtype=np.complex64),
    )
    assert main(["describe", "tor_vergata"]) == 2
    assert "source changed" in capsys.readouterr().err
    assert not Path("runs").exists()


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_data(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("csi_bench.runner.cli.dataset_table_path", lambda d: tmp_path / f"{d}.pq")
    assert main(["describe", "tor_vergata"]) == 0
    rows = _rows(tmp_path / "tor_vergata.pq")
    assert len([r for r in rows if r["scope"] == "recording"]) == 19
    assert len([r for r in rows if r["scope"] == "total"]) == 1


def test_console_script_entry_point():
    scripts = {e.name: e.value for e in entry_points(group="console_scripts")}
    assert scripts.get("csi-bench") == "csi_bench.runner.cli:main"


def test_absent_class_omitted(project):
    (project / "data/csi_proc_python/saved_csi/r2_walk_1.npy").unlink()
    _plan()
    assert main(["describe", "tor_vergata"]) == 0
    class_rows = [r for r in _rows(dataset_table_path("tor_vergata")) if r["scope"] == "class"]
    assert {r["class"] for r in class_rows} == {"empty", "sitting", "standing"}
    assert all(r["n_packets"] > 0 for r in class_rows)


def test_describe_window_plan_has_no_part_counts(project, capsys):
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.random_window import RandomWindow

    plan = RandomWindow().split(TorVergata().load(), "tor_vergata")
    write_split_plan(plan)
    assert main(["describe", "tor_vergata"]) != 0  # a window plan is never implicit
    assert main(["describe", "tor_vergata", "--split", plan.split_hash]) == 0
    rows = _rows(dataset_table_path("tor_vergata"))
    assert all(r[p] is None for r in rows for p in ("train", "val", "test"))
    (total,) = [r for r in rows if r["scope"] == "total"]
    assert total["n_packets"] == sum(1000 + i for i in range(4))
