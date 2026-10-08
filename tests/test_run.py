import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from csi_bench.core import registry
from csi_bench.core import results_schema as rs
from csi_bench.core.splitplan import write_split_plan
from csi_bench.runner.cli import main
from csi_bench.runner.run import run

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/csi_proc_python/saved_csi"
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
    "selection": {"rooms": ["r1"]},
    "window_len": 40,
    "stride": 1,
    "extractors": [{"name": "amplitude_stats"}],
    "head": {"name": "logreg"},
    "seeds": [0],
    "role": "comparison",
}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    yield
    registry.reset()


def _make_project(tmp_path: Path, constant: bool = False) -> Path:
    data = tmp_path / "data/csi_proc_python/saved_csi"
    data.mkdir(parents=True)
    for i, name in enumerate(NAMES):
        if constant:
            csi = np.ones((4, 800, 3), dtype=np.complex64)
        else:
            rng = np.random.default_rng(i)
            real = 1 + rng.normal(0, 0.1, size=(4, 800, 3))
            imag = rng.normal(0, 0.1, size=(4, 800, 3))
            csi = (real + 1j * imag).astype(np.complex64)
        np.save(data / f"{name}.npy", csi)
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.time_block import TimeBlock

    recs = TorVergata(TorVergata.Params(root=str(data))).load()
    plan = TimeBlock().split(recs, "tor_vergata")
    write_split_plan(plan, tmp_path / "splits")
    return tmp_path


def _config_file(tmp_path: Path, **changes) -> str:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, **changes}, sort_keys=True), encoding="utf-8")
    return str(path)


def _run_dirs(tmp_path: Path) -> list[Path]:
    runs = tmp_path / "runs"
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "_dataset")


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text())


def test_happy_path_writes_all_artifacts(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path)]) == 0
    run_id = capsys.readouterr().out.strip()
    root = tmp_path / "runs"
    for path in (
        rs.config_path(run_id, root),
        rs.status_path(run_id, root),
        rs.metrics_path(run_id, root),
        rs.provenance_path(run_id, root),
        rs.log_path(run_id, root),
        rs.predictions_path(run_id, 0, root),
    ):
        assert path.is_file(), path
    assert _status(rs.run_dir(run_id, root))["status"] == "done"

    predictions = pq.read_table(rs.predictions_path(run_id, 0, root))
    assert predictions.schema.equals(rs.PREDICTIONS_TABLE_SCHEMA)
    splits = {row["split"] for row in predictions.to_pylist()}
    assert splits == {"val", "test"}

    metrics = json.loads(rs.metrics_path(run_id, root).read_text())
    assert metrics["class_names"] == ["empty", "sitting", "standing", "walking"]
    for split in ("val", "test"):
        record = metrics["seeds"]["0"][split]
        assert 0.0 <= record["accuracy"] <= 1.0
        assert set(record["per_class"]) == set(metrics["class_names"])
        assert record["confusion_matrix"]

    provenance = json.loads(rs.provenance_path(run_id, root).read_text())
    assert provenance["split_hash"]
    assert "package_versions" in provenance
    assert provenance["device"] == "cpu"


def test_rerun_saved_config_same_run_id(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path)]) == 0
    first = capsys.readouterr().out.strip()
    saved = rs.config_path(first, tmp_path / "runs")
    assert main(["run", str(saved)]) == 0
    second = capsys.readouterr().out.strip()
    assert first == second
    assert len(_run_dirs(tmp_path)) == 1


def test_bad_field_fails_before_loading(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, window_len="x")]) == 2
    assert "window_len" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_unknown_adapter_lists_names(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, head={"name": "nope"})]) == 2
    assert "logreg" in capsys.readouterr().err


def test_window_too_long_no_done_run(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, window_len=100)]) == 2
    assert "gap_packets" in capsys.readouterr().err
    assert all(_status(d)["status"] != "done" for d in _run_dirs(tmp_path))


def test_failure_mid_run_records_failed(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path, constant=True)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path)]) == 2
    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1
    status = _status(dirs[0])
    assert status["status"] == "failed"
    assert "trace" in status


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_data_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    config = ROOT / "experiments/r1_amplitude_stats_logreg.yaml"
    assert run(config, tmp_path) == 0
    run_id = capsys.readouterr().out.strip()
    assert _status(rs.run_dir(run_id, tmp_path))["status"] == "done"
    metrics = json.loads(rs.metrics_path(run_id, tmp_path).read_text())
    assert set(metrics["seeds"]["0"]) == {"val", "test"}
