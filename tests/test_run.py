import json
import logging
import re
import sys
import threading
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


@pytest.fixture(autouse=True)
def single_worker(monkeypatch):
    """Spies on extractors need the in-process path; parallel tests pass workers explicitly."""
    monkeypatch.setenv("CSI_BENCH_WORKERS", "1")


def _make_project(tmp_path: Path, constant: bool = False, packets: int = 800) -> Path:
    data = tmp_path / "data/csi_proc_python/saved_csi"
    data.mkdir(parents=True)
    for i, name in enumerate(NAMES):
        if constant:
            csi = np.ones((4, packets, 3), dtype=np.complex64)
        else:
            rng = np.random.default_rng(i)
            real = 1 + rng.normal(0, 0.1, size=(4, packets, 3))
            imag = rng.normal(0, 0.1, size=(4, packets, 3))
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
    assert set(provenance["git"]) == {"commit", "dirty", "diff"}
    assert isinstance(provenance["git"]["dirty"], bool)
    status = _status(rs.run_dir(run_id, root))
    assert status["started_at"] <= status["finished_at"]


def test_status_started_at_is_stable(tmp_path, monkeypatch):
    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen: list[tuple[str, str]] = []
    original = run_module._write_status

    def spy(run_id, root, status, **extra):
        if "started_at" in extra:
            seen.append((status, extra["started_at"]))
        return original(run_id, root, status, **extra)

    monkeypatch.setattr(run_module, "_write_status", spy)
    assert run_module.run(_config_file(tmp_path), tmp_path / "runs") == 0
    started = [s for st, s in seen if st in ("pending", "running", "done")]
    assert started and len(set(started)) == 1


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
    assert "started_at" in status


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_data_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    config = ROOT / "experiments/r1_amplitude_stats_logreg.yaml"
    assert run(config, tmp_path, tmp_path / "cache") == 0
    run_id = capsys.readouterr().out.strip()
    assert _status(rs.run_dir(run_id, tmp_path))["status"] == "done"
    metrics = json.loads(rs.metrics_path(run_id, tmp_path).read_text())
    assert set(metrics["seeds"]["0"]) == {"val", "test", "feature_names"}


def test_provenance_git_state(monkeypatch):
    from csi_bench.config.experiment import ExperimentConfig
    from csi_bench.runner import run as run_module

    config = ExperimentConfig.model_validate(CONFIG)
    outputs = {
        ("status", "--porcelain"): " M file.py",
        ("rev-parse", "HEAD"): "abc123",
        ("diff", "HEAD"): "the diff",
    }
    monkeypatch.setattr(run_module, "_git", lambda args: outputs[tuple(args)])
    git = run_module.collect_provenance(config, "id", "split", 1.0, [])["git"]
    assert git == {"commit": "abc123", "dirty": True, "diff": "the diff"}

    outputs[("status", "--porcelain")] = ""
    git = run_module.collect_provenance(config, "id", "split", 1.0, [])["git"]
    assert git == {"commit": "abc123", "dirty": False, "diff": None}


def test_invalid_config_never_loads_data(tmp_path, monkeypatch, capsys):
    from csi_bench.adapters.datasets.tor_vergata import TorVergata

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        TorVergata, "load", lambda self: pytest.fail("data must not be loaded"), raising=True
    )
    for changes in (
        {"window_len": 0},
        {"dataset": "nope"},
        {"seeds": []},
        {"preprocessing": ["nope"]},
        {"preprocessing": [{"antennas": {"wrong": 1}}]},
    ):
        assert main(["run", _config_file(tmp_path, **changes)]) == 2
    assert _run_dirs(tmp_path) == []


def test_unknown_room_is_a_config_error(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, selection={"rooms": ["r9"]})]) == 2
    assert "selection.rooms" in capsys.readouterr().err


def test_preprocessing_chain_end_to_end(tmp_path, monkeypatch, capsys):
    from csi_bench.adapters.extractors.amplitude_stats import AmplitudeStats

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen = []
    original = AmplitudeStats.extract

    def spy(self, slices):
        seen.extend(s.shape[0] for s in slices)
        return original(self, slices)

    monkeypatch.setattr(AmplitudeStats, "extract", spy)
    config = _config_file(tmp_path, preprocessing=[{"antennas": [0, 1]}, {"global_rescale": {}}])
    assert main(["run", config]) == 0
    run_id = capsys.readouterr().out.strip()
    root = tmp_path / "runs"
    assert _status(rs.run_dir(run_id, root))["status"] == "done"
    assert set(seen) == {2}
    steps = json.loads(rs.provenance_path(run_id, root).read_text())["preprocessing"]
    assert [s["name"] for s in steps] == ["antennas", "global_rescale"]
    assert steps[0]["fitted"] == {"ids": [0, 1]}
    assert steps[1]["fitted"]["mean_amplitude"] > 0

    # the mean is the train-block mean of the two selected antennas in room r1
    plan_file = next((tmp_path / "splits/tor_vergata").glob("*.json"))
    plan = json.loads(plan_file.read_text())
    total = count = 0
    for name in NAMES:
        if not name.startswith("r1"):
            continue
        start, end = plan["blocks"][f"tor_vergata/{name}"]["train"]
        csi = np.load(tmp_path / "data/csi_proc_python/saved_csi" / f"{name}.npy")[:2, start:end]
        total += np.abs(csi).astype(np.float64).sum()
        count += csi.size
    assert steps[1]["fitted"]["mean_amplitude"] == pytest.approx(total / count, rel=1e-6)


def test_missing_antenna_is_config_error(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, preprocessing=[{"antennas": [0, 7]}])]) == 2
    assert "antennas" in capsys.readouterr().err


def _run_id(capsys) -> str:
    return capsys.readouterr().out.strip()


def test_multi_seed_run(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, seeds=[0, 1, 2])]) == 0
    run_id = _run_id(capsys)
    root = tmp_path / "runs"
    for seed in (0, 1, 2):
        assert rs.predictions_path(run_id, seed, root).is_file()
    metrics = json.loads(rs.metrics_path(run_id, root).read_text())
    assert set(metrics["seeds"]) == {"0", "1", "2"}
    assert len(metrics["mean_std"]["test"]["macro_f1"]) == 2


def test_runtime_provenance(tmp_path, monkeypatch, capsys):
    import torch

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path)]) == 0
    run_id = _run_id(capsys)
    runtime = json.loads(rs.provenance_path(run_id, tmp_path / "runs").read_text())["runtime"]
    assert runtime == {
        "device": "cpu",
        "torch_threads": 1,
        "deterministic": True,
        "torch_version": torch.__version__,
    }


def _spy_fits(monkeypatch):
    from csi_bench.adapters.heads.logreg import LogReg

    fitted: list[int] = []
    original = LogReg.fit

    def spy(self, *args):
        fitted.append(self.seed)
        return original(self, *args)

    monkeypatch.setattr(LogReg, "fit", spy)
    return fitted


def test_add_a_seed_runs_only_the_new_seed(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, seeds=[0, 1, 2])]) == 0
    run_id = _run_id(capsys)
    root = tmp_path / "runs"
    old = {s: rs.predictions_path(run_id, s, root).read_bytes() for s in (0, 1, 2)}
    old_mtime = {s: rs.predictions_path(run_id, s, root).stat().st_mtime_ns for s in (0, 1, 2)}
    before = json.loads(rs.metrics_path(run_id, root).read_text())

    fitted = _spy_fits(monkeypatch)
    assert main(["run", _config_file(tmp_path, seeds=[0, 1, 2, 3])]) == 0
    assert _run_id(capsys) == run_id
    assert fitted == [3]
    for s in (0, 1, 2):
        path = rs.predictions_path(run_id, s, root)
        assert path.read_bytes() == old[s]
        assert path.stat().st_mtime_ns == old_mtime[s]
    after = json.loads(rs.metrics_path(run_id, root).read_text())
    assert set(after["seeds"]) == {"0", "1", "2", "3"}
    for s in ("0", "1", "2"):
        assert after["seeds"][s] == before["seeds"][s]
    assert rs.predictions_path(run_id, 3, root).is_file()
    for split, stats in after["mean_std"].items():
        for metric, (mean, std) in stats.items():
            values = [after["seeds"][str(s)][split][metric] for s in range(4)]
            assert mean == pytest.approx(np.mean(values))
            assert std == pytest.approx(np.std(values))
    assert _status(rs.run_dir(run_id, root))["status"] == "done"


def test_same_seeds_is_a_no_op(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = _config_file(tmp_path, seeds=[0, 1])
    assert main(["run", config]) == 0
    run_id = _run_id(capsys)
    metrics_file = rs.metrics_path(run_id, tmp_path / "runs")
    stamp = metrics_file.stat().st_mtime_ns
    fitted = _spy_fits(monkeypatch)
    assert main(["run", config]) == 0
    assert _run_id(capsys) == run_id
    assert fitted == []
    assert metrics_file.stat().st_mtime_ns == stamp


def test_deleted_seed_reruns(tmp_path, monkeypatch, capsys):
    import shutil

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = _config_file(tmp_path, seeds=[0, 1])
    assert main(["run", config]) == 0
    run_id = _run_id(capsys)
    root = tmp_path / "runs"
    shutil.rmtree(rs.predictions_path(run_id, 1, root).parent)
    fitted = _spy_fits(monkeypatch)
    assert main(["run", config]) == 0
    assert fitted == [1]
    assert rs.predictions_path(run_id, 1, root).is_file()


@pytest.mark.parametrize(
    "head",
    [
        {"name": "svm", "params": {"C_grid": [1.0]}},
        {"name": "knn", "params": {"k_grid": [3]}},
        {"name": "random_forest", "params": {"n_estimators_grid": [10], "max_depth_grid": [3]}},
        {"name": "mlp", "params": {"alpha_grid": [1e-3], "max_iter": 50}},
        {"name": "logreg"},
    ],
)
def test_each_head_end_to_end(tmp_path, monkeypatch, capsys, head):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, head=head)]) == 0
    run_id = _run_id(capsys)
    metrics = json.loads(rs.metrics_path(run_id, tmp_path / "runs").read_text())
    assert set(metrics["seeds"]["0"]) == {"val", "test", "feature_names"}


def test_bad_head_grid_is_config_error(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    head = {"name": "knn", "params": {"k_grid": []}}
    assert main(["run", _config_file(tmp_path, head=head)]) == 2
    assert "k_grid" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_run_without_a_split_plan_is_a_data_error(tmp_path, monkeypatch, capsys):
    import shutil

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(tmp_path / "splits")
    assert main(["run", _config_file(tmp_path)]) == 2
    assert "no split plan" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_run_with_several_split_plans_is_a_data_error(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "splits/tor_vergata"
    plan = next(folder.glob("*.json"))
    (folder / "second.json").write_text(plan.read_text())
    assert main(["run", _config_file(tmp_path)]) == 2
    assert "several split plans" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_part_shorter_than_window_len_is_config_error(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path, packets=300)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path)]) == 2
    assert re.search(r"no val windows .* window_len", capsys.readouterr().err)


def _spy_extract(monkeypatch):
    from csi_bench.adapters.extractors.amplitude_stats import AmplitudeStats

    calls: list[int] = []
    original = AmplitudeStats.extract

    def spy(self, slices):
        calls.append(len(slices))
        return original(self, slices)

    monkeypatch.setattr(AmplitudeStats, "extract", spy)
    return calls


def _provenance_cache(run_id, root):
    return json.loads(rs.provenance_path(run_id, root).read_text())["feature_cache"]


def _metrics(run_id, root):
    return json.loads(rs.metrics_path(run_id, root).read_text())["seeds"]


def test_head_only_change_loads_features_from_cache(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_extract(monkeypatch)
    cache = tmp_path / "cache"
    root = tmp_path / "runs"
    assert run(_config_file(tmp_path), root, cache) == 0
    first = _run_id(capsys)
    assert len(calls) == 1
    entries = _provenance_cache(first, root)
    assert [(e["extractor"], e["hit"]) for e in entries] == [("amplitude_stats", False)]
    assert (cache / f"{entries[0]['key']}.npz").is_file()
    assert list(cache.glob("*.tmp")) == []

    head = {"name": "knn", "params": {"k_grid": [3]}}
    assert run(_config_file(tmp_path, head=head), root, cache) == 0
    second = _run_id(capsys)
    assert second != first
    assert len(calls) == 1
    entries2 = _provenance_cache(second, root)
    assert entries2[0]["hit"] is True
    assert entries2[0]["key"] == entries[0]["key"]


def test_cold_cache_gives_identical_metrics(tmp_path, monkeypatch, capsys):
    import shutil

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_extract(monkeypatch)
    cache = tmp_path / "cache"
    assert run(_config_file(tmp_path), tmp_path / "runs", cache) == 0
    run_id = _run_id(capsys)
    cached = _metrics(run_id, tmp_path / "runs")
    shutil.rmtree(tmp_path / "runs")
    assert run(_config_file(tmp_path), tmp_path / "runs", cache) == 0
    assert len(calls) == 1
    assert _metrics(run_id, tmp_path / "runs") == cached
    shutil.rmtree(tmp_path / "runs")
    shutil.rmtree(cache)
    assert run(_config_file(tmp_path), tmp_path / "runs", cache) == 0
    assert len(calls) == 2
    assert _metrics(run_id, tmp_path / "runs") == cached


def test_corrupt_cache_entry_is_recomputed_and_replaced(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_extract(monkeypatch)
    cache = tmp_path / "cache"
    assert run(_config_file(tmp_path), tmp_path / "runs", cache) == 0
    first = _run_id(capsys)
    cached = _metrics(first, tmp_path / "runs")
    (entry,) = cache.glob("*.npz")
    entry.write_bytes(entry.read_bytes()[:30])
    head = {"name": "knn", "params": {"k_grid": [3]}}
    assert run(_config_file(tmp_path, head=head), tmp_path / "runs", cache) == 0
    second = _run_id(capsys)
    assert len(calls) == 2
    assert _provenance_cache(second, tmp_path / "runs")[0]["hit"] is False
    assert run(_config_file(tmp_path), tmp_path / "runs" / "again", cache) == 0
    assert len(calls) == 2
    assert _metrics(first, tmp_path / "runs" / "again") == cached


def test_failed_cache_write_fails_the_run(tmp_path, monkeypatch, capsys):
    import os

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"

    def boom(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    assert run(_config_file(tmp_path), tmp_path / "runs", cache) == 2
    assert "disk full" in capsys.readouterr().err
    (directory,) = _run_dirs(tmp_path)
    assert _status(directory)["status"] == "failed"
    assert list(cache.iterdir()) == []


def test_two_families_are_concatenated_with_unique_names(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = _config_file(
        tmp_path,
        window_len=56,
        preprocessing=[{"antennas": [0, 1]}],
        extractors=[{"name": "amplitude_stats"}, {"name": "paper_beamforming"}],
    )
    assert main(["run", config]) == 0
    run_id = capsys.readouterr().out.strip()
    hits = _provenance_cache(run_id, tmp_path / "runs")
    assert [h["extractor"] for h in hits] == ["amplitude_stats", "paper_beamforming"]


def test_duplicate_feature_names_across_extractors_fail_the_run(tmp_path, monkeypatch, capsys):
    from csi_bench.adapters.extractors import amplitude_stats, paper_beamforming

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paper_beamforming, "NAMES", amplitude_stats.NAMES[: len(paper_beamforming.NAMES)]
    )
    config = _config_file(
        tmp_path,
        window_len=56,
        preprocessing=[{"antennas": [0, 1]}],
        extractors=[{"name": "amplitude_stats"}, {"name": "paper_beamforming"}],
    )
    assert main(["run", config]) != 0
    assert "duplicate feature names" in capsys.readouterr().err


FAMILIES = {
    "window_len": 56,
    "preprocessing": [{"antennas": [0, 1]}],
    "extractors": [{"name": "amplitude_stats"}, {"name": "paper_beamforming"}],
}
ALL_NAMES = [
    "amplitude_stats.mean_power",
    "amplitude_stats.std",
    "amplitude_stats.flatness",
    "paper_beamforming.doppler_theta_var",
    "paper_beamforming.doppler_integral_mean",
    "paper_beamforming.distance_theta_var",
    "paper_beamforming.distance_mean",
    "paper_beamforming.flatness_theta_var",
    "paper_beamforming.flatness_mean",
]


def _spy_head_inputs(monkeypatch):
    from csi_bench.adapters.heads.logreg import LogReg

    seen: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    original = LogReg.fit

    def spy(self, train, y_train, val, y_val):
        seen.append((train.feature_names, val.feature_names, ()))
        return original(self, train, y_train, val, y_val)

    monkeypatch.setattr(LogReg, "fit", spy)
    return seen


def test_manual_selection_feeds_exactly_the_chosen_columns(tmp_path, monkeypatch, capsys):
    from csi_bench.adapters.heads.logreg import LogReg

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    tests_seen: list[tuple[str, ...]] = []
    original_predict = LogReg.predict

    def spy_predict(self, features):
        tests_seen.append(features.feature_names)
        return original_predict(self, features)

    monkeypatch.setattr(LogReg, "predict", spy_predict)
    seen = _spy_head_inputs(monkeypatch)
    selection = {"manual": ["paper_beamforming", "amplitude_stats.mean_power"]}
    config = _config_file(tmp_path, seeds=[0, 1], feature_selection=selection, **FAMILIES)
    assert main(["run", config]) == 0
    run_id = _run_id(capsys)
    expected = [ALL_NAMES[0], *ALL_NAMES[3:]]
    assert [s[0] for s in seen] == [tuple(expected)] * 2
    assert [s[1] for s in seen] == [tuple(expected)] * 2
    assert set(tests_seen) == {tuple(expected)}
    seeds = _metrics(run_id, tmp_path / "runs")
    assert seeds["0"]["feature_names"] == expected
    assert seeds["1"]["feature_names"] == expected


def test_null_selection_uses_all_columns_and_records_them(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, **FAMILIES)]) == 0
    first = _run_id(capsys)
    assert _metrics(first, tmp_path / "runs")["0"]["feature_names"] == ALL_NAMES
    assert main(["run", _config_file(tmp_path, feature_selection=None, **FAMILIES)]) == 0
    assert _run_id(capsys) == first


def test_selection_changes_run_id(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, **FAMILIES)]) == 0
    plain = _run_id(capsys)
    selection = {"manual": ["amplitude_stats"]}
    assert main(["run", _config_file(tmp_path, feature_selection=selection, **FAMILIES)]) == 0
    assert _run_id(capsys) != plain


def test_mrmr_picks_k_on_val_and_records_a_prefix(tmp_path, monkeypatch, capsys):
    from csi_bench.adapters.heads.logreg import LogReg
    from csi_bench.adapters.selectors.mrmr import Mrmr

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rankings: list[tuple[str, ...]] = []
    original_fit = Mrmr.fit

    def spy_fit(self, *args):
        original_fit(self, *args)
        rankings.append(self.candidates[-1])

    monkeypatch.setattr(Mrmr, "fit", spy_fit)
    scores = iter([0.1, 0.9, 0.9])  # k=2 loses, k=3 wins, k=5 ties and loses
    original_head_fit = LogReg.fit

    def fake_score(self, *args):
        original_head_fit(self, *args)
        self.val_score = next(scores)

    monkeypatch.setattr(LogReg, "fit", fake_score)
    selection = {"mrmr": {"k": [2, 3, 5]}}
    assert main(["run", _config_file(tmp_path, feature_selection=selection, **FAMILIES)]) == 0
    run_id = _run_id(capsys)
    names = _metrics(run_id, tmp_path / "runs")["0"]["feature_names"]
    assert len(names) == 3
    assert tuple(names) == rankings[0][:3]
    assert set(names) <= set(ALL_NAMES)
    assert _metrics(run_id, tmp_path / "runs")["0"]["selection"] == {
        "chosen": 1,
        "candidates": [
            {"n_features": 2, "val_score": 0.1},
            {"n_features": 3, "val_score": 0.9},
            {"n_features": 5, "val_score": 0.9},
        ],
    }


def test_mrmr_k_too_large_fails_before_any_seed(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    fitted = _spy_fits(monkeypatch)
    selection = {"mrmr": {"k": [2, 10]}}
    config = _config_file(tmp_path, feature_selection=selection, **FAMILIES)
    assert main(["run", config]) == 2
    assert "exceeds" in capsys.readouterr().err
    assert fitted == []


def test_unknown_manual_entry_fails_before_any_seed(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    fitted = _spy_fits(monkeypatch)
    config = _config_file(tmp_path, feature_selection={"manual": ["nope"]}, **FAMILIES)
    assert main(["run", config]) == 2
    err = capsys.readouterr().err
    assert "nope" in err and "amplitude_stats.std" in err
    assert fitted == []


def test_selector_sees_only_train_and_val_rows(tmp_path, monkeypatch, capsys):
    from pydantic import BaseModel, ConfigDict

    from csi_bench.adapters.selectors.manual import Manual

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    log: list[tuple[str, int]] = []
    seen_rows: set[bytes] = set()
    transformed: list[np.ndarray] = []

    class SpyParams(BaseModel):
        model_config = ConfigDict(extra="forbid")

    @registry.register("selector", "spy")
    class Spy:
        VERSION = 1
        Params = SpyParams

        def __init__(self, params):
            self._inner = Manual(Manual.Params(columns=["amplitude_stats"]))

        candidates = property(lambda self: self._inner.candidates)
        selected_names = property(lambda self: self._inner.selected_names)

        def validate(self, names):
            self._inner.validate(names)

        def fit(self, train, y_train, val, y_val):
            log.append(("fit", len(train.values) + len(val.values)))
            seen_rows.update(map(bytes, np.vstack([train.values, val.values])))
            self._inner.fit(train, y_train, val, y_val)

        def choose(self, index):
            self._inner.choose(index)

        def transform(self, matrix):
            log.append(("transform", len(matrix.values)))
            transformed.append(matrix.values)
            return self._inner.transform(matrix)

    config = _config_file(tmp_path, feature_selection={"name": "spy"}, **FAMILIES)
    assert main(["run", config]) == 0
    run_id = _run_id(capsys)
    rows = pq.read_table(rs.predictions_path(run_id, 0, tmp_path / "runs")).to_pylist()
    n_val = sum(r["split"] == "val" for r in rows)
    n_test = sum(r["split"] == "test" for r in rows)
    (fit_rows,) = [n for kind, n in log if kind == "fit"]
    n_train = fit_rows - n_val  # fit saw exactly train + val rows
    assert n_train > 0
    # test rows are only ever transformed, once, after the choice is made
    transforms = [n for kind, n in log if kind == "transform"]
    assert transforms == [n_train, n_val, n_val, n_test]
    # the rows handed to fit are exactly the train and val rows, never a test row
    assert {bytes(r) for m in transformed[:2] for r in m} == seen_rows
    assert not {bytes(r) for r in transformed[-1]} & seen_rows


def test_selector_only_change_hits_the_feature_cache(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_extract(monkeypatch)
    cache = tmp_path / "cache"
    first = _config_file(tmp_path, feature_selection={"manual": ["amplitude_stats"]}, **FAMILIES)
    assert run(first, tmp_path / "runs", cache) == 0
    assert len(calls) == 1
    second = _config_file(tmp_path, feature_selection={"mrmr": {"k": [2]}}, **FAMILIES)
    assert run(second, tmp_path / "runs", cache) == 0
    assert len(calls) == 1
    run_id = _run_id(capsys).splitlines()[-1]
    assert all(h["hit"] for h in _provenance_cache(run_id, tmp_path / "runs"))


def test_head_only_change_hits_the_feature_cache(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_extract(monkeypatch)
    cache = tmp_path / "cache"
    selection = {"manual": ["amplitude_stats"]}
    first = _config_file(tmp_path, feature_selection=selection, **FAMILIES)
    assert run(first, tmp_path / "runs", cache) == 0
    assert len(calls) == 1
    second = _config_file(
        tmp_path,
        feature_selection=selection,
        head={"name": "knn", "params": {}},
        **FAMILIES,
    )
    assert run(second, tmp_path / "runs", cache) == 0
    assert len(calls) == 1
    run_id = _run_id(capsys).splitlines()[-1]
    assert all(h["hit"] for h in _provenance_cache(run_id, tmp_path / "runs"))


def test_resume_keeps_feature_names(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    selection = {"manual": ["amplitude_stats.std"]}
    assert main(["run", _config_file(tmp_path, feature_selection=selection, **FAMILIES)]) == 0
    run_id = _run_id(capsys)
    config = _config_file(tmp_path, feature_selection=selection, seeds=[0, 1], **FAMILIES)
    assert main(["run", config]) == 0
    seeds = _metrics(run_id, tmp_path / "runs")
    assert seeds["0"]["feature_names"] == ["amplitude_stats.std"]
    assert seeds["1"]["feature_names"] == ["amplitude_stats.std"]
    assert seeds["0"]["selection"] == seeds["1"]["selection"]
    assert seeds["0"]["selection"]["chosen"] == 0


def _run_capture(tmp_path, monkeypatch, label, workers, **changes):
    """Run once into its own root/cache; return (feature matrix, metrics document)."""
    from csi_bench.runner import run as run_module

    captured = []
    original = run_module._extract

    def spy(*args, **kwargs):
        matrix = original(*args, **kwargs)
        captured.append(matrix)
        return matrix

    monkeypatch.setattr(run_module, "_extract", spy)
    root = tmp_path / f"runs_{label}"
    code = run(_config_file(tmp_path, **changes), root, tmp_path / f"cache_{label}", workers)
    assert code == 0
    (run_dir,) = [d for d in root.iterdir() if d.is_dir()]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return captured[-1], metrics


def test_parallel_extraction_matches_single_process(tmp_path, monkeypatch):
    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_module, "_fork_is_safe", lambda log: True)
    calls: list[int] = []
    original_parallel = run_module._extract_parallel

    def spy_parallel(*args, **kwargs):
        calls.append(1)
        return original_parallel(*args, **kwargs)

    monkeypatch.setattr(run_module, "_extract_parallel", spy_parallel)
    changes = {"preprocessing": ["ant2"], "extractors": [{"name": "amplitude_stats"}]}
    one, m_one = _run_capture(tmp_path, monkeypatch, "one", 1, **changes)
    assert calls == []
    many, m_many = _run_capture(tmp_path, monkeypatch, "many", 3, **changes)
    assert calls == [1]
    assert one.feature_names == many.feature_names
    np.testing.assert_array_equal(one.values, many.values)
    assert m_one["seeds"] == m_many["seeds"]


def test_loky_head_tuning_does_not_disable_parallel_extraction(tmp_path, monkeypatch):
    """A head-tuning run leaves loky housekeeping threads behind; the next run in the
    same process must still fork parallel extraction instead of falling back to slices."""
    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls: list[int] = []
    original_parallel = run_module._extract_parallel

    def spy_parallel(*args, **kwargs):
        calls.append(1)
        return original_parallel(*args, **kwargs)

    monkeypatch.setattr(run_module, "_extract_parallel", spy_parallel)
    changes = {
        "preprocessing": ["ant2"],
        "extractors": [{"name": "amplitude_stats"}],
        "head": {
            "name": "random_forest",
            "params": {"n_estimators_grid": [10], "max_depth_grid": [None]},
        },
    }
    _run_capture(tmp_path, monkeypatch, "first", 3, **changes)
    assert calls == [1]
    _run_capture(tmp_path, monkeypatch, "second", 3, **changes)
    assert calls == [1, 1]


def test_fork_is_safe_ignores_loky_housekeeping_threads(tmp_path, monkeypatch, caplog):
    from csi_bench.runner import run as run_module

    started = threading.Event()
    released = threading.Event()

    def worker():
        started.set()
        released.wait(10)

    t = threading.Thread(target=worker, name="plain-thread")
    t.start()
    started.wait(10)
    log = logging.getLogger("test")
    try:
        assert not run_module._fork_is_safe(log)
    finally:
        released.set()
        t.join()

    # one live thread per allowlist entry, each matching only that entry (by class or name)
    class _ExecutorManagerThread(threading.Thread):
        pass

    class TMonitor(threading.Thread):
        pass

    released.clear()
    housekeeping = [
        cls(target=released.wait, args=(10,), name=name)
        for cls, name in (
            (_ExecutorManagerThread, "loky-manager"),
            (threading.Thread, "ExecutorManagerThread"),
            (threading.Thread, "QueueFeederThread"),
            (TMonitor, "monitor"),
            (threading.Thread, "tqdm_monitor"),
        )
    ]
    for thread in housekeeping:
        thread.start()
    caplog.clear()
    try:
        assert all(thread.is_alive() for thread in housekeeping)
        assert run_module._fork_is_safe(log)
        assert not caplog.records
    finally:
        released.set()
        for thread in housekeeping:
            thread.join(timeout=10)


def test_block_context_hits_and_misses_are_logged(tmp_path, monkeypatch, caplog):
    import types

    from csi_bench.adapters.extractors.paper_beamforming import PaperBeamforming
    from csi_bench.adapters.storage.context_cache import NpzContextCache
    from csi_bench.config.experiment import ExperimentConfig
    from csi_bench.runner import run as run_module
    from csi_bench.runner.discover import discover

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    discover()
    config = ExperimentConfig.model_validate(
        {**CONFIG, "preprocessing": ["ant2"], "extractors": [{"name": "paper_beamforming"}]}
    )
    caplog.set_level(logging.INFO)
    cache = NpzContextCache(tmp_path / "cache" / "context")
    log = logging.getLogger("test")
    bounds = {"r1_empty": {"train": (0, 100)}}
    by_id = {
        "r1_empty": types.SimpleNamespace(
            csi=np.zeros((2, 100, 3), dtype=np.complex64),
            meta=types.SimpleNamespace(source_sha256="sha"),
        )
    }
    helper = run_module._block_context_getter(
        config, config.extractors[0], [], PaperBeamforming(), by_id, bounds, cache, log
    )
    ctx = helper("r1_empty", "train")
    assert ctx is not None
    again = helper("r1_empty", "train")
    np.testing.assert_array_equal(again, ctx)
    assert caplog.text.count("block context paper_beamforming r1_empty:train computed") == 1
    assert caplog.text.count("block context paper_beamforming r1_empty:train cached") == 1


def test_chunk_bounds_contiguous_and_capped():
    from csi_bench.runner.run import _chunk_bounds

    for n_items, n_chunks in [(10, 3), (2, 8), (7, 7), (100, 4)]:
        bounds = _chunk_bounds(n_items, n_chunks)
        assert len(bounds) == min(n_items, n_chunks)
        assert bounds[0][0] == 0 and bounds[-1][1] == n_items
        assert all(a < b for a, b in bounds)
        assert all(x[1] == y[0] for x, y in zip(bounds, bounds[1:], strict=False))


def test_few_windows_uses_fewer_chunks_in_order(monkeypatch):
    import types

    from csi_bench.core.contracts import FeatureMatrix
    from csi_bench.runner import run as run_module

    class Echo:
        def extract(self, slices):
            return FeatureMatrix(
                values=np.array([[s[0, 0, 0].real] for s in slices], dtype=np.float32),
                feature_names=("f",),
            )

    windows = [types.SimpleNamespace(recording_id="a", start=i, end=i + 1) for i in range(3)]
    csi = np.arange(3, dtype=np.float32).reshape(1, 3, 1).astype(np.complex64)
    by_id = {"a": types.SimpleNamespace(csi=csi)}
    matrix = run_module._extract_parallel(Echo(), windows, by_id, 8)
    assert matrix.values[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_extract_parallel_rejects_mismatched_feature_names():
    import types

    from csi_bench.core.contracts import FeatureMatrix
    from csi_bench.core.errors import ExtractorError
    from csi_bench.runner import run as run_module

    class VaryingNames:
        def extract(self, slices):
            return FeatureMatrix(
                values=np.zeros((len(slices), 1), dtype=np.float32),
                feature_names=(f"c{len(slices)}",),
            )

    windows = [types.SimpleNamespace(recording_id="a", start=i, end=i + 1) for i in range(3)]
    csi = np.arange(3, dtype=np.float32).reshape(1, 3, 1).astype(np.complex64)
    by_id = {"a": types.SimpleNamespace(csi=csi)}
    with pytest.raises(ExtractorError, match="different feature names"):
        run_module._extract_parallel(VaryingNames(), windows, by_id, 2)


def test_parallel_extraction_falls_back_when_threads_live(tmp_path, monkeypatch):
    import threading

    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls: list[int] = []
    original = run_module._extract_parallel

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(run_module, "_extract_parallel", spy)
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait)
    worker.start()
    try:
        changes = {"preprocessing": ["ant2"], "extractors": [{"name": "amplitude_stats"}]}
        _, metrics = _run_capture(tmp_path, monkeypatch, "threaded", 3, **changes)
    finally:
        stop.set()
        worker.join()
    assert calls == []
    assert "seeds" in metrics


def test_uses_context_extractor_dispatches_to_block_path(tmp_path, monkeypatch):
    from csi_bench.adapters.extractors.paper_beamforming import PaperBeamforming
    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    blocks: list[int] = []
    original_blocks = run_module._extract_blocks

    def spy_blocks(*args, **kwargs):
        blocks.append(1)
        return original_blocks(*args, **kwargs)

    def no_slices(self, slices):
        raise AssertionError("slice path used for a uses_context extractor")

    monkeypatch.setattr(run_module, "_extract_blocks", spy_blocks)
    monkeypatch.setattr(PaperBeamforming, "extract", no_slices)
    matrix = _run_capture(
        tmp_path,
        monkeypatch,
        "blocks",
        1,
        preprocessing=["ant2"],
        extractors=[{"name": "paper_beamforming"}],
    )[0]
    assert blocks == [1]
    assert matrix.values.shape[0] > 0


@pytest.mark.parametrize("head", ["random_forest", "knn"])
def test_head_metrics_equal_across_workers_and_see_jobs(tmp_path, monkeypatch, head):
    import joblib

    from csi_bench.adapters.heads.knn import Knn
    from csi_bench.adapters.heads.random_forest import RandomForest

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen: list[int] = []
    cls = {"random_forest": RandomForest, "knn": Knn}[head]
    original = cls.fit

    def spy(self, *args, **kwargs):
        seen.append(joblib.effective_n_jobs(None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", spy)
    changes = {"head": {"name": head, "params": {}}}
    if head == "random_forest":
        changes["head"]["params"] = {"n_estimators_grid": [10], "max_depth_grid": [None]}
    _, one = _run_capture(tmp_path, monkeypatch, "one", 1, **changes)
    _, many = _run_capture(tmp_path, monkeypatch, "many", 3, **changes)
    assert one["seeds"] == many["seeds"]
    assert seen[0] == 1 and seen[-1] == 3


def test_resolve_workers_sources(monkeypatch):
    from csi_bench.core.errors import ConfigError
    from csi_bench.runner.run import resolve_workers

    monkeypatch.delenv("CSI_BENCH_WORKERS", raising=False)
    assert 1 <= resolve_workers(None) <= 4
    monkeypatch.setenv("CSI_BENCH_WORKERS", "2")
    assert resolve_workers(None) == 2
    assert resolve_workers("5") == 5
    monkeypatch.setenv("CSI_BENCH_WORKERS", "x")
    with pytest.raises(ConfigError, match="CSI_BENCH_WORKERS"):
        resolve_workers(None)
    with pytest.raises(ConfigError, match="--workers"):
        resolve_workers("0")


def test_bad_worker_count_rejected_before_any_run(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path), "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    monkeypatch.setenv("CSI_BENCH_WORKERS", "abc")
    assert main(["run", _config_file(tmp_path)]) == 2
    assert "CSI_BENCH_WORKERS" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_extract_blocks_called_per_block_in_window_order():
    import types

    from csi_bench.core.contracts import FeatureMatrix
    from csi_bench.runner import run as run_module

    calls: list[tuple[int, list[int], int]] = []

    class Ctx:
        uses_context = True

        def extract_blocks(self, block, offsets, window_len):
            calls.append((block.shape[1], list(offsets), window_len))
            return FeatureMatrix(
                values=np.array([[block[0, o, 0].real] for o in offsets], dtype=np.float32),
                feature_names=("f",),
            )

    def window(rec, split, start, length=2):
        return types.SimpleNamespace(recording_id=rec, split=split, start=start, end=start + length)

    # AD-4 order is (recording, start): train/val windows of one recording interleave by start
    windows = [
        window("a", "train", 0),
        window("a", "train", 1),
        window("a", "val", 10),
        window("b", "train", 0),
    ]
    csi = np.arange(20, dtype=np.float32).reshape(1, 20, 1).astype(np.complex64)
    by_id = {r: types.SimpleNamespace(csi=csi) for r in "ab"}
    matrix = run_module._extract_blocks(Ctx(), windows, by_id)
    assert matrix.values[:, 0].tolist() == [0.0, 1.0, 10.0, 0.0]
    assert calls == [(3, [0, 1], 2), (2, [0], 2), (2, [0], 2)]


def _shared_context_runs(tmp_path, monkeypatch, runs):
    """Run `(window_len, stride, rooms)` configs against one cache; return matrices and the
    number of `compute_context` calls after each run."""
    from csi_bench.adapters.extractors.paper_beamforming import PaperBeamforming
    from csi_bench.runner import run as run_module

    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    computed: list[int] = []
    original = PaperBeamforming.compute_context

    def spy(self, block):
        computed.append(1)
        return original(self, block)

    monkeypatch.setattr(PaperBeamforming, "compute_context", spy)
    captured: list = []
    extract = run_module._extract

    def capture(*args, **kwargs):
        captured.append(extract(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(run_module, "_extract", capture)
    counts = []
    for i, (window_len, stride, rooms) in enumerate(runs):
        config = _config_file(
            tmp_path,
            window_len=window_len,
            stride=stride,
            selection={"rooms": rooms},
            preprocessing=["ant2"],
            extractors=[{"name": "paper_beamforming"}],
        )
        assert run(config, tmp_path / f"runs_{i}", tmp_path / "cache", 1) == 0
        counts.append(len(computed))
    return captured, counts


def test_block_arrays_shared_across_window_len_stride_and_rooms(tmp_path, monkeypatch):
    runs = [(20, 1, ["r1"]), (30, 3, ["r1"]), (25, 1, ["r2"]), (20, 2, ["r1", "r2"])]
    matrices, counts = _shared_context_runs(tmp_path, monkeypatch, runs)
    # r1 has 4 recordings x 3 blocks, r2 the same; later runs compute nothing new
    assert counts[0] > 0
    assert counts[1] == counts[0]
    assert counts[2] == 2 * counts[0]
    assert counts[3] == counts[2]
    assert len(list((tmp_path / "cache" / "context").glob("*.npz"))) == counts[3]


def test_shared_arrays_equal_slice_path_exactly(tmp_path, monkeypatch):
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.extractors.paper_beamforming import PaperBeamforming
    from csi_bench.core.splitplan import find_split, load_split_plan
    from csi_bench.core.windows import build_windows

    runs = [(20, 1, ["r1"]), (30, 3, ["r1", "r2"])]
    matrices, _ = _shared_context_runs(tmp_path, monkeypatch, runs)
    data = tmp_path / "data/csi_proc_python/saved_csi"
    recs = TorVergata(TorVergata.Params(root=str(data))).load()
    plan = load_split_plan(find_split("tor_vergata", None, tmp_path / "splits"), recs)
    for (window_len, stride, rooms), matrix in zip(runs, matrices, strict=True):
        by_id = {r.recording_id: r for r in recs if r.room in rooms}
        windows = build_windows(plan, list(by_id.values()), rooms, window_len, stride)
        slices = [by_id[w.recording_id].csi[:2, w.start : w.end, :] for w in windows]
        expected = PaperBeamforming().extract(slices)
        np.testing.assert_array_equal(matrix.values, expected.values)


def test_context_cache_key_ignores_window_stride_rooms_but_not_fitted_state():
    from csi_bench.config.experiment import ExperimentConfig, context_cache_key
    from csi_bench.core.contracts import FeatureMatrix  # noqa: F401
    from csi_bench.runner.discover import discover

    discover()
    base = {**CONFIG, "preprocessing": ["ant2"], "extractors": [{"name": "paper_beamforming"}]}
    a = ExperimentConfig.model_validate(base)
    b = ExperimentConfig.model_validate(
        {**base, "window_len": 20, "stride": 5, "selection": {"rooms": ["r1", "r2"]}}
    )
    spec = a.extractors[0]
    args = ("r1_empty", "sha", (0, 100))
    assert context_cache_key(a, spec, [{"ids": [0, 1]}], *args) == context_cache_key(
        b, spec, [{"ids": [0, 1]}], *args
    )
    assert context_cache_key(a, spec, [{"ids": [0, 1]}], *args) != context_cache_key(
        a, spec, [{"ids": [0, 2]}], *args
    )
    assert context_cache_key(a, spec, [], *args) != context_cache_key(
        a, spec, [], "r1_empty", "sha", (0, 101)
    )


def test_corrupt_context_entry_is_recomputed(tmp_path):
    from csi_bench.adapters.storage.context_cache import NpzContextCache

    cache = NpzContextCache(tmp_path)
    assert cache.get("k") is None
    cache.put("k", np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(cache.get("k"), np.arange(6.0).reshape(2, 3))
    (tmp_path / "k.npz").write_bytes(b"junk")
    assert cache.get("k") is None


def test_random_window_reference_run_and_comparison_refused(tmp_path, monkeypatch, capsys):
    import shutil

    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.random_window import RandomWindow

    _make_project(tmp_path)
    shutil.rmtree(tmp_path / "splits")
    recs = TorVergata(
        TorVergata.Params(root=str(tmp_path / "data/csi_proc_python/saved_csi"))
    ).load()
    plan = RandomWindow().split(recs, "tor_vergata")
    write_split_plan(plan, tmp_path / "splits")
    monkeypatch.chdir(tmp_path)
    assert main(["run", _config_file(tmp_path, role="reference")]) == 0
    run_id = capsys.readouterr().out.strip()
    prov = json.loads(rs.provenance_path(run_id, tmp_path / "runs").read_text())
    assert prov["split_hash"] == plan.split_hash
    assert main(["run", _config_file(tmp_path, role="comparison")]) != 0
    assert "role: reference" in capsys.readouterr().err


def test_run_refuses_comparison_on_window_plan_without_validation(tmp_path, monkeypatch, capsys):
    """The `run` gate holds on its own, even if config validation let the config through."""
    import shutil

    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.splitters.random_window import RandomWindow
    from csi_bench.config import experiment

    _make_project(tmp_path)
    shutil.rmtree(tmp_path / "splits")
    recs = TorVergata(
        TorVergata.Params(root=str(tmp_path / "data/csi_proc_python/saved_csi"))
    ).load()
    write_split_plan(RandomWindow().split(recs, "tor_vergata"), tmp_path / "splits")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "check_split_role", lambda unit, role: None)
    assert main(["run", _config_file(tmp_path, role="comparison")]) != 0
    assert "role: reference" in capsys.readouterr().err
