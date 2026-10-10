import json
import re
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
