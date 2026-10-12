import json
from pathlib import Path

import pandas as pd
import yaml

from csi_bench.core import results_schema as rs
from csi_bench.runner.cli import main
from csi_bench.runner.migrate import find_legacy_dirs, migrate_runs
from tests.test_run import (  # noqa: F401 - fixture re-export
    _config_file,
    _make_project,
    fresh_registry,
    single_worker,
)
from tests.test_sweep import _write

BLOCK, WINDOW = "aaaaaaaaaaaa", "bbbbbbbbbbbb"


def _splits(root: Path) -> Path:
    folder = root / "splits" / "ds"
    folder.mkdir(parents=True)
    (folder / f"{BLOCK}.json").write_text(json.dumps({"splitter": "time_block"}))
    (folder / f"{WINDOW}.json").write_text(
        json.dumps({"splitter": "random_window", "unit": "window"})
    )
    return root / "splits"


def _config(rooms=("r1",), split=None, dataset="ds"):
    return {
        "dataset": dataset,
        "split": split,
        "selection": {"rooms": list(rooms)},
        "window_len": 40,
        "stride": 1,
        "extractors": [{"name": "amplitude_stats", "params": {}}],
        "feature_selection": None,
        "head": {"name": "svm", "params": {}},
        "seeds": [0],
        "role": "comparison",
    }


def _legacy(parent: Path, run_id: str, config=None, split_hash=None, finished="2026-01-01"):
    d = parent / run_id
    (d / "seeds" / "0").mkdir(parents=True)
    (d / "config.yaml").write_text(
        config if isinstance(config, str) else yaml.safe_dump(config or _config())
    )
    (d / "status.json").write_text(json.dumps({"status": "done", "finished_at": finished}))
    (d / "metrics.json").write_text(json.dumps({"run_id": run_id, "finished": finished}))
    (d / "seeds" / "0" / "predictions.parquet").write_bytes(b"x")
    if split_hash is not None:
        (d / "provenance.json").write_text(json.dumps({"split_hash": split_hash}))
    return d


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "index.parquet"
    }


def test_moves_resolves_splitter_and_rebuilds_index(tmp_path, capsys):
    splits = _splits(tmp_path)
    runs = tmp_path / "runs"
    _legacy(runs / "backup", "000000000001", split_hash=WINDOW)
    _legacy(runs / "backup", "000000000002", _config(split=WINDOW))
    _legacy(runs / "paper", "000000000003")  # no provenance, no split: the only block plan
    _legacy(runs, "000000000004", _config(dataset="other"))  # no plan at all
    _legacy(runs, "000000000005", "- not\n- a mapping\n")
    metrics_before = {p.parent.name: p.read_bytes() for p in runs.rglob("metrics.json")}
    assert migrate_runs(runs, splits) == 0
    out = capsys.readouterr().out
    assert "moved 5, parked 0, skipped 0" in out
    slug = "amplitude_stats-none-svm-r1-w40"
    expected = {
        runs / "ds" / "random_window" / f"{slug}--000000000001",
        runs / "ds" / "random_window" / f"{slug}--000000000002",
        runs / "ds" / "time_block" / f"{slug}--000000000003",
        runs / "other" / "unknown" / f"{slug}--000000000004",
        runs / "unknown" / "unknown" / "unknown--000000000005",
    }
    assert set(rs.iter_run_dirs(runs)) == expected
    assert find_legacy_dirs(runs) == []
    assert (runs / "backup").is_dir() and (runs / "paper").is_dir()  # emptied folders stay
    for path in expected:
        run_id = rs.parse_run_dir(path, runs)[2]
        assert rs.metrics_path(run_id, runs).read_bytes() == metrics_before[run_id]
    index = pd.read_parquet(rs.index_path(runs))
    assert set(index.run_id) == {f"00000000000{i}" for i in range(1, 5)}
    assert dict(zip(index.run_id, index.splitter, strict=True))["000000000003"] == "time_block"


def test_second_pass_moves_nothing_and_touches_nothing(tmp_path, capsys):
    splits = _splits(tmp_path)
    runs = tmp_path / "runs"
    for i in range(3):
        _legacy(runs / "backup", f"00000000000{i}")
    assert migrate_runs(runs, splits) == 0
    before = _snapshot(runs)
    index_stat = rs.index_path(runs).stat().st_mtime_ns
    capsys.readouterr()
    assert main(["migrate-runs", "--runs", str(runs)]) == 0
    assert "moved 0, parked 0, skipped 3" in capsys.readouterr().out
    assert _snapshot(runs) == before
    assert rs.index_path(runs).stat().st_mtime_ns == index_stat


def test_duplicates_newest_kept_others_parked(tmp_path, capsys):
    splits = _splits(tmp_path)
    runs = tmp_path / "runs"
    _legacy(runs / "backup", "00000000000a", finished="2026-10-11")
    _legacy(runs / "paper", "00000000000a", finished="2026-10-10")
    assert migrate_runs(runs, splits) == 0
    out = capsys.readouterr().out
    assert "moved 1, parked 1" in out and "paper" in out
    kept = rs.run_dir("00000000000a", runs)
    assert json.loads((kept / "metrics.json").read_text())["finished"] == "2026-10-11"
    parked = runs / "_duplicates" / "paper" / "00000000000a"
    assert json.loads((parked / "metrics.json").read_text())["finished"] == "2026-10-10"
    assert find_legacy_dirs(runs) == []
    assert len(pd.read_parquet(rs.index_path(runs))) == 1


def test_existing_layout_dir_wins_over_legacy_copies(tmp_path, capsys):
    splits = _splits(tmp_path)
    runs = tmp_path / "runs"
    layout = rs.new_run_dir(runs, "ds", "time_block", "s", "00000000000b")
    layout.mkdir(parents=True)
    _legacy(runs, "00000000000b", finished="2099-01-01")
    assert migrate_runs(runs, splits) == 0
    assert "moved 0, parked 1, skipped 1" in capsys.readouterr().out
    assert rs.run_dir("00000000000b", runs) == layout
    assert (runs / "_duplicates" / "00000000000b" / "metrics.json").is_file()


def test_legacy_finder_ignores_non_runs(tmp_path):
    runs = tmp_path / "runs"
    _legacy(runs / "_dataset", "00000000000c")
    (runs / "abcdefabcdef").mkdir(parents=True)  # no config.yaml
    (runs / "not-a-hash").mkdir()
    (runs / "not-a-hash" / "config.yaml").write_text("{}")
    assert find_legacy_dirs(runs) == []
    assert find_legacy_dirs(tmp_path / "missing") == []


def test_real_run_migrated_then_sweep_skips(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    assert main(["sweep", path]) == 0
    (layout,) = rs.iter_run_dirs(tmp_path / "runs")
    run_id = rs.parse_run_dir(layout, tmp_path / "runs")[2]
    metrics = rs.metrics_path(run_id, tmp_path / "runs").read_bytes()
    legacy = tmp_path / "runs" / "old" / run_id
    legacy.parent.mkdir()
    layout.rename(legacy)  # back to the old layout
    capsys.readouterr()
    assert main(["report", "--runs", "runs"]) == 0
    assert "migrate-runs" in capsys.readouterr().err
    assert main(["migrate-runs", "--runs", "runs"]) == 0
    assert rs.run_dir(run_id, "runs") == Path("runs") / layout.relative_to(tmp_path / "runs")
    assert rs.metrics_path(run_id, "runs").read_bytes() == metrics
    capsys.readouterr()
    assert main(["sweep", path]) == 0
    assert f"{run_id} skipped" in capsys.readouterr().out
    assert len(rs.iter_run_dirs("runs")) == 1


def test_failure_midway_lists_what_moved_and_refreshes_index(tmp_path, monkeypatch, capsys):
    splits = _splits(tmp_path)
    runs = tmp_path / "runs"
    _legacy(runs / "backup", "000000000001")
    _legacy(runs / "backup", "000000000002")
    _legacy(runs / "paper", "000000000002", finished="2027-01-01")  # kept: parked first
    original = Path.rename
    calls = []

    def flaky(self, target):
        calls.append(self)
        if len(calls) == 3:
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    assert migrate_runs(runs, splits) == 2
    cap = capsys.readouterr()
    assert "disk full" in cap.err
    assert "moved 1, parked 1" in cap.out and "parked duplicate" in cap.out
    assert set(pd.read_parquet(rs.index_path(runs)).run_id) == {"000000000001"}


def _swept_run(tmp_path, monkeypatch) -> tuple[str, Path]:
    """One real finished run whose `metrics.json` is stripped of `intervals` (an old run)."""
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["sweep", _write(tmp_path, {"window_len": [40]})]) == 0
    (layout,) = rs.iter_run_dirs(tmp_path / "runs")
    run_id = rs.parse_run_dir(layout, tmp_path / "runs")[2]
    metrics_file = layout / "metrics.json"
    document = json.loads(metrics_file.read_text())
    document.pop("intervals")
    metrics_file.write_text(json.dumps(document, indent=2, sort_keys=True))
    return run_id, metrics_file


def _snapshot_paths(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "index.parquet"]


def test_backfill_fills_intervals_leaving_everything_else_unchanged(tmp_path, monkeypatch, capsys):
    run_id, metrics_file = _swept_run(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    old_text = metrics_file.read_text()
    others = {p: p.read_bytes() for p in _snapshot_paths(runs) if p != metrics_file}
    capsys.readouterr()
    assert main(["report", "--runs", "runs"]) == 0
    cap = capsys.readouterr()
    assert "1 runs have no confidence intervals" in cap.err and "[-]" in cap.out
    assert metrics_file.read_text() == old_text  # report does not write run files
    assert main(["migrate-runs", "--runs", "runs"]) == 0
    assert "filled confidence intervals for 1 runs" in capsys.readouterr().out
    after = json.loads(metrics_file.read_text())
    intervals = after.pop("intervals")
    assert after == json.loads(old_text)
    # the rest of the file is byte-identical: re-serialising without the key gives the old text
    assert json.dumps(after, indent=2, sort_keys=True) == old_text
    assert set(intervals) == {"test"} and intervals["test"]["unit"] == "recording"
    for score in ("accuracy", "macro_f1"):
        lo, hi = intervals["test"][score]
        assert lo <= after["mean_std"]["test"][score][0] <= hi
    assert {p: p.read_bytes() for p in others if p.exists()} == others
    assert [rs.parse_run_dir(p, runs)[2] for p in rs.iter_run_dirs(runs)] == [run_id]
    index = pd.read_parquet(rs.index_path(runs))
    assert index.test_macro_f1_lo.notna().all() and index.test_accuracy_hi.notna().all()
    capsys.readouterr()
    assert main(["report", "--runs", "runs"]) == 0
    cap = capsys.readouterr()
    assert "no confidence intervals" not in cap.err and "[-]" not in cap.out


def test_backfill_second_pass_rewrites_nothing(tmp_path, monkeypatch, capsys):
    _, metrics_file = _swept_run(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    assert main(["migrate-runs", "--runs", "runs"]) == 0
    before = _snapshot(runs)
    stamps = {p: p.stat().st_mtime_ns for p in [*_snapshot_paths(runs), rs.index_path(runs)]}
    capsys.readouterr()
    assert main(["migrate-runs", "--runs", "runs"]) == 0
    assert "filled confidence intervals for 0 runs" in capsys.readouterr().out
    assert _snapshot(runs) == before
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps


def test_backfill_skips_run_with_missing_predictions(tmp_path, monkeypatch, capsys):
    run_id, metrics_file = _swept_run(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    rs.predictions_path(run_id, 0, runs).unlink()
    before = metrics_file.read_bytes()
    capsys.readouterr()
    assert main(["migrate-runs", "--runs", "runs"]) == 0
    cap = capsys.readouterr()
    assert "warning:" in cap.err and "no intervals" in cap.err
    assert "1 could not be filled" in cap.out
    assert metrics_file.read_bytes() == before
    assert main(["report", "--runs", "runs"]) == 0
    assert "[-]" in capsys.readouterr().out


def test_backfill_ignores_unfinished_runs(tmp_path, capsys):
    runs = tmp_path / "runs"
    d = rs.new_run_dir(runs, "ds", "time_block", "s", "00000000000d")
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": "failed"}))
    assert migrate_runs(runs, _splits(tmp_path)) == 0
    cap = capsys.readouterr()
    assert "filled confidence intervals for 0 runs" in cap.out and cap.err == ""
