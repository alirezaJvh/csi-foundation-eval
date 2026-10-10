from pathlib import Path

import pytest
import yaml

from csi_bench.config.sweep import _set_path, expand_sweep
from csi_bench.core.errors import ConfigError
from csi_bench.runner import run as run_module
from csi_bench.runner.cli import main
from csi_bench.runner.discover import discover
from tests.test_run import (  # noqa: F401 - fixture re-export
    CONFIG,
    ROOT,
    _make_project,
    _run_dirs,
    _status,
    fresh_registry,
)


def _write(tmp_path: Path, sweep: object, **changes) -> str:
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, **changes, "sweep": sweep}, sort_keys=False))
    return str(path)


def _expand(project: Path, sweep: object, monkeypatch, **changes):
    monkeypatch.chdir(project)
    discover()
    return expand_sweep({**CONFIG, **changes, "sweep": sweep}, "test")


def test_expansion_count_and_order(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    out = _expand(project, {"window_len": [10, 30], "head.name": ["svm", "knn"]}, monkeypatch)
    assert [(o["window_len"], o["head.name"]) for o, _ in out] == [
        (10, "svm"),
        (10, "knn"),
        (30, "svm"),
        (30, "knn"),
    ]
    assert out[1][1].head.name == "knn" and out[2][1].window_len == 30


def test_duplicates_collapse(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    out = _expand(project, {"window_len": [30, 30]}, monkeypatch)
    assert len(out) == 1


@pytest.mark.parametrize(
    "sweep",
    [
        {"seeds": [[0], [1]]},
        {"seeds.0": [1, 2]},
        {"window_len": []},
        [1, 2],
        {"window_len": 5},
        {"extractors.1.name": ["x"]},
        {"extractors.5.name": ["x"]},
    ],
)
def test_bad_sweep_rejected(tmp_path, monkeypatch, sweep):
    project = _make_project(tmp_path)
    with pytest.raises(ConfigError):
        _expand(project, sweep, monkeypatch)


def test_missing_base_field(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    monkeypatch.chdir(project)
    discover()
    base = {k: v for k, v in CONFIG.items() if k != "head"}
    with pytest.raises(ConfigError):
        expand_sweep({**base, "sweep": {"window_len": [30]}}, "test")


def test_invalid_combo_names_overrides_and_runs_nothing(tmp_path, monkeypatch, capsys):
    project = _make_project(tmp_path)
    monkeypatch.chdir(project)
    path = _write(tmp_path, {"window_len": [40, 30], "head.name": ["logreg", "nope"]})
    assert main(["sweep", path]) == 2
    assert "nope" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_resume_skips_done_and_force_reruns(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [30, 40]})
    assert main(["sweep", path]) == 0
    assert len(_run_dirs(tmp_path)) == 2
    calls = []
    original = run_module._execute
    monkeypatch.setattr(
        run_module, "_execute", lambda *a, **k: (calls.append(k.get("force")), original(*a, **k))
    )
    assert main(["sweep", path]) == 0
    assert calls == []
    assert main(["sweep", path, "--force"]) == 0
    assert calls == [True, True]


def test_dirty_rerun_rules(tmp_path, monkeypatch):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    monkeypatch.setattr(run_module, "_git", lambda args: " M x" if args[0] == "status" else "h")
    assert main(["sweep", path]) == 0
    calls = []
    original = run_module._execute
    monkeypatch.setattr(
        run_module, "_execute", lambda *a, **k: (calls.append(1), original(*a, **k))
    )
    assert main(["sweep", path]) == 0  # still dirty: skipped
    assert calls == []
    monkeypatch.setattr(run_module, "_git", lambda args: "" if args[0] == "status" else "h")
    assert main(["sweep", path]) == 0  # clean now: rerun
    assert calls == [1]
    assert main(["sweep", path]) == 0  # recorded clean: skipped
    assert calls == [1]


def test_failure_continues_and_exits_1(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [30, 40]})
    original = run_module._extract

    def flaky(config, *a, **k):
        if config.window_len == 30:
            raise RuntimeError("boom")
        return original(config, *a, **k)

    monkeypatch.setattr(run_module, "_extract", flaky)
    assert main(["sweep", path]) == 1
    out = capsys.readouterr().out
    assert "failed" in out and "1 of 2" in out
    statuses = sorted(_status(d)["status"] for d in _run_dirs(tmp_path))
    assert statuses == ["done", "failed"]
    failed = next(d for d in _run_dirs(tmp_path) if _status(d)["status"] == "failed")
    assert "boom" in _status(failed)["trace"]


def test_example_sweep_expands(monkeypatch):
    monkeypatch.chdir(ROOT)
    discover()
    raw = yaml.safe_load((ROOT / "experiments/classical_baseline.yaml").read_text())
    out = expand_sweep(raw, "classical_baseline")
    assert len(out) == 4 * 3 * 2 * 3 * 3


def test_set_path_list_index_and_missing_mapping():
    raw = {"extractors": [{"name": "a"}]}
    _set_path(raw, "extractors.0.name", "b")
    assert raw == {"extractors": [{"name": "b"}]}
    with pytest.raises(ConfigError, match="valid index"):
        _set_path(raw, "extractors.1.name", "x")
    _set_path(raw, "head.params.C", 2.0)
    assert raw["head"] == {"params": {"C": 2.0}}


def test_list_index_override_lands_on_extractors(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    out = _expand(project, {"extractors.0.name": ["paper_beamforming"]}, monkeypatch)
    assert len(out) == 1
    assert out[0][1].extractors[0].name == "paper_beamforming"


def test_missing_mapping_created_sweeping_head_params(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    out = _expand(project, {"head.params.max_iter": [2000]}, monkeypatch)
    assert len(out) == 1
    assert out[0][1].head.params["max_iter"] == 2000


def test_git_unavailable_skips_dirty_done_runs(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    monkeypatch.setattr(run_module, "_git", lambda args: " M x" if args[0] == "status" else "h")
    assert main(["sweep", path]) == 0
    calls = []
    original = run_module._execute
    monkeypatch.setattr(
        run_module, "_execute", lambda *a, **k: (calls.append(1), original(*a, **k))
    )
    monkeypatch.setattr(run_module, "_git", lambda args: None)
    assert main(["sweep", path]) == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "skipped" in out and "1 skipped" in out
