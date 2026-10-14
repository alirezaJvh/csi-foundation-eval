from pathlib import Path

import pytest
import yaml

from csi_bench.config.experiment import compute_run_id
from csi_bench.config.sweep import _set_path, expand_sweep
from csi_bench.core import results_schema as rs
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


def test_running_status_is_rerun_on_resume(tmp_path, monkeypatch):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    assert main(["sweep", path]) == 0
    (run_dir,) = _run_dirs(tmp_path)
    status = run_dir / "status.json"
    status.write_text(status.read_text().replace('"done"', '"running"'))
    calls = []
    original = run_module._execute
    monkeypatch.setattr(
        run_module, "_execute", lambda *a, **k: (calls.append(1), original(*a, **k))
    )
    assert main(["sweep", path]) == 0
    assert calls == [1]
    assert _status(run_dir)["status"] == "done"


def test_sweep_forwards_workers(tmp_path, monkeypatch):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    seen: list[int | None] = []
    monkeypatch.setattr(run_module, "_execute", lambda *a, **k: seen.append(k.get("workers")))
    assert main(["sweep", path, "--workers", "3"]) == 0
    assert seen == [3]


def test_sweep_bad_worker_count_rejected_before_any_run(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [40]})
    assert main(["sweep", path, "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    monkeypatch.setenv("CSI_BENCH_WORKERS", "abc")
    assert main(["sweep", path]) == 2
    assert "CSI_BENCH_WORKERS" in capsys.readouterr().err
    assert _run_dirs(tmp_path) == []


def test_paired_axes_apply_fields_together(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    sweep = {
        "@a": [{"window_len": 10, "head.name": "svm"}, {"window_len": 30, "head.name": "knn"}],
        "@b": [{"stride": 1}, {"stride": 2}],
    }
    out = _expand(project, sweep, monkeypatch)
    got = [(c.window_len, c.head.name, c.stride) for _, c in out]
    assert got == [(10, "svm", 1), (10, "svm", 2), (30, "knn", 1), (30, "knn", 2)]
    assert out[0][0]["@a"] == {"window_len": 10, "head.name": "svm"}


@pytest.mark.parametrize(
    "sweep",
    [
        {"@": [{"window_len": 10}]},
        {"@a": []},
        {"@a": {"window_len": 10}},
        {"@a": [5]},
        {"@a": [{}]},
        {"@a": [{"seeds": [1]}]},
        {"@a": [{"seeds.0": 1}]},
        {"@a": [{"window_len": 10.5}]},
        {"@a": [{"head": {"name": "svm"}, "head.name": "knn"}]},
        {"window_len": [10], "@a": [{"window_len": 20}]},
        {"@a": [{"window_len": 10}], "@b": [{"stride": 1}, {"window_len": 20}]},
        {"head": [{"name": "svm"}], "@a": [{"head.params.c": 1}]},
    ],
)
def test_bad_paired_axis_rejected(tmp_path, monkeypatch, sweep):
    project = _make_project(tmp_path)
    with pytest.raises(ConfigError):
        _expand(project, sweep, monkeypatch)


def test_invalid_paired_combo_is_rejected_with_overrides(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    sweep = {"@a": [{"window_len": 10}, {"role": "nonsense", "window_len": 20}]}
    with pytest.raises(ConfigError, match="nonsense"):
        _expand(project, sweep, monkeypatch)


def test_find_split_ignores_window_plans_unless_named(tmp_path):
    from csi_bench.core.errors import DataError
    from csi_bench.core.splitplan import find_split

    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "aaa.json").write_text('{"unit": "window"}')
    with pytest.raises(DataError, match="no block split plan"):
        find_split("ds", None, tmp_path)
    (folder / "bbb.json").write_text("{}")
    assert find_split("ds", None, tmp_path).stem == "bbb"
    assert find_split("ds", "aaa", tmp_path).stem == "aaa"
    (folder / "ccc.json").write_text('{"unit": "block"}')
    with pytest.raises(DataError, match="several split plans"):
        find_split("ds", None, tmp_path)


def test_resume_in_place_after_seed_added(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["sweep", _write(tmp_path, {"window_len": [40]})]) == 0
    (first,) = _run_dirs(tmp_path)
    # seeds are not part of the run id: adding one resumes the same directory
    assert main(["sweep", _write(tmp_path, {"window_len": [40]}, seeds=[0, 1])]) == 0
    assert _run_dirs(tmp_path) == [first]
    assert (first / "seeds" / "1" / "predictions.parquet").is_file()


def test_run_with_several_dirs_fails_and_sweep_continues(tmp_path, monkeypatch, capsys):
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, {"window_len": [30, 40]})
    assert main(["sweep", path]) == 0
    first, second = _run_dirs(tmp_path)
    run_id = rs.parse_run_dir(first, tmp_path / "runs")[2]
    rs.new_run_dir(tmp_path / "runs", "tor_vergata", "random_window", "copy", run_id).mkdir(
        parents=True
    )
    capsys.readouterr()
    assert main(["sweep", path]) == 1
    cap = capsys.readouterr()
    assert f"{run_id} failed" in cap.out and "several directories" in cap.err
    assert f"{rs.parse_run_dir(second, tmp_path / 'runs')[2]} skipped" in cap.out


def test_list_sweep_is_union_in_item_order(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    sweep = [{"window_len": [10, 30]}, {"head.name": ["knn"], "window_len": [30]}]
    out = _expand(project, sweep, monkeypatch)
    got = [(c.window_len, c.head.name) for _, c in out]
    assert got == [(10, CONFIG["head"]["name"]), (30, CONFIG["head"]["name"]), (30, "knn")]


def test_list_sweep_dedups_across_items(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    sweep = [{"window_len": [10, 30]}, {"window_len": [30, 40]}]
    out = _expand(project, sweep, monkeypatch)
    assert [c.window_len for _, c in out] == [10, 30, 40]


def test_mapping_sweep_matches_single_item_list(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    sweep = {"window_len": [10, 30], "head.name": ["svm", "knn"]}
    as_map = _expand(project, sweep, monkeypatch)
    as_list = _expand(project, [sweep], monkeypatch)
    assert [compute_run_id(c) for _, c in as_map] == [compute_run_id(c) for _, c in as_list]


@pytest.mark.parametrize(
    ("sweep", "match"),
    [
        ([{"window_len": [10]}, 5], r"sweep\.1"),
        ([], "sweep must be"),
        ([{"window_len": [10]}, {"window_len": []}], r"sweep\.1\.window_len"),
        ([{"extractors.5.name": ["x"]}], r"sweep\.0\.extractors\.5: "),
    ],
)
def test_bad_list_sweep_names_item(tmp_path, monkeypatch, sweep, match):
    project = _make_project(tmp_path)
    with pytest.raises(ConfigError, match=match):
        _expand(project, sweep, monkeypatch)


def test_mapping_sweep_error_text_unchanged(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    with pytest.raises(ConfigError, match=r"test: sweep\.window_len: expected a non-empty"):
        _expand(project, {"window_len": []}, monkeypatch)


def test_cross_room_sweep_expands(monkeypatch):
    monkeypatch.chdir(ROOT)
    discover()
    raw = yaml.safe_load((ROOT / "experiments/classical_cross_room.yaml").read_text())
    configs = [c for _, c in expand_sweep(raw, "classical_cross_room")]
    assert len({compute_run_id(c) for c in configs}) == len(configs) == 146
    by_split = {s: [c for c in configs if c.split == s] for s in ("1e8688472a1f", "c4faaf799ad5")}
    assert [len(v) for v in by_split.values()] == [73, 73]
    for c in configs:
        assert c.selection.rooms == ["r1", "r2"]
        assert (c.role, c.seeds) == ("comparison", [0])
        assert [(s.name, s.params) for s in c.preprocessing] == [("antennas", {"ids": [0, 1]})]
    paper = [c for c in configs if c.window_len == 56]
    assert len(paper) == 2
    for c in paper:
        assert [e.name for e in c.extractors] == ["paper_beamforming"]
        assert (c.head.name, c.stride, c.feature_selection) == ("random_forest", 1, None)
    assert {c.window_len for c in configs if c.window_len != 56} == {20, 30, 40, 50}
