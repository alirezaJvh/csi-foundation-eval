import json
from pathlib import Path

import pytest

from csi_bench.core import results_schema as rs
from csi_bench.runner.cli import main
from csi_bench.runner.rescore import rescore_report
from tests.test_report import _config as _report_config
from tests.test_report import _make_run
from tests.test_run import (  # noqa: F401 - fixture re-export
    _config_file,
    _make_project,
    _run_id,
    fresh_registry,
    single_worker,
)


def _as_old_run(run_id: str, root: Path) -> dict:
    """Strip the score, as `metrics.json` looked before it existed."""
    path = rs.metrics_path(run_id, root)
    document = json.loads(path.read_text())
    del document["mean_std"][rs.TUNING_VAL]
    for record in document["seeds"].values():
        del record[rs.TUNING_SCORE]
    path.write_text(json.dumps(document, indent=2, sort_keys=True))
    return document


@pytest.fixture
def store(tmp_path, monkeypatch, capsys):
    """Two real old runs in a real project; returns (runs root, backup, run ids)."""
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    ids = []
    for window in (20, 40):
        assert main(["run", _config_file(tmp_path, seeds=[0, 1], window_len=window)]) == 0
        ids.append(_run_id(capsys))
        _as_old_run(ids[-1], root)
    return root, tmp_path / "backup", ids


def _rescore(root, backup, capsys):
    code = main(["rescore-runs", "--runs", str(root), "--backup", str(backup)])
    return code, capsys.readouterr()


def _backup_file(root: Path, backup: Path, run_id: str, name: str) -> Path:
    return backup / rs.run_dir(run_id, root).relative_to(root) / name


def _compare(root, backup, capsys):
    out = root.parent / "compare.md"
    code = main(["rescore-report", "--runs", str(root), "--backup", str(backup), "--out", str(out)])
    capsys.readouterr()
    return code, out.read_text()


def test_rescore_fills_the_score_and_keeps_ids_and_test_metrics(store, capsys):
    root, backup, ids = store
    old = {i: json.loads(rs.metrics_path(i, root).read_text()) for i in ids}
    old_provenance = {i: rs.provenance_path(i, root).read_bytes() for i in ids}
    code, cap = _rescore(root, backup, capsys)
    assert code == 0, cap.err
    assert "2 rerun, 0 already scored, 0 failed" in cap.out
    assert {rs.parse_run_dir(p, root)[2] for p in rs.iter_run_dirs(root)} == set(ids)
    for run_id in ids:
        new = json.loads(rs.metrics_path(run_id, root).read_text())
        assert rs.TUNING_VAL in new["mean_std"]
        assert all(rs.TUNING_SCORE in record for record in new["seeds"].values())
        for seed in new["seeds"]:
            assert new["seeds"][seed]["test"] == old[run_id]["seeds"][seed]["test"]
            assert new["seeds"][seed]["val"] == old[run_id]["seeds"][seed]["val"]
        backed = _backup_file(root, backup, run_id, rs.METRICS_FILE)
        assert json.loads(backed.read_text()) == old[run_id]
        provenance = _backup_file(root, backup, run_id, rs.PROVENANCE_FILE)
        assert provenance.read_bytes() == old_provenance[run_id]
    assert not list(backup.rglob("*.partial"))


def test_rescore_resumes_and_leaves_backups_alone(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, ids = store
    original = rescore._execute
    calls = []

    def interrupted(config, config_path, run_id, *args, **kwargs):
        calls.append(run_id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(config, config_path, run_id, *args, **kwargs)

    monkeypatch.setattr(rescore, "_execute", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(["rescore-runs", "--runs", str(root), "--backup", str(backup)])
    capsys.readouterr()
    monkeypatch.setattr(rescore, "_execute", original)
    files = [_backup_file(root, backup, i, rs.METRICS_FILE) for i in ids]
    files[0].write_text('{"marker": "kept"}')  # a rerun must not rewrite an existing backup
    before = {f: f.read_bytes() for f in files}
    done_first = rs.metrics_path(calls[0], root).read_bytes()

    code, cap = _rescore(root, backup, capsys)
    assert code == 0, cap.err
    assert "1 rerun, 1 already scored, 0 failed" in cap.out
    assert {f: f.read_bytes() for f in files} == before
    assert rs.metrics_path(calls[0], root).read_bytes() == done_first


def test_a_run_interrupted_after_writing_metrics_is_rerun(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    rs.status_path(ids[0], root).write_text(json.dumps({"status": "running"}))
    code, cap = _rescore(root, backup, capsys)
    assert code == 0
    assert "1 rerun, 1 already scored" in cap.out
    assert json.loads(rs.status_path(ids[0], root).read_text())["status"] == "done"


@pytest.mark.parametrize("inside", ["runs", "runs/nested/backup"])
def test_backup_inside_runs_refused_before_anything_changes(store, tmp_path, capsys, inside):
    root, _, ids = store
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code, cap = _rescore(root, tmp_path / inside, capsys)
    assert code == 2
    assert "outside --runs" in cap.err
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_failed_run_is_listed_and_the_rest_continue(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, ids = store
    first = rs.parse_run_dir(rs.iter_run_dirs(root)[0], root)[2]
    original = rescore._execute

    def flaky(config, config_path, run_id, *args, **kwargs):
        if run_id == first:
            raise RuntimeError("boom")
        return original(config, config_path, run_id, *args, **kwargs)

    monkeypatch.setattr(rescore, "_execute", flaky)
    code, cap = _rescore(root, backup, capsys)
    assert code == 1
    assert f"failed: {first}" in cap.out and "boom" in cap.err
    assert "1 rerun, 0 already scored, 1 failed" in cap.out


def test_run_whose_id_differs_from_its_directory_is_refused(store, capsys):
    root, backup, ids = store
    directory = rs.run_dir(ids[0], root)
    renamed = directory.with_name(f"{directory.name.rpartition('--')[0]}--000000000000")
    directory.rename(renamed)
    before = (renamed / rs.METRICS_FILE).read_bytes()
    code, cap = _rescore(root, backup, capsys)
    assert code == 1
    assert "differs from its directory" in cap.err and "failed: 000000000000" in cap.out
    assert (renamed / rs.METRICS_FILE).read_bytes() == before  # nothing was rerun
    assert (backup / renamed.relative_to(root) / rs.METRICS_FILE).read_bytes() == before


def test_never_finished_run_is_left_alone(store, capsys):
    root, backup, ids = store
    directory = rs.run_dir(ids[0], root)
    (directory / rs.METRICS_FILE).unlink()
    code, cap = _rescore(root, backup, capsys)
    assert code == 0
    assert "1 rerun" in cap.out and "1 never finished" in cap.out
    assert not (directory / rs.METRICS_FILE).exists()


def test_report_on_a_clean_rescore_has_no_mismatch(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    code, text = _compare(root, backup, capsys)
    assert code == 0, text
    assert "- runs before: 2\n- runs after: 2\n" in text
    assert "- ids only before: 0\n- ids only after: 0\n" in text
    assert "- test-metric mismatches: 0" in text
    assert "- runs not done with the score: 0" in text
    assert "## tor_vergata / time_block" in text


def _backup_of(root: Path, backup: Path, run_id: str, val: float, test: float) -> None:
    """A backup with the old in-sample `val` and test scores of `run_id`."""
    document = json.loads(rs.metrics_path(run_id, root).read_text())
    del document["mean_std"][rs.TUNING_VAL]
    for record in document["seeds"].values():
        record.pop(rs.TUNING_SCORE, None)
    document["mean_std"]["val"]["macro_f1"] = [val, 0.01]
    document["mean_std"]["test"]["macro_f1"] = [test, 0.02]
    for name, payload in (
        (rs.METRICS_FILE, json.dumps(document)),
        (rs.PROVENANCE_FILE, rs.provenance_path(run_id, root).read_text()),
    ):
        target = _backup_file(root, backup, run_id, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)


def test_report_shows_a_changed_cell_and_only_changed_cells(tmp_path, capsys):
    root, backup = tmp_path / "runs", tmp_path / "backup"
    # r1: in-sample picks `mem` before, the tuning score picks `hon` after
    _make_run(root, "mem", _report_config(["r1"], window=20), 1.00, 0.61, tuning=0.70)
    _make_run(root, "hon", _report_config(["r1"], window=40), 0.90, 0.72, tuning=0.85)
    # r2: the same run is selected on both rules
    _make_run(root, "sta", _report_config(["r2"]), 0.80, 0.70, tuning=0.75)
    _backup_of(root, backup, "mem", 1.00, 0.61)
    _backup_of(root, backup, "hon", 0.90, 0.72)
    _backup_of(root, backup, "sta", 0.80, 0.70)
    code, text = _compare(root, backup, capsys)
    assert code == 0, text
    assert "cells changed: 1 of 2" in text
    row = next(line for line in text.splitlines() if " mem " in line)
    assert " hon " in row and "F1 61.00" in row and "F1 72.00" in row
    assert " r2 " not in text


def test_report_groups_gap_346_plans_in_their_own_table(tmp_path, capsys):
    from tests.test_report import _gap_plans

    root, backup, split_root = tmp_path / "runs", tmp_path / "backup", tmp_path / "splits"
    _gap_plans(split_root, "padova_sharp", {"g64": 64, "g346": 346})
    # one cell (svm / bedroom) per plan: grouped together, `w346` would win both
    for run_id, split, window, tuning in (("w50", "g64", 50, 0.70), ("w346", "g346", 346, 0.90)):
        config = {
            **_report_config(["bedroom"], window=window),
            "dataset": "padova_sharp",
            "split": split,
        }
        _make_run(root, run_id, config, 0.80, 0.60, split_hash=split, tuning=tuning)
        _backup_of(root, backup, run_id, 0.80, 0.60)
    code = rescore_report(root, backup, split_root=split_root)
    text = capsys.readouterr().out
    assert code == 0, text
    assert "## padova_sharp / time_block\n" in text
    assert "## padova_sharp / time_block, gap_packets 346\n" in text
    assert "cells changed: 0 of 2" in text


def test_report_lists_a_test_mismatch_and_keeps_the_new_metrics(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    path = rs.metrics_path(ids[0], root)
    document = json.loads(path.read_text())
    document["seeds"]["1"]["test"]["accuracy"] += 0.01
    document["mean_std"]["test"]["accuracy"][0] += 0.005
    path.write_text(json.dumps(document))
    kept = path.read_bytes()
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert "- test-metric mismatches: 1" in text
    assert f"- {ids[0]}: seed 1, mean_std" in text
    assert path.read_bytes() == kept


def test_report_flags_run_count_and_id_changes(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    directory = rs.run_dir(ids[0], root)
    directory.rename(directory.with_name(f"{directory.name.rpartition('--')[0]}--000000000000"))
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- ids only before: 1 ({ids[0]})" in text
    assert "- ids only after: 1 (000000000000)" in text


def test_report_flags_runs_without_the_score(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    _as_old_run(ids[0], root)
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- runs not done with the score: 1 ({ids[0]})" in text


def test_report_refuses_a_bad_backup(tmp_path, capsys):
    root = tmp_path / "runs"
    root.mkdir()
    assert rescore_report(root, root / "inside") == 2
    assert "outside --runs" in capsys.readouterr().err
    assert rescore_report(root, tmp_path / "missing") == 2
    assert "no such directory" in capsys.readouterr().err


def _tree(found):
    """A `git status --porcelain` that returns each value of `found` in turn, then the last."""
    values = iter(found)
    last = [None]

    def status(_args):
        last[0] = next(values, last[0])
        return last[0]

    return status


def _files(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _mark_dirty(run_id: str, root: Path) -> None:
    path = rs.provenance_path(run_id, root)
    document = json.loads(path.read_text())
    document["git"]["dirty"] = True
    path.write_text(json.dumps(document))


def test_report_lists_an_interval_mismatch(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    path = rs.metrics_path(ids[0], root)
    document = json.loads(path.read_text())
    document["intervals"]["test"]["macro_f1"][0] -= 0.01
    path.write_text(json.dumps(document))
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- {ids[0]}: intervals" in text


def test_report_lists_lost_intervals(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    path = rs.metrics_path(ids[0], root)
    document = json.loads(path.read_text())
    del document["intervals"]
    path.write_text(json.dumps(document))
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- {ids[0]}: intervals" in text


def test_real_rerun_failure_keeps_old_metrics_and_is_retried(store, monkeypatch, capsys):
    from csi_bench.runner import run as run_module

    root, backup, ids = store
    first = rs.parse_run_dir(rs.iter_run_dirs(root)[0], root)[2]
    old = rs.metrics_path(first, root).read_bytes()
    original = run_module.run_intervals
    calls = []

    def flaky(run_id, *args, **kwargs):
        calls.append(run_id)
        if run_id == first and calls.count(first) == 1:
            raise RuntimeError("boom")
        return original(run_id, *args, **kwargs)

    monkeypatch.setattr(run_module, "run_intervals", flaky)
    code, cap = _rescore(root, backup, capsys)
    assert code == 1 and f"failed: {first}" in cap.out
    assert json.loads(rs.status_path(first, root).read_text())["status"] == "failed"
    assert rs.metrics_path(first, root).read_bytes() == old
    code, text = _compare(root, backup, capsys)
    assert code == 1 and f"- runs not done with the score: 1 ({first})" in text
    code, cap = _rescore(root, backup, capsys)
    assert code == 0 and "1 rerun, 1 already scored, 0 failed" in cap.out
    assert json.loads(rs.status_path(first, root).read_text())["status"] == "done"
    assert _compare(root, backup, capsys)[0] == 0


def test_backup_failure_reruns_nothing(store, tmp_path, capsys):
    root, _, _ = store
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")
    before = _files(root)
    code, cap = _rescore(root, blocked, capsys)
    assert code == 2 and "cannot back up" in cap.err
    assert _files(root) == before


def test_before_column_reads_the_backup_not_the_store(tmp_path, capsys):
    root, backup = tmp_path / "runs", tmp_path / "backup"
    # On store values alone `hon` wins the old rule and scores 72; the backup says `mem` did, at 33.
    _make_run(root, "mem", _report_config(["r1"], window=20), 0.50, 0.61, tuning=0.70)
    _make_run(root, "hon", _report_config(["r1"], window=40), 0.90, 0.72, tuning=0.85)
    _backup_of(root, backup, "mem", 1.00, 0.33)
    _backup_of(root, backup, "hon", 0.90, 0.72)
    code, text = _compare(root, backup, capsys)
    assert code == 1  # `mem`'s test score differs from its backup
    row = next(line for line in text.splitlines() if " mem " in line)
    assert " hon " in row and "F1 33.00" in row and "F1 72.00" in row


def test_rescore_workers_flag(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, _ = store
    args = ["rescore-runs", "--runs", str(root), "--backup", str(backup)]
    before = _files(root)
    assert main([*args, "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    assert _files(root) == before
    seen = []
    monkeypatch.setattr(rescore, "_execute", lambda *a, **k: seen.append(k["workers"]))
    assert main([*args, "--workers", "2"]) == 0
    assert seen == [2, 2]


def test_dirty_tree_is_refused_before_anything_changes(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, _ = store
    monkeypatch.setattr(rescore, "_git", _tree([" M src/x.py"]))
    before = _files(root)
    code, cap = _rescore(root, backup, capsys)
    assert code == 2 and "--allow-dirty" in cap.err
    assert _files(root) == before and not backup.exists()


def test_allow_dirty_runs_on_a_dirty_tree(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, _ = store
    monkeypatch.setattr(rescore, "_git", _tree([" M src/x.py"]))
    args = ["rescore-runs", "--runs", str(root), "--backup", str(backup), "--allow-dirty"]
    assert main(args) == 0
    assert "2 rerun" in capsys.readouterr().out


def test_backup_that_dirties_the_tree_reruns_nothing(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, ids = store
    monkeypatch.setattr(rescore, "_git", _tree(["", "?? backup/"]))
    before = rs.metrics_path(ids[0], root).read_bytes()
    code, cap = _rescore(root, backup, capsys)
    assert code == 2 and "git tree is dirty" in cap.err
    assert rs.metrics_path(ids[0], root).read_bytes() == before
    assert "rerun" not in cap.out


def test_clean_tree_repairs_a_dirty_scored_run(store, monkeypatch, capsys):
    from csi_bench.runner import rescore

    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    _mark_dirty(ids[0], root)
    code, cap = _rescore(root, backup, capsys)
    assert code == 0 and "1 rerun, 1 already scored" in cap.out
    assert not json.loads(rs.provenance_path(ids[0], root).read_text())["git"]["dirty"]
    _mark_dirty(ids[0], root)
    monkeypatch.setattr(rescore, "_git", _tree([" M src/x.py"]))
    args = ["rescore-runs", "--runs", str(root), "--backup", str(backup), "--allow-dirty"]
    assert main(args) == 0  # on a dirty tree the rerun would only be dirty again
    assert "0 rerun, 2 already scored" in capsys.readouterr().out


def test_report_hides_and_fails_dirty_runs_unless_allowed(tmp_path, capsys):
    root, backup = tmp_path / "runs", tmp_path / "backup"
    config = _report_config(["r1"], window=20)
    _make_run(root, "dty", config, 0.80, 0.61, tuning=0.90, dirty=True)
    _make_run(root, "cln", _report_config(["r1"], window=40), 0.90, 0.72, tuning=0.70)
    _backup_of(root, backup, "dty", 0.80, 0.61)
    _backup_of(root, backup, "cln", 0.90, 0.72)
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert "- dirty after the rescore (`report` hides them): 1" in text
    assert "cells changed: 0 of 1" in text  # `cln` stays selected, as `report` would show
    out = root.parent / "compare.md"
    args = ["rescore-report", "--runs", str(root), "--backup", str(backup), "--out", str(out)]
    assert main([*args, "--allow-dirty"]) == 0
    capsys.readouterr()
    assert "cells changed: 1 of 1" in out.read_text()  # the dirty run is best on the tuning score


def test_backup_holds_the_old_predictions(store, capsys):
    root, backup, ids = store
    old = rs.predictions_path(ids[0], 0, root).read_bytes()
    assert _rescore(root, backup, capsys)[0] == 0
    copy = _backup_file(root, backup, ids[0], "seeds/0/predictions.parquet")
    assert copy.read_bytes() == old


def test_new_backup_does_not_store_rescored_metrics_as_old(store, tmp_path, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    other = tmp_path / "other"
    code, cap = _rescore(root, other, capsys)
    assert code == 0 and "0 rerun, 2 already scored" in cap.out
    assert not other.exists()


def test_report_on_a_lost_backup_fails_instead_of_comparing_new_with_new(store, tmp_path, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    other = tmp_path / "other"
    other.mkdir()
    code, text = _compare(root, other, capsys)
    assert code == 1
    assert "- ids only after: 2" in text


def test_report_fails_a_scored_run_that_is_not_done(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    rs.status_path(ids[0], root).write_text(json.dumps({"status": "running"}))
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- runs not done with the score: 1 ({ids[0]})" in text


def test_report_fails_a_run_it_cannot_read(store, capsys):
    root, backup, ids = store
    assert _rescore(root, backup, capsys)[0] == 0
    rs.config_path(ids[0], root).unlink()
    code, text = _compare(root, backup, capsys)
    assert code == 1
    assert f"- runs not done with the score: 1 ({ids[0]})" in text


def test_empty_store_is_an_error_not_a_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # outside git, so the tree check passes
    missing = tmp_path / "typo"
    code, cap = _rescore(missing, tmp_path / "backup", capsys)
    assert code == 2 and "no runs under" in cap.err
    (tmp_path / "empty-backup").mkdir()
    assert rescore_report(missing, tmp_path / "empty-backup") == 2
    assert "nothing to compare" in capsys.readouterr().err
