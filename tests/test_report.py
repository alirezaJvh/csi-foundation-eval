import ast
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from csi_bench import results_api
from csi_bench.core import results_schema as rs
from csi_bench.runner.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _config(rooms, head="svm", extractor="amplitude_stats", fs=None, window=40, role="comparison"):
    return {
        "dataset": "tor_vergata",
        "selection": {"rooms": rooms},
        "preprocessing": [],
        "window_len": window,
        "stride": 1,
        "extractors": [{"name": extractor, "params": {}}],
        "feature_selection": fs,
        "head": {"name": head, "params": {}},
        "seeds": [0, 1],
        "role": role,
    }


def _make_run(
    root,
    run_id,
    config,
    val,
    test,
    split_hash="h1",
    dirty=False,
    status="done",
    names=None,
    splitter="time_block",
    intervals=None,
):
    d = rs.new_run_dir(root, config["dataset"], splitter, rs.run_slug(config), run_id)
    d.mkdir(parents=True)
    rs.config_path(run_id, root).write_text(yaml.safe_dump(config))
    rs.status_path(run_id, root).write_text(json.dumps({"status": status}))
    if status != "done":
        return
    rs.provenance_path(run_id, root).write_text(
        json.dumps({"split_hash": split_hash, "git": {"dirty": dirty}})
    )
    seeds = {str(s): {"feature_names": names} if names else {} for s in (0, 1)}
    rs.metrics_path(run_id, root).write_text(
        json.dumps(
            {
                "seeds": seeds,
                "mean_std": {
                    "val": {"macro_f1": [val, 0.01], "accuracy": [val, 0.01]},
                    "test": {"macro_f1": [test, 0.02], "accuracy": [test + 0.01, 0.02]},
                },
                **({"intervals": {"test": intervals}} if intervals else {}),
            }
        )
    )


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "aaa", _config(["r1"], window=20), 0.80, 0.70)
    _make_run(root, "bbb", _config(["r1"], window=40), 0.90, 0.60)
    _make_run(root, "ccc", _config(["r1", "r2"]), 0.70, 0.65)
    _make_run(
        root,
        "ddd",
        _config(["r2"], fs={"name": "mrmr", "params": {"k": 2}}),
        0.5,
        0.5,
        names=["amp_mean", "amp_std"],
    )
    _make_run(
        root,
        "eee",
        _config(["r2"], fs={"name": "mrmr", "params": {"k": 2}}, window=20),
        0.5,
        0.5,
        names=["amp_mean", "amp_max"],
    )
    return root


def _report(root, capsys, *args):
    code = main(["report", "--runs", str(root), *args])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_table_picks_best_val_variant_and_refreshes_index(runs, capsys):
    code, out, _ = _report(runs, capsys)
    assert code == 0
    # bbb has the best val F1 for r1, so its test score (60.00) is shown, not aaa's (70.00)
    assert "F1 60.00 [-] acc 61.00 [-]" in out
    assert "70.00" not in out
    assert f"csi-bench run {rs.config_path('bbb', runs)}" in out
    assert f"{runs}/tor_vergata/time_block/amplitude_stats-none-svm-r1-w40--bbb/" in out
    index = pd.read_parquet(rs.index_path(runs))
    assert set(index.run_id) == {"aaa", "bbb", "ccc", "ddd", "eee"}
    assert set(index.room_setting) == {"r1", "r2", "mixed"}


def test_all_variants_lists_every_run(runs, capsys):
    _, out, _ = _report(runs, capsys, "--all-variants")
    for run_id in ("aaa", "bbb", "ccc"):
        assert run_id in out


def test_stability_counts_selected_features(runs, capsys):
    _, out, _ = _report(runs, capsys)
    assert "amplitude_stats | mrmr | svm / r2 (4 selections)" in out
    assert "amp_mean  1.00" in out
    assert "amp_std  0.50" in out


def test_mixed_split_hashes_refused(runs, capsys):
    _make_run(runs, "fff", _config(["r1"], head="knn"), 0.5, 0.5, split_hash="h2")
    code, out, err = _report(runs, capsys)
    assert code == 2
    assert "h1" in err and "h2" in err
    assert "tor_vergata / time_block" in err and "column r1" in err
    assert out == ""


def test_different_hashes_in_other_columns_or_sections_are_fine(runs, capsys):
    # one r1 run on another time_block plan would clash; a different column or section does not
    for run_id in ("ddd", "eee"):  # the whole r2 column moves to another plan
        rs.provenance_path(run_id, runs).write_text(
            json.dumps({"split_hash": "h3", "git": {"dirty": False}})
        )
    _make_run(
        runs,
        "fff",
        _config(["r1"], head="knn"),
        0.5,
        0.5,
        split_hash="h2",
        splitter="random_window",
    )
    code, out, _ = _report(runs, capsys)
    assert code == 0
    assert "tor_vergata / random_window" in out and "tor_vergata / time_block" in out


def test_one_section_per_dataset_and_splitter(runs, capsys):
    ref = _config(["r1"], head="logreg", role="reference")
    _make_run(runs, "rrr", ref, 0.9, 0.95, split_hash="random", splitter="random_window")
    code, out, _ = _report(
        runs, capsys, "--reference", str(ROOT / "experiments/published_reference.yaml")
    )
    assert code == 0
    random_at = out.index("tor_vergata / random_window")
    block_at = out.index("tor_vergata / time_block")
    leak_at = out.index("Leakage")
    assert random_at < block_at < leak_at
    random_part = out[random_at:block_at]
    assert "logreg" in random_part and "svm" not in random_part
    assert "logreg" not in out[block_at:leak_at]
    index = pd.read_parquet(rs.index_path(runs))
    assert dict(zip(index.run_id, index.splitter, strict=True))["rrr"] == "random_window"
    assert set(index.splitter) == {"random_window", "time_block"}


def test_unknown_splitter_runs_get_a_section(runs, capsys):
    _make_run(
        runs, "uuu", _config(["r1"], head="knn"), 0.5, 0.5, split_hash="h9", splitter="unknown"
    )
    code, out, _ = _report(runs, capsys)
    assert code == 0 and "tor_vergata / unknown" in out


def test_legacy_dirs_noted_on_stderr(runs, capsys):
    legacy = runs / "0123456789ab"
    legacy.mkdir()
    (legacy / "config.yaml").write_text(yaml.safe_dump(_config(["r1"])))
    code, out, err = _report(runs, capsys)
    assert code == 0 and "F1 60.00 [-]" in out
    assert "old layout" in err and "csi-bench migrate-runs" in err


def test_dirty_runs_excluded_unless_included(runs, capsys):
    _make_run(runs, "ggg", _config(["r1"], head="knn"), 0.99, 0.99, dirty=True)
    _, out, _ = _report(runs, capsys)
    assert "knn" not in out
    _, out, _ = _report(runs, capsys, "--include-dirty")
    assert "knn" in out


def test_failed_runs_ignored(runs, capsys):
    _make_run(runs, "hhh", _config(["r1"], head="knn"), 0, 0, status="failed")
    _, out, _ = _report(runs, capsys)
    assert "knn" not in out
    assert "hhh" in set(pd.read_parquet(rs.index_path(runs)).run_id)


def test_reference_rows_shown_separately_with_published(runs, capsys):
    ref = _config(["r1"], head="logreg", role="reference")
    _make_run(runs, "rrr", ref, 0.9, 0.95, split_hash="random")
    code, out, _ = _report(
        runs, capsys, "--reference", str(ROOT / "experiments/published_reference.yaml")
    )
    assert code == 0
    head, _, tail = out.partition("Reference runs")
    assert "logreg" not in head and "logreg" in tail
    assert "99.41" in tail and "99.59" in tail and "97.37" in tail


def test_missing_reference_file_is_an_error(runs, capsys):
    _make_run(runs, "rrr", _config(["r1"], role="reference"), 0.9, 0.95, split_hash="random")
    code, _, err = _report(runs, capsys, "--reference", str(runs / "nope.yaml"))
    assert code == 2 and "published reference" in err


def test_no_runs(tmp_path, capsys):
    code, out, _ = _report(tmp_path / "runs", capsys)
    assert code == 0 and "no finished runs" in out
    assert pd.read_parquet(rs.index_path(tmp_path / "runs")).empty


def test_results_api_reads_and_cannot_write(runs, capsys):
    _report(runs, capsys)
    assert len(results_api.load_index(runs)) == 5
    assert results_api.load_metrics("aaa", runs)["mean_std"]["val"]["macro_f1"][0] == 0.80
    public = [
        n for n in dir(results_api) if not n.startswith("_") and callable(getattr(results_api, n))
    ]
    assert not [n for n in public if n.startswith(("write", "save", "delete", "set"))]
    with pytest.raises(FileNotFoundError):
        results_api.load_index(runs / "missing")


def test_report_writes_only_the_index(runs, capsys):
    before = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    _report(runs, capsys)
    after = {p for p in runs.rglob("*") if p.is_file()}
    assert after - set(before) == {rs.index_path(runs)}
    assert all(p.read_bytes() == b for p, b in before.items())


def test_malformed_runs_do_not_abort_report(runs, capsys):
    _make_run(runs, "bad1", _config(["r1"]), 0.9, 0.9)
    rs.metrics_path("bad1", runs).write_text("{not json")
    _make_run(runs, "bad2", _config(["r1"]), 0.9, 0.9)
    rs.config_path("bad2", runs).write_text("- just\n- a list\n")
    _make_run(runs, "bad3", _config(["r1"]), 0.9, 0.9)
    rs.status_path("bad3", runs).write_text("[]")
    code, out, _ = _report(runs, capsys)
    assert code == 0 and "F1 60.00 [-]" in out


def test_notebooks_import_only_results_api():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                for mod in mods:
                    assert mod.split(".")[0] != "csi_bench" or mod == "csi_bench.results_api", (
                        path.name,
                        mod,
                    )


WINDOW_PLAN, BLOCK_PLAN = "499148ef70db", "e31e1b7dd9b4"  # committed under splits/tor_vergata


def _leakage_lines(runs, capsys):
    code, out, _ = _report(
        runs, capsys, "--reference", str(ROOT / "experiments/published_reference.yaml")
    )
    assert code == 0
    _, _, leak = out.partition("Leakage")
    return leak.splitlines()


def test_leakage_block_random_minus_time_block(runs, capsys):
    # r1 has both sides; the mrmr method (r2) only the comparison side
    ref = _config(["r1"], role="reference")
    _make_run(runs, "r_r1", ref, 0.9, 0.98, split_hash=WINDOW_PLAN, splitter="random_window")
    leak = _leakage_lines(runs, capsys)
    assert any("random split" in line and "time-block split" in line for line in leak)
    r1 = next(line for line in leak if line.split()[:1] == ["amplitude_stats"] and " r1 " in line)
    # bbb is the best-validation comparison run for r1: accuracy 0.61; reference 0.99
    assert r1.split()[-4:] == ["99.00", "61.00", "+38.00", "99.41"]
    # no reference side: no difference and no published number
    r2 = next(line for line in leak if " r2 " in line)
    assert r2.split()[-4:] == ["-", "51.00", "-", "-"]


def test_leakage_ignores_reference_runs_off_the_random_window_section(runs, capsys):
    ref = _config(["r1"], role="reference")
    _make_run(runs, "r_block", ref, 0.9, 0.98, split_hash=BLOCK_PLAN)
    _make_run(runs, "r_unknown", _config(["r2"], role="reference"), 0.9, 0.98, splitter="unknown")
    mixed = _config(["r1", "r2"], role="reference")
    _make_run(runs, "r_mix", mixed, 0.9, 0.97, split_hash=WINDOW_PLAN, splitter="random_window")
    leak = _leakage_lines(runs, capsys)
    r1 = next(line for line in leak if line.split()[:1] == ["amplitude_stats"] and " r1 " in line)
    assert r1.split()[-4:] == ["-", "61.00", "-", "-"]
    r2 = next(line for line in leak if line.split()[:1] == ["amplitude_stats"] and " r2 " in line)
    assert r2.split()[-4:] == ["-", "51.00", "-", "-"]
    assert not any("99.00" in line for line in leak)


def test_no_leakage_block_without_both_sections(runs, capsys):
    _make_run(runs, "r_block", _config(["r1"], role="reference"), 0.9, 0.98, split_hash="h1")
    code, out, _ = _report(
        runs, capsys, "--reference", str(ROOT / "experiments/published_reference.yaml")
    )
    assert code == 0 and "Reference runs" in out and "Leakage" not in out


def test_no_leakage_block_when_time_block_has_no_comparison_runs(tmp_path, capsys):
    root = tmp_path / "runs"
    ref = _config(["r1"], role="reference")
    _make_run(root, "r_win", ref, 0.9, 0.98, split_hash=WINDOW_PLAN, splitter="random_window")
    _make_run(root, "r_blk", _config(["r2"], role="reference"), 0.9, 0.97, split_hash=BLOCK_PLAN)
    code, out, _ = _report(
        root, capsys, "--reference", str(ROOT / "experiments/published_reference.yaml")
    )
    assert code == 0 and "tor_vergata / time_block" in out and "Leakage" not in out


def _holdout_plans(split_root):
    folder = split_root / "tor_vergata"
    folder.mkdir(parents=True)
    for name, train, test in (("h12", "r1", "r2"), ("h21", "r2", "r1")):
        params = {"train_rooms": [train], "test_rooms": [test]}
        plan = {"splitter": "room_holdout", "params": params}
        (folder / f"{name}.json").write_text(json.dumps(plan))


def test_room_holdout_section_has_direction_columns(runs, tmp_path, capsys):
    from csi_bench.runner.report import report

    split_root = tmp_path / "splits"
    _holdout_plans(split_root)
    for run_id, split, head, test in (("x12", "h12", "svm", 0.41), ("x21", "h21", "knn", 0.42)):
        config = {**_config(["r1", "r2"], head=head), "split": split}
        _make_run(runs, run_id, config, 0.9, test, split_hash=split, splitter="room_holdout")
    # `run` refuses reference runs on a holdout plan; one stored anyway still gets its own columns
    ref = {**_config(["r1", "r2"], head="logreg", role="reference"), "split": "h21"}
    _make_run(runs, "x99", ref, 0.9, 0.43, split_hash="h21", splitter="room_holdout")
    reference = ROOT / "experiments/published_reference.yaml"
    assert report(runs, reference_path=reference, split_root=split_root) == 0
    out = capsys.readouterr().out
    holdout = out[out.index("tor_vergata / room_holdout") : out.index("tor_vergata / time_block")]
    headers = [line.split() for line in holdout.splitlines() if line.startswith("method")]
    assert headers == [["method", "r1→r2", "r2→r1"], ["method", "r2→r1"]]
    assert "F1 41.00 [-]" in holdout and "F1 42.00 [-]" in holdout
    assert "published" not in holdout and "99.41" not in holdout
    index = pd.read_parquet(rs.index_path(runs))
    columns = dict(zip(index.run_id, index.room_setting, strict=True))
    assert (columns["x12"], columns["x21"], columns["aaa"]) == ("r1→r2", "r2→r1", "r1")


def test_room_holdout_hash_check_is_per_direction(runs, tmp_path, capsys):
    from csi_bench.runner.report import report

    split_root = tmp_path / "splits"
    _holdout_plans(split_root)
    planned = (("x12", "h12", "svm"), ("y12", "h12", "knn"), ("x21", "h21", "svm"))
    for run_id, split, head in planned:
        config = {**_config(["r1", "r2"], head=head), "split": split}
        hashed = "other" if run_id == "y12" else split
        _make_run(runs, run_id, config, 0.9, 0.5, split_hash=hashed, splitter="room_holdout")
    assert report(runs, split_root=split_root) == 2
    err = capsys.readouterr().err
    assert "column r1→r2" in err and "r2→r1" not in err


def test_room_holdout_without_plan_falls_back_to_rooms(runs, tmp_path, capsys):
    from csi_bench.runner.report import refresh_index

    config = {**_config(["r1", "r2"]), "split": "gone"}
    _make_run(runs, "x12", config, 0.9, 0.5, split_hash="gone", splitter="room_holdout")
    rows = refresh_index(runs, tmp_path / "nosplits")
    assert {r["run_id"]: r["room_setting"] for r in rows}["x12"] == "mixed"


INTERVALS = {
    "accuracy": [0.6012, 0.6234],
    "macro_f1": [0.5871, 0.6129],
    "n_resamples": 1000,
    "seed": 0,
    "level": 0.95,
    "unit": "recording",
    "single_recording_classes": [],
}


def test_cell_shows_intervals_and_index_has_bounds(runs, capsys):
    path = rs.metrics_path("bbb", runs)
    path.write_text(json.dumps({**json.loads(path.read_text()), "intervals": {"test": INTERVALS}}))
    code, out, err = _report(runs, capsys, "--all-variants")
    assert code == 0
    assert "F1 60.00 [58.71, 61.29] acc 61.00 [60.12, 62.34]" in out
    assert "F1 60.00 [58.71, 61.29] acc 61.00 [60.12, 62.34] *" not in out
    assert "60.00±2.00  [58.71, 61.29]  61.00±2.00" in out  # all-variants keeps the seed std
    assert "test F1 95%" in out
    assert "note: 4 runs have no confidence intervals" in err and "migrate-runs" in err
    index = pd.read_parquet(rs.index_path(runs)).set_index("run_id")
    assert index.loc["bbb", "test_macro_f1_lo"] == 0.5871
    assert index.loc["bbb", "test_accuracy_hi"] == 0.6234
    assert index.loc["bbb", "single_recording_classes"] == ""
    assert pd.isna(index.loc["aaa", "test_macro_f1_lo"])
    assert pd.isna(index.loc["aaa", "single_recording_classes"])


def test_single_recording_class_flags_cell_with_footnote(runs, capsys):
    flagged = {**INTERVALS, "single_recording_classes": ["empty", "sitting"]}
    _make_run(runs, "x01", _config(["r1"], head="knn"), 0.9, 0.9, intervals=flagged)
    code, out, err = _report(runs, capsys)
    assert code == 0
    assert "acc 91.00 [60.12, 62.34] *" in out
    assert "* r1 / amplitude_stats | none | knn: single test recording for empty, sitting" in out
    assert out.count("single test recording") == 1
    index = pd.read_parquet(rs.index_path(runs)).set_index("run_id")
    assert index.loc["x01", "single_recording_classes"] == "empty,sitting"


def test_no_interval_note_when_every_run_has_intervals(tmp_path, capsys):
    root = tmp_path / "runs"
    _make_run(root, "y01", _config(["r1"]), 0.9, 0.8, intervals=INTERVALS)
    code, out, err = _report(root, capsys)
    assert code == 0 and "[-]" not in out and "confidence intervals" not in err
