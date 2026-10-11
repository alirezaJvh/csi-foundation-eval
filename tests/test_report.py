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
    root, run_id, config, val, test, split_hash="h1", dirty=False, status="done", names=None
):
    d = rs.run_dir(run_id, root)
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
    assert "F1 60.00±2.00 acc 61.00±2.00" in out
    assert "70.00" not in out
    assert f"csi-bench run {runs}/bbb/config.yaml" in out
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
    assert out == ""


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
    assert code == 0 and "F1 60.00±2.00" in out


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


def test_leakage_block_random_minus_time_block(runs, capsys, monkeypatch):
    # r1 has both sides; the mrmr method (r2) only the comparison side
    monkeypatch.chdir(ROOT)
    ref = _config(["r1"], role="reference")
    _make_run(runs, "r_r1", ref, 0.9, 0.98, split_hash=WINDOW_PLAN)
    leak = _leakage_lines(runs, capsys)
    assert any("random split" in line and "time-block split" in line for line in leak)
    r1 = next(line for line in leak if line.split()[:1] == ["amplitude_stats"] and " r1 " in line)
    # bbb is the best-validation comparison run for r1: accuracy 0.61; reference 0.99
    assert r1.split()[-4:] == ["99.00", "61.00", "+38.00", "99.41"]
    # no reference side: no difference and no published number
    r2 = next(line for line in leak if " r2 " in line)
    assert r2.split()[-4:] == ["-", "51.00", "-", "-"]


def test_leakage_ignores_reference_runs_off_a_window_plan(runs, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    ref = _config(["r1"], role="reference")
    _make_run(runs, "r_block", ref, 0.9, 0.98, split_hash=BLOCK_PLAN)
    _make_run(runs, "r_unknown", _config(["r2"], role="reference"), 0.9, 0.98, split_hash="nope")
    leak = _leakage_lines(runs, capsys)
    r1 = next(line for line in leak if line.split()[:1] == ["amplitude_stats"] and " r1 " in line)
    assert r1.split()[-4:] == ["-", "61.00", "-", "-"]
    assert not any("99.00" in line for line in leak)
