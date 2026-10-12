import pytest

from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import Metrics
from csi_bench.core.errors import DataError

CLASSES = ("empty", "sitting")


def _metrics(split: str, accuracy: float, macro_f1: float) -> Metrics:
    return Metrics(
        split=split,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class={c: {"precision": 1.0, "recall": 1.0, "f1": 1.0} for c in CLASSES},
        confusion_matrix=((1, 0), (0, 1)),
        class_names=CLASSES,
    )


def test_mean_std_over_two_seeds():
    doc = rs.metrics_document(
        {
            0: {"val": _metrics("val", 0.5, 0.4), "test": _metrics("test", 0.6, 0.5)},
            1: {"val": _metrics("val", 0.7, 0.6), "test": _metrics("test", 0.6, 0.5)},
        }
    )
    assert doc["class_names"] == list(CLASSES)
    assert set(doc["seeds"]) == {"0", "1"}
    assert doc["mean_std"]["val"]["accuracy"] == pytest.approx([0.6, 0.1])
    assert doc["mean_std"]["val"]["macro_f1"] == pytest.approx([0.5, 0.1])
    assert doc["mean_std"]["test"]["accuracy"] == pytest.approx([0.6, 0.0])


def test_mean_std_single_seed_has_zero_std():
    doc = rs.metrics_document({0: {"val": _metrics("val", 0.8, 0.7)}})
    assert doc["mean_std"]["val"]["accuracy"] == pytest.approx([0.8, 0.0])


def test_metrics_round_trip():
    from csi_bench.core.contracts import Metrics

    m = Metrics(
        split="val",
        accuracy=0.5,
        macro_f1=0.4,
        per_class={"a": {"precision": 1.0, "recall": 0.5, "f1": 0.6}},
        confusion_matrix=((1, 2), (3, 4)),
        class_names=("a", "b"),
    )
    assert rs.metrics_from_dict(rs.metrics_as_dict(m)) == m


def _cfg(**changes):
    return {
        "extractors": [{"name": "paper_beamforming"}, {"name": "Amplitude Stats"}],
        "feature_selection": None,
        "head": {"name": "svm"},
        "selection": {"rooms": ["r2", "r1"]},
        "window_len": 40,
        **changes,
    }


def test_run_slug_sorts_lowercases_and_sanitises():
    assert rs.run_slug(_cfg()) == "amplitude_stats+paper_beamforming-none-svm-mixed-w40"
    slug = rs.run_slug(_cfg(feature_selection={"name": "mRMR/x"}, selection={"rooms": ["r1"]}))
    assert slug == "amplitude_stats+paper_beamforming-mrmr_x-svm-r1-w40"
    assert rs.run_slug(_cfg(head={"name": "a.b c"})).split("-")[2] == "a_b_c"


def test_run_dir_lookup_none_one_many(tmp_path):
    with pytest.raises(FileNotFoundError):
        rs.run_dir("abc123", tmp_path)
    one = rs.new_run_dir(tmp_path, "ds", "time_block", "slug", "abc123")
    one.mkdir(parents=True)
    (tmp_path / "_duplicates" / "x" / "y--abc123").mkdir(parents=True)
    (tmp_path / "ds" / "time_block" / "slug--zabc123").mkdir()
    assert rs.run_dir("abc123", tmp_path) == one
    assert rs.metrics_path("abc123", tmp_path) == one / "metrics.json"
    assert rs.parse_run_dir(one, tmp_path) == ("ds", "time_block", "abc123")
    rs.new_run_dir(tmp_path, "ds", "random_window", "other", "abc123").mkdir(parents=True)
    with pytest.raises(DataError, match="several"):
        rs.run_dir("abc123", tmp_path)


def test_iter_run_dirs_skips_underscore_and_non_layout(tmp_path):
    good = rs.new_run_dir(tmp_path, "ds", "sp", "s", "id1")
    good.mkdir(parents=True)
    (tmp_path / "_dataset" / "a" / "b--c").mkdir(parents=True)
    (tmp_path / "legacy" / "0123456789ab").mkdir(parents=True)
    (tmp_path / "ds" / "sp" / "noseparator").mkdir()
    assert rs.iter_run_dirs(tmp_path) == [good]
    assert rs.iter_run_dirs(tmp_path / "missing") == []
    with pytest.raises(DataError):
        rs.parse_run_dir(tmp_path / "ds" / "sp" / "noseparator", tmp_path)


def test_index_schema_gains_interval_fields_at_the_end():
    names = rs.INDEX_TABLE_SCHEMA.names
    added = [name for name, _ in rs.INTERVAL_FIELDS]
    assert added == [
        "test_macro_f1_lo",
        "test_macro_f1_hi",
        "test_accuracy_lo",
        "test_accuracy_hi",
        "single_recording_classes",
    ]
    assert names[-len(added) :] == added
    assert names[: -len(added)][-1] == "test_accuracy_std"
