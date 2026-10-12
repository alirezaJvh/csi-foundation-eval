import sys

import pytest
import yaml
from pydantic import BaseModel, Field

from csi_bench.config.experiment import compute_run_id, load_config, save_config
from csi_bench.core import registry
from csi_bench.core.errors import ConfigError
from csi_bench.runner.discover import discover

BASE = {
    "dataset": "tor_vergata",
    "selection": {"rooms": ["r1"]},
    "window_len": 40,
    "extractors": [{"name": "amplitude_stats"}],
    "head": {"name": "logreg"},
    "role": "comparison",
}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _write(tmp_path, data) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return str(path)


def _load(tmp_path, **changes):
    data = {**BASE, **changes}
    return load_config(_write(tmp_path, data))


def test_load_valid(tmp_path):
    config = _load(tmp_path)
    assert config.dataset == "tor_vergata"
    assert config.role == "comparison"
    assert config.seeds == [0]
    assert config.split is None
    assert config.stride == 1


def test_bad_field_names_path(tmp_path):
    with pytest.raises(ConfigError, match="window_len"):
        _load(tmp_path, window_len="x")


def test_unknown_key(tmp_path):
    with pytest.raises(ConfigError, match="bogus"):
        _load(tmp_path, bogus=1)


def test_missing_role(tmp_path):
    data = {k: v for k, v in BASE.items() if k != "role"}
    with pytest.raises(ConfigError, match="role"):
        load_config(_write(tmp_path, data))


def test_unknown_head_lists_registered(tmp_path):
    with pytest.raises(ConfigError, match="logreg"):
        _load(tmp_path, head={"name": "nope"})


def test_bad_head_params_names_path(tmp_path):
    with pytest.raises(ConfigError, match="head.params"):
        _load(tmp_path, head={"name": "logreg", "params": {"C_grid": "x"}})


def test_run_id_round_trip(tmp_path):
    config = _load(tmp_path)
    saved = tmp_path / "saved.yaml"
    save_config(config, saved)
    assert compute_run_id(load_config(saved)) == compute_run_id(config)


def test_run_id_excludes_seeds(tmp_path):
    one = _load(tmp_path)
    two = _load(tmp_path, seeds=[0, 1, 2])
    assert compute_run_id(one) == compute_run_id(two)


def test_run_id_changes_with_config(tmp_path):
    assert compute_run_id(_load(tmp_path)) != compute_run_id(_load(tmp_path, window_len=30))
    assert compute_run_id(_load(tmp_path)) != compute_run_id(
        _load(tmp_path, head={"name": "logreg", "params": {"C_grid": [1.0]}})
    )


def test_run_id_excludes_affects_result_false(tmp_path):
    class FakeHead:
        VERSION = 1

        class Params(BaseModel):
            c: float = 1.0
            threads: int = Field(default=1, json_schema_extra={"affects_result": False})

        def fit(self, *args): ...

        def predict(self, *args): ...

    registry.register("head", "fake")(FakeHead)
    same_a = _load(tmp_path, head={"name": "fake", "params": {"c": 1.0, "threads": 1}})
    same_b = _load(tmp_path, head={"name": "fake", "params": {"c": 1.0, "threads": 16}})
    other = _load(tmp_path, head={"name": "fake", "params": {"c": 2.0, "threads": 1}})
    assert compute_run_id(same_a) == compute_run_id(same_b)
    assert compute_run_id(same_a) != compute_run_id(other)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"window_len": 0}, "window_len"),
        ({"stride": 0}, "stride"),
        ({"extractors": []}, "extractors"),
        ({"seeds": []}, "seeds"),
        ({"seeds": [1, 1]}, "seeds"),
        ({"selection": {"rooms": []}}, "rooms"),
        ({"dataset": "nope"}, "dataset"),
        ({"head": {"name": "logreg", "params": {"C_grd": [1]}}}, "head.params"),
        (
            {"extractors": [{"name": "amplitude_stats", "params": {"unknown": 1}}]},
            "extractors.0.params",
        ),
        (
            {"extractors": [{"name": "amplitude_stats"}, {"name": "amplitude_stats"}]},
            "extractors",
        ),
    ],
)
def test_invalid_fields_name_the_field(tmp_path, changes, field):
    with pytest.raises(ConfigError, match=field):
        _load(tmp_path, **changes)


def test_run_id_follows_the_split_plan(tmp_path):
    config = _load(tmp_path)
    only_a = tmp_path / "a"
    only_b = tmp_path / "b"
    for root, name in ((only_a, "aaaaaaaaaaaa"), (only_b, "bbbbbbbbbbbb")):
        (root / "tor_vergata").mkdir(parents=True)
        (root / "tor_vergata" / f"{name}.json").write_text("{}")
    assert compute_run_id(config, only_a) != compute_run_id(config, only_b)


def _steps(config):
    return [(s.name, s.params) for s in config.preprocessing]


def test_preprocessing_default_empty(tmp_path):
    assert _load(tmp_path).preprocessing == []


def test_preprocessing_shorthand_and_explicit(tmp_path):
    config = _load(
        tmp_path,
        preprocessing=[
            {"antennas": [0, 1]},
            {"global_rescale": {}},
            {"name": "recording_mean_norm", "params": {}},
        ],
    )
    assert _steps(config) == [
        ("antennas", {"ids": [0, 1]}),
        ("global_rescale", {}),
        ("recording_mean_norm", {}),
    ]


def test_preset_equals_written_out(tmp_path):
    preset = _load(tmp_path, preprocessing=["ant2_rescale"])
    explicit = _load(tmp_path, preprocessing=[{"antennas": [0, 1]}, {"global_rescale": {}}])
    assert _steps(preset) == _steps(explicit)
    assert compute_run_id(preset) == compute_run_id(explicit)


def test_chain_order_changes_run_id(tmp_path):
    a = _load(tmp_path, preprocessing=[{"antennas": [0, 1]}, {"global_rescale": {}}])
    b = _load(tmp_path, preprocessing=[{"global_rescale": {}}, {"antennas": [0, 1]}])
    assert compute_run_id(a) != compute_run_id(b)
    assert compute_run_id(a) != compute_run_id(_load(tmp_path))


def test_saved_config_holds_explicit_steps(tmp_path):
    config = _load(tmp_path, preprocessing=["ant2"])
    saved = save_config(config, tmp_path / "out" / "config.yaml")
    assert yaml.safe_load(saved.read_text())["preprocessing"] == [
        {"name": "antennas", "params": {"ids": [0, 1]}}
    ]


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("nope", r"preprocessing\.0"),
        ({"bogus": {}}, r"preprocessing\.0"),
        ({"antennas": {"wrong": 1}}, r"preprocessing\.0"),
        ({"antennas": []}, r"preprocessing\.0"),
    ],
)
def test_bad_preprocessing_names_path(tmp_path, entry, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, preprocessing=[entry])


def test_fit_scope_whole_recording_needs_reference(tmp_path):
    step = {"recording_mean_norm": {"fit_scope": "whole_recording"}}
    assert _load(tmp_path, preprocessing=[step], role="reference")
    with pytest.raises(ConfigError, match=r"preprocessing\.0\.params\.fit_scope"):
        _load(tmp_path, preprocessing=[step], role="comparison")


def test_window_shorter_than_extractor_minimum_is_rejected(tmp_path):
    with pytest.raises(
        ConfigError, match=r"extractors\.1\.name: window_len 17 < min_window_len 18"
    ):
        _load(
            tmp_path,
            window_len=17,
            extractors=[{"name": "amplitude_stats"}, {"name": "paper_beamforming"}],
        )


def test_feature_selection_shorthand_normalizes(tmp_path):
    manual = _load(tmp_path, feature_selection={"manual": ["amplitude_stats.std"]})
    assert manual.feature_selection is not None
    assert manual.feature_selection.model_dump() == {
        "name": "manual",
        "params": {"columns": ["amplitude_stats.std"]},
    }
    mrmr = _load(tmp_path, feature_selection={"mrmr": {"k": [2, 3]}})
    assert mrmr.feature_selection is not None
    assert mrmr.feature_selection.params == {"k": [2, 3]}
    saved = save_config(mrmr, tmp_path / "out" / "config.yaml")
    reloaded = load_config(saved)
    assert reloaded.feature_selection == mrmr.feature_selection
    assert compute_run_id(reloaded) == compute_run_id(mrmr)


def test_feature_selection_null_keeps_run_id(tmp_path):
    base = compute_run_id(_load(tmp_path))
    assert compute_run_id(_load(tmp_path, feature_selection=None)) == base
    selected = _load(tmp_path, feature_selection={"mrmr": {"k": [2]}})
    assert compute_run_id(selected) != base
    other_k = _load(tmp_path, feature_selection={"mrmr": {"k": [3]}})
    assert compute_run_id(other_k) != compute_run_id(selected)
    manual = _load(tmp_path, feature_selection={"manual": ["amplitude_stats"]})
    assert compute_run_id(manual) not in (base, compute_run_id(selected))


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"manual": ["a"], "mrmr": {"k": [1]}}, "exactly one"),
        ({}, "exactly one"),
        ({"other": 1}, "exactly one"),
        ({"manual": []}, "feature_selection"),
        ({"manual": "a"}, "list"),
        ({"mrmr": {"k": []}}, "feature_selection"),
        ({"mrmr": {"k": [2, 2]}}, "unique"),
        ({"mrmr": {"k": [0]}}, "feature_selection"),
        ({"mrmr": {"k": [1], "bogus": 1}}, "bogus"),
        ({"mrmr": [2]}, "mapping"),
        ({"name": "wrapper", "params": {}}, "unknown selector"),
        ("manual", "mapping"),
    ],
)
def test_bad_feature_selection_shape(tmp_path, entry, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, feature_selection=entry)


ROOM_HOLDOUT_PLANS = ("1e8688472a1f", "c4faaf799ad5")  # r1→r2, r2→r1 (committed)


@pytest.mark.parametrize("split", ROOM_HOLDOUT_PLANS)
def test_room_holdout_needs_both_rooms(tmp_path, split):
    config = _load(tmp_path, split=split, selection={"rooms": ["r2", "r1"]})
    assert config.split == split
    with pytest.raises(ConfigError, match=r"\['r1'\].*\['r1', 'r2'\]"):
        _load(tmp_path, split=split, selection={"rooms": ["r1"]})
    with pytest.raises(ConfigError, match="room_holdout split requires role: comparison"):
        _load(tmp_path, split=split, selection={"rooms": ["r1", "r2"]}, role="reference")


def test_implicit_split_ignores_room_holdout_plans(tmp_path):
    from csi_bench.config.experiment import resolved_body

    config = _load(tmp_path)
    assert resolved_body(config)["split"] not in ROOM_HOLDOUT_PLANS
