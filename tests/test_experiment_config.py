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
