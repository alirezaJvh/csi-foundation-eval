import pytest
from pydantic import BaseModel

from csi_bench.core import registry
from csi_bench.core.errors import ConfigError, DataError, ExtractorError
from csi_bench.core.hashing import canonical_json, short_hash


@pytest.fixture(autouse=True)
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def _adapter():
    class A:
        class Params(BaseModel):
            pass

        VERSION = 1

    return A


def test_errors_are_exceptions():
    for e in (ConfigError, DataError, ExtractorError):
        assert issubclass(e, Exception)


def test_hash_key_order_independent():
    a, b = {"x": 1, "y": {"b": 2, "a": 3}}, {"y": {"a": 3, "b": 2}, "x": 1}
    assert short_hash(a) == short_hash(b)
    assert len(short_hash(a)) == 12
    assert canonical_json(a) == '{"x":1,"y":{"a":3,"b":2}}'


def test_register_and_get():
    cls = registry.register("splitter", "x")(_adapter())
    assert registry.get("splitter", "x") is cls


def test_duplicate_raises():
    registry.register("splitter", "x")(_adapter())
    with pytest.raises(ConfigError):
        registry.register("splitter", "x")(_adapter())


def test_unknown_raises():
    with pytest.raises(ConfigError):
        registry.get("splitter", "nope")


def test_missing_params_or_version_raises():
    with pytest.raises(ConfigError):
        registry.register("s", "a")(type("B", (), {"VERSION": 1}))
    with pytest.raises(ConfigError):
        registry.register("s", "b")(type("C", (), {"Params": BaseModel}))


@pytest.mark.parametrize("version", [True, "1", None])
def test_bad_version_raises(version):
    cls = type("D", (), {"Params": BaseModel, "VERSION": version})
    with pytest.raises(ConfigError):
        registry.register("s", "v")(cls)


def test_non_pydantic_params_raises():
    cls = type("E", (), {"Params": 5, "VERSION": 1})
    with pytest.raises(ConfigError):
        registry.register("s", "p")(cls)


def test_names_filters_by_stage_and_sorts():
    for stage, name in [("b", "z"), ("a", "y"), ("a", "x")]:
        registry.register(stage, name)(_adapter())
    assert registry.names("a") == ["x", "y"]
    assert registry.names("none") == []


def test_nan_rejected():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
