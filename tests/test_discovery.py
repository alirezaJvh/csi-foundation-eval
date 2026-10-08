import sys
import textwrap

import pytest

from csi_bench.core import registry
from csi_bench.runner.discover import discover


@pytest.fixture(autouse=True)
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def test_discovery_registers_fixture_adapter(tmp_path, monkeypatch):
    pkg = tmp_path / "fakeadapters"
    (pkg / "splitters").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "splitters" / "__init__.py").write_text("")
    (pkg / "splitters" / "x.py").write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel
            from csi_bench.core.registry import register

            @register("splitter", "x")
            class X:
                class Params(BaseModel):
                    pass

                VERSION = 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        found = discover("fakeadapters")
        assert "fakeadapters.splitters.x" in found
        assert registry.get("splitter", "x").__name__ == "X"
    finally:
        for m in [m for m in sys.modules if m.startswith("fakeadapters")]:
            del sys.modules[m]


def test_discovery_of_real_package_runs():
    assert isinstance(discover(), list)
