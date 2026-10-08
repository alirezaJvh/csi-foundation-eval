import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _lint(cwd: Path) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    assert exe
    return subprocess.run(
        [exe],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(cwd / "src")},
    )


def test_project_contracts_pass():
    assert _lint(ROOT).returncode == 0


@pytest.mark.parametrize(
    ("path", "source", "contract"),
    [
        ("core/bad.py", "import csi_bench.adapters\n", "core imports nothing internal"),
        ("adapters/s/a.py", "import csi_bench.config\n", "adapters import only core"),
        ("adapters/s/a.py", "import csi_bench.runner\n", "adapters import only core"),
        ("adapters/s/a.py", "import csi_bench.adapters.s.b\n", "adapters never import each other"),
        ("config/bad.py", "import csi_bench.adapters\n", "only runner imports adapters"),
    ],
)
def test_violating_module_fails(tmp_path, path, source, contract):
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    pkg = tmp_path / "src/csi_bench/adapters/s"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "b.py").write_text("")
    target = tmp_path / "src/csi_bench" / path
    target.write_text(source)
    r = _lint(tmp_path)
    assert r.returncode != 0
    assert f"{contract} BROKEN" in r.stdout
