"""Import every module under csi_bench.adapters so adapters self-register."""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def discover(package: str = "csi_bench.adapters") -> list[str]:
    pkg = importlib.import_module(package)
    imported = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        importlib.import_module(info.name)
        imported.append(info.name)
        logger.debug("discovered %s", info.name)
    return imported
