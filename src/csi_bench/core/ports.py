"""Stage ports (typing.Protocol) implemented by adapters."""

from typing import Protocol

from csi_bench.core.contracts import Recording


class DatasetSource(Protocol):
    @property
    def classes(self) -> tuple[str, ...]: ...

    def load(self) -> list[Recording]:
        """Return all recordings, sorted by recording_id."""
        ...
