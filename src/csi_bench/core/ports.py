"""Stage ports (typing.Protocol) implemented by adapters."""

from typing import Protocol

from csi_bench.core.contracts import Recording, SplitPlan


class DatasetSource(Protocol):
    @property
    def classes(self) -> tuple[str, ...]: ...

    def load(self) -> list[Recording]:
        """Return all recordings, sorted by recording_id."""
        ...


class Splitter(Protocol):
    def split(self, recordings: list[Recording], dataset: str) -> SplitPlan:
        """Return the train/val/test packet blocks for every recording."""
        ...
