"""Write and load committed split plans."""

import json
from dataclasses import asdict
from pathlib import Path

from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import DataError
from csi_bench.core.hashing import canonical_json, short_hash

PARTS = ("train", "val", "test")


def compute_split_hash(plan: SplitPlan) -> str:
    body = asdict(plan)
    del body["split_hash"]
    return short_hash(body)


def find_split(
    dataset: str,
    split: str | None = None,
    root: str | Path = "splits",
    hint: str = "set `split:` in the config",
) -> Path:
    """Path of the committed plan: the one for `split`, or the only plan of the dataset."""
    folder = Path(root) / dataset
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if split is not None:
        for f in files:
            if f.stem == split:
                return f
        known = ", ".join(f.stem for f in files) or "none"
        raise DataError(f"no split {split} for {dataset} in {folder} (known: {known})")
    if not files:
        raise DataError(f"no split plan found in {folder}; generate one first")
    if len(files) > 1:
        hashes = ", ".join(f.stem for f in files)
        raise DataError(f"several split plans in {folder} ({hashes}); {hint}")
    return files[0]


def write_split_plan(plan: SplitPlan, root: str | Path = "splits") -> Path:
    """Write the plan to `<root>/<dataset>/<split_hash>.json` and return the path."""
    path = Path(root) / plan.dataset / f"{plan.split_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(asdict(plan)) + "\n", encoding="utf-8")
    return path


def load_split_plan(path: str | Path, recordings: list[Recording]) -> SplitPlan:
    """Load a plan, checking it matches the given recordings' source hashes."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        plan = SplitPlan(
            dataset=raw["dataset"],
            splitter=raw["splitter"],
            params=raw["params"],
            gap_packets=raw["gap_packets"],
            blocks={
                rid: {part: (b[part][0], b[part][1]) for part in PARTS}
                for rid, b in raw["blocks"].items()
            },
            source_sha256=raw["source_sha256"],
            split_hash=raw["split_hash"],
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        raise DataError(f"{path}: unreadable split plan: {e}") from e
    if compute_split_hash(plan) != plan.split_hash:
        raise DataError(f"{path}: split_hash does not match content")
    by_id = {r.recording_id: r for r in recordings}
    if set(by_id) != set(plan.source_sha256) or set(by_id) != set(plan.blocks):
        missing = sorted(set(plan.source_sha256) - set(by_id))
        extra = sorted(set(by_id) - set(plan.source_sha256))
        raise DataError(f"{path}: recording sets differ (missing {missing}, unplanned {extra})")
    for rid, rec in sorted(by_id.items()):
        if rec.meta.source_sha256 != plan.source_sha256[rid]:
            raise DataError(f"{path}: source changed for recording {rid}")
    return plan
