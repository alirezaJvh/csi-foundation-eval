"""Write and load committed split plans."""

import json
from dataclasses import asdict
from pathlib import Path

from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.hashing import canonical_json, short_hash

PARTS = ("train", "val", "test")
ROOM_HOLDOUT = "room_holdout"


def compute_split_hash(plan: SplitPlan) -> str:
    body = asdict(plan)
    del body["split_hash"]
    if body["unit"] == "block":
        del body["unit"]  # keeps the hashes of plans written before `unit` existed
    return short_hash(body)


def find_split(
    dataset: str,
    split: str | None = None,
    root: str | Path = "splits",
    hint: str = "set `split:` in the config",
) -> Path:
    """Path of the committed plan: the one for `split`, or the only block plan of the dataset.

    With `split` unset, window-unit plans (random_window) and `room_holdout` plans are
    never chosen implicitly.
    """
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
    blocks = [f for f in files if read_plan_unit(f) == "block" and not _is_room_holdout(f)]
    if not blocks:
        raise DataError(f"no block split plan in {folder}; a window plan needs an explicit split")
    if len(blocks) > 1:
        hashes = ", ".join(f.stem for f in blocks)
        raise DataError(f"several split plans in {folder} ({hashes}); {hint}")
    return blocks[0]


def _is_room_holdout(path: Path) -> bool:
    try:
        return read_plan_splitter(path) == ROOM_HOLDOUT
    except DataError:
        return False  # unreadable plans are reported when they are loaded


def read_plan_unit(path: str | Path) -> str:
    """The `unit` of a committed plan without loading recordings."""
    try:
        return str(json.loads(Path(path).read_text(encoding="utf-8")).get("unit", "block"))
    except (OSError, ValueError, AttributeError) as e:
        raise DataError(f"{path}: unreadable split plan: {e}") from e


def read_plan_splitter(path: str | Path) -> str:
    """The `splitter` name of a committed plan without loading recordings."""
    try:
        return str(json.loads(Path(path).read_text(encoding="utf-8"))["splitter"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DataError(f"{path}: unreadable split plan: {e}") from e


def read_plan_params(path: str | Path) -> dict:
    """The `params` of a committed plan without loading recordings."""
    try:
        params = json.loads(Path(path).read_text(encoding="utf-8"))["params"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DataError(f"{path}: unreadable split plan: {e}") from e
    if not isinstance(params, dict):
        raise DataError(f"{path}: unreadable split plan: params is not a mapping")
    return params


def check_split_rooms(splitter: str, params: dict, rooms: list[str], role: str) -> None:
    """A `room_holdout` split is for comparison runs only (a reference run's whole-recording
    fit would see the test room) and needs `selection.rooms` == `train_rooms` ∪ `test_rooms`.
    """
    if splitter != ROOM_HOLDOUT:
        return
    if role != "comparison":
        raise ConfigError("a room_holdout split requires role: comparison")
    try:
        planned = set(params["train_rooms"]) | set(params["test_rooms"])
    except (KeyError, TypeError) as e:
        raise ConfigError(f"room_holdout plan has no usable train/test rooms: {e}") from e
    if set(rooms) != planned:
        raise ConfigError(
            f"selection.rooms {sorted(set(rooms))} must equal the room_holdout plan's "
            f"train_rooms ∪ test_rooms {sorted(planned)}"
        )


def check_split_role(unit: str, role: str) -> None:
    """Window-level splits leak across neighbouring windows: reference runs only (AD-6)."""
    if unit == "window" and role != "reference":
        raise ConfigError("a window-unit split (random_window) requires role: reference")


def write_split_plan(plan: SplitPlan, root: str | Path = "splits") -> Path:
    """Write the plan to `<root>/<dataset>/<split_hash>.json` and return the path."""
    path = Path(root) / plan.dataset / f"{plan.split_hash}.json"
    body = asdict(plan)
    if body["unit"] == "block":
        del body["unit"]  # block plans omit the key, as in `compute_split_hash`
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(body) + "\n", encoding="utf-8")
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
            unit=raw.get("unit", "block"),
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
