"""`csi-bench report`: refresh the run index and print the comparison table (AD-13)."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from csi_bench.core import results_schema as rs
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.splitplan import ROOM_HOLDOUT, find_split, read_plan_params
from csi_bench.results_api import load_metrics
from csi_bench.runner.migrate import find_legacy_dirs

DEFAULT_REFERENCE = "experiments/published_reference.yaml"
_SCORES = ("macro_f1", "accuracy")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method(config: dict) -> str:
    extractors = "+".join(sorted(e["name"] for e in config["extractors"]))
    selector = (config.get("feature_selection") or {}).get("name", "none")
    return f"{extractors} | {selector} | {config['head']['name']}"


def _variant(config: dict) -> str:
    parts = {
        "window_len": config["window_len"],
        "stride": config["stride"],
        "preprocessing": [s["name"] for s in config.get("preprocessing", [])],
        "extractors": {e["name"]: e["params"] for e in config["extractors"] if e["params"]},
        "pooler": (config.get("pooler") or {}).get("name"),
        "feature_selection": (config.get("feature_selection") or {}).get("params") or None,
        "head": config["head"]["params"] or None,
    }
    return json.dumps({k: v for k, v in parts.items() if v not in (None, {}, [])}, sort_keys=True)


def _room_column(config: dict, dataset: str, splitter: str, split_root: str | Path) -> str:
    """The report column: the room setting, or the direction (`r1→r2`) of a `room_holdout` plan."""
    if splitter == ROOM_HOLDOUT:
        try:
            params = read_plan_params(find_split(dataset, config.get("split"), split_root))
            return rs.room_direction(params["train_rooms"], params["test_rooms"])
        except (DataError, KeyError, TypeError):
            pass
    return rs.room_setting(config["selection"]["rooms"])


def _index_row(path: Path, root: Path, split_root: str | Path = "splits") -> dict[str, Any] | None:
    try:
        dataset, splitter, run_id = rs.parse_run_dir(path, root)
        config = yaml.safe_load(rs.config_path(run_id, root).read_text(encoding="utf-8"))
        status = _read_json(rs.status_path(run_id, root))["status"]
        row: dict[str, Any] = dict.fromkeys(f.name for f in rs.INDEX_TABLE_SCHEMA)
        row.update(
            run_id=run_id,
            status=status,
            role=config["role"],
            dataset=dataset,
            splitter=splitter,
            dirty=False,
            method=_method(config),
            room_setting=_room_column(config, dataset, splitter, split_root),
            window_len=config["window_len"],
            stride=config["stride"],
            variant=_variant(config),
            n_seeds=len(config["seeds"]),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, DataError):
        return None
    try:
        provenance = _read_json(rs.provenance_path(run_id, root))
        mean_std = _read_json(rs.metrics_path(run_id, root))["mean_std"]
        scores = {}
        for split in rs.SPLIT_NAMES:
            for score in _SCORES:
                mean, std = mean_std[split][score]
                scores[f"{split}_{score}_mean"] = mean
                scores[f"{split}_{score}_std"] = std
        row["split_hash"] = provenance["split_hash"]
        row["dirty"] = bool(provenance["git"]["dirty"])
        row.update(scores)
    except (OSError, ValueError, KeyError, TypeError, DataError):
        pass
    try:
        test = _read_json(rs.metrics_path(run_id, root))["intervals"]["test"]
        bounds = {
            f"test_{score}_{side}": float(test[score][i])
            for score in _SCORES
            for i, side in enumerate(("lo", "hi"))
        }
        single = ",".join(str(c) for c in test["single_recording_classes"])
        row.update(bounds, single_recording_classes=single)
    except (OSError, ValueError, KeyError, TypeError, IndexError, DataError):
        pass
    return row


def refresh_index(
    root: str | Path = "runs", split_root: str | Path = "splits"
) -> list[dict[str, Any]]:
    """Rebuild `runs/index.parquet` from the run directories and return its rows."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in rs.iter_run_dirs(root):
        row = _index_row(path, root, split_root)
        if row is not None:
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=rs.INDEX_TABLE_SCHEMA), rs.index_path(root))
    return rows


def _pm(row: dict, split: str, score: str) -> str:
    return f"{100 * row[f'{split}_{score}_mean']:.2f}±{100 * row[f'{split}_{score}_std']:.2f}"


def _interval(row: dict, score: str) -> str:
    lo, hi = row.get(f"test_{score}_lo"), row.get(f"test_{score}_hi")
    if lo is None or hi is None:
        return "[-]"
    return f"[{100 * lo:.2f}, {100 * hi:.2f}]"


def _single(row: dict) -> list[str]:
    return [c for c in (row.get("single_recording_classes") or "").split(",") if c]


def _cell(row: dict) -> str:
    """`F1 <mean> [lo, hi] acc <mean> [lo, hi]`, with `*` if a class has one test recording."""
    scores = " ".join(
        f"{label} {100 * row[f'test_{score}_mean']:.2f} {_interval(row, score)}"
        for label, score in (("F1", "macro_f1"), ("acc", "accuracy"))
    )
    return f"{scores} *" if _single(row) else scores


def _grid(header: list[str], body: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    return [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in [header, *body]
    ]


def _command(row: dict, root: Path) -> str:
    return f"csi-bench run {rs.config_path(row['run_id'], root)}"


def _best(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """The best-validation row per (method, room setting)."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["method"], r["room_setting"])
        if key not in best or r["val_macro_f1_mean"] > best[key]["val_macro_f1_mean"]:
            best[key] = r
    return best


def _leakage(random_side: list[dict], block_side: list[dict], published: dict | None) -> list[str]:
    """Test accuracy of the same method and room under the random split and the time-block split.

    The random side is the reference runs of the dataset's `random_window` section, the
    time-block side the comparison runs of its `time_block` section.
    """
    random_rows = _best(random_side)
    block_rows = _best(block_side)
    order = {s: i for i, s in enumerate(rs.ROOM_SETTINGS)}
    keys = sorted(
        set(random_rows) | set(block_rows), key=lambda k: (k[0], order.get(k[1], len(order)))
    )

    def acc(row: dict | None) -> float | None:
        return None if row is None else 100 * row["test_accuracy_mean"]

    body = []
    for method, room in keys:
        random_acc = acc(random_rows.get((method, room)))
        block_acc = acc(block_rows.get((method, room)))
        diff = None if random_acc is None or block_acc is None else random_acc - block_acc
        pub = None if published is None or random_acc is None else published.get(room)
        body.append(
            [
                method,
                room,
                "-" if random_acc is None else f"{random_acc:.2f}",
                "-" if block_acc is None else f"{block_acc:.2f}",
                "-" if diff is None else f"{diff:+.2f}",
                "-" if pub is None else f"{pub:.2f}",
            ]
        )
    return _grid(
        ["method", "room", "random split", "time-block split", "difference", "published"], body
    )


def _comparison(
    rows: list[dict],
    root: Path,
    published: dict | None,
    columns: tuple[str, ...] | list[str] = rs.ROOM_SETTINGS,
) -> list[str]:
    best = _best(rows)
    methods = sorted({m for m, _ in best})
    body = [[m, *(_cell(best[m, s]) if (m, s) in best else "-" for s in columns)] for m in methods]
    if published is not None:
        body.append(["published (accuracy)", *(f"{published[s]:.2f}" for s in columns)])
    lines = _grid(["method", *columns], body)
    lines += [
        f"* {s} / {m}: single test recording for {', '.join(_single(best[m, s]))}"
        for m in methods
        for s in columns
        if (m, s) in best and _single(best[m, s])
    ]
    lines.append("")
    lines += [
        f"{m} / {s}: {_command(best[m, s], root)}"
        for m in methods
        for s in sorted({s for _, s in best})
        if (m, s) in best
    ]
    return lines


def _variants(rows: list[dict]) -> list[str]:
    body = [
        [
            r["method"],
            r["room_setting"],
            r["variant"],
            _pm(r, "val", "macro_f1"),
            _pm(r, "test", "macro_f1"),
            _interval(r, "macro_f1"),
            _pm(r, "test", "accuracy"),
            r["run_id"],
        ]
        for r in sorted(rows, key=lambda r: (r["method"], r["room_setting"], r["variant"]))
    ]
    header = ["method", "room", "variant", "val F1", "test F1", "test F1 95%", "test acc"]
    return _grid([*header, "run_id"], body)


def _stability(rows: list[dict], root: Path) -> list[str]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if " | none | " in r["method"]:
            continue
        try:
            seeds = load_metrics(r["run_id"], root)["seeds"]
        except (OSError, ValueError):
            continue
        for record in seeds.values():
            names = record.get("feature_names")
            if names is None:
                continue
            key = (r["method"], r["room_setting"])
            totals[key] += 1
            for name in names:
                counts[key][name] += 1
    lines: list[str] = []
    for key in sorted(counts):
        lines.append(f"{key[0]} / {key[1]} ({totals[key]} selections)")
        ranked = sorted(counts[key].items(), key=lambda kv: (-kv[1], kv[0]))
        lines += [f"  {name}  {n / totals[key]:.2f}" for name, n in ranked]
    return lines


def _load_reference(path: str | Path) -> dict:
    try:
        published = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["published"]
        return {s: float(published[s]) for s in rs.ROOM_SETTINGS}
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"cannot read published reference {path}: {e}") from e


def report(
    root: str | Path = "runs",
    all_variants: bool = False,
    include_dirty: bool = False,
    reference_path: str | Path = DEFAULT_REFERENCE,
    split_root: str | Path = "splits",
) -> int:
    root = Path(root)
    try:
        rows = refresh_index(root, split_root)
    except OSError as e:
        sys.stderr.write(f"error: cannot write index: {e}\n")
        return 2
    usable = [
        r
        for r in rows
        if r["status"] == "done"
        and r["split_hash"]
        and r["val_macro_f1_mean"] is not None
        and (include_dirty or not r["dirty"])
    ]
    legacy = find_legacy_dirs(root)
    if legacy:
        sys.stderr.write(
            f"note: {len(legacy)} run directories are in the old layout and not reported; "
            f"run `csi-bench migrate-runs --runs {root}`\n"
        )
    missing = sum(1 for r in usable if r["test_macro_f1_lo"] is None)
    if missing:
        sys.stderr.write(
            f"note: {missing} runs have no confidence intervals; "
            f"run `csi-bench migrate-runs --runs {root}` to fill them from saved predictions "
            "(runs whose predictions are missing stay without)\n"
        )
    sections: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in usable:
        sections[r["dataset"], r["splitter"]].append(r)
    try:
        for (dataset, splitter), rows_in in sections.items():
            for role in ("comparison", "reference"):
                by_column: dict[str, set[str]] = defaultdict(set)
                for r in rows_in:
                    if r["role"] == role:
                        by_column[r["room_setting"]].add(r["split_hash"])
                for column, hashes in sorted(by_column.items()):
                    if len(hashes) > 1:
                        raise ConfigError(
                            f"{dataset} / {splitter}: {role} runs in column {column} use "
                            f"different split hashes ({', '.join(sorted(hashes))}); "
                            "report them separately by moving the others out of the store"
                        )
        has_reference = any(r["role"] == "reference" for r in usable)
        published = _load_reference(reference_path) if has_reference else None
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    if not usable:
        sys.stdout.write("no finished runs to report\n")
        return 0
    out: list[str] = []
    for dataset, splitter in sorted(sections):
        heading = f"{dataset} / {splitter}"
        out += ["", heading, "=" * len(heading), ""]
        out += _section(sections[dataset, splitter], root, all_variants, published, splitter)
    for dataset in sorted({d for d, _ in sections}):
        random_side = [
            r for r in sections.get((dataset, "random_window"), []) if r["role"] == "reference"
        ]
        block_side = [
            r for r in sections.get((dataset, "time_block"), []) if r["role"] == "comparison"
        ]
        if not random_side or not block_side:
            continue
        heading = f"Leakage, {dataset} (test accuracy %, random minus time-block)"
        out += ["", heading, ""]
        out += _leakage(random_side, block_side, published)
    sys.stdout.write("\n".join(out).strip("\n") + "\n")
    return 0


def _section(
    rows: list[dict],
    root: Path,
    all_variants: bool,
    published: dict | None,
    splitter: str | None = None,
) -> list[str]:
    """The comparison and reference tables of one `(dataset, splitter)` section.

    A `room_holdout` section has one column per direction present (`r1→r2`, `r2→r1`) and no
    published row; the others have the room columns `r1`, `r2`, `mixed`.
    """
    holdout = splitter == ROOM_HOLDOUT
    if holdout:
        published = None

    def columns(table: list[dict]) -> tuple[str, ...] | list[str]:
        return sorted({r["room_setting"] for r in table}) if holdout else rs.ROOM_SETTINGS

    comparison = [r for r in rows if r["role"] == "comparison"]
    reference = [r for r in rows if r["role"] == "reference"]
    out: list[str] = []
    if comparison:
        out += ["Comparison (test score of the best-validation variant)", ""]
        out += _comparison(comparison, root, None, columns(comparison))
        if all_variants:
            out += ["", "All variants", ""] + _variants(comparison)
        stability = _stability(comparison, root)
        if stability:
            out += ["", "Feature-selection stability (fraction of selections)", ""] + stability
    if reference:
        out += ["", "Reference runs (not comparable to the table above)", ""]
        out += _comparison(reference, root, published, columns(reference))
        if all_variants:
            out += [""] + _variants(reference)
    return out[1:] if out[:1] == [""] else out
