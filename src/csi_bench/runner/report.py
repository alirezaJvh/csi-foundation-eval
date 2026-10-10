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
from csi_bench.core.errors import ConfigError
from csi_bench.results_api import load_metrics

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
        "feature_selection": (config.get("feature_selection") or {}).get("params") or None,
        "head": config["head"]["params"] or None,
    }
    return json.dumps({k: v for k, v in parts.items() if v not in (None, {}, [])}, sort_keys=True)


def _index_row(run_id: str, root: Path) -> dict[str, Any] | None:
    try:
        config = yaml.safe_load(rs.config_path(run_id, root).read_text(encoding="utf-8"))
        status = _read_json(rs.status_path(run_id, root))["status"]
        row: dict[str, Any] = dict.fromkeys(f.name for f in rs.INDEX_TABLE_SCHEMA)
        row.update(
            run_id=run_id,
            status=status,
            role=config["role"],
            dataset=config["dataset"],
            dirty=False,
            method=_method(config),
            room_setting=rs.room_setting(config["selection"]["rooms"]),
            window_len=config["window_len"],
            stride=config["stride"],
            variant=_variant(config),
            n_seeds=len(config["seeds"]),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
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
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return row


def refresh_index(root: str | Path = "runs") -> list[dict[str, Any]]:
    """Rebuild `runs/index.parquet` from the run directories and return its rows."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            row = _index_row(path.name, root)
            if row is not None:
                rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=rs.INDEX_TABLE_SCHEMA), rs.index_path(root))
    return rows


def _pm(row: dict, split: str, score: str) -> str:
    return f"{100 * row[f'{split}_{score}_mean']:.2f}±{100 * row[f'{split}_{score}_std']:.2f}"


def _cell(row: dict) -> str:
    return f"F1 {_pm(row, 'test', 'macro_f1')} acc {_pm(row, 'test', 'accuracy')}"


def _grid(header: list[str], body: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    return [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in [header, *body]
    ]


def _command(row: dict, root: Path) -> str:
    return f"csi-bench run {rs.config_path(row['run_id'], root)}"


def _comparison(rows: list[dict], root: Path, published: dict | None) -> list[str]:
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["method"], r["room_setting"])
        if key not in best or r["val_macro_f1_mean"] > best[key]["val_macro_f1_mean"]:
            best[key] = r
    methods = sorted({m for m, _ in best})
    body = [
        [m, *(_cell(best[m, s]) if (m, s) in best else "-" for s in rs.ROOM_SETTINGS)]
        for m in methods
    ]
    if published is not None:
        body.append(["published (accuracy)", *(f"{published[s]:.2f}" for s in rs.ROOM_SETTINGS)])
    lines = _grid(["method", *rs.ROOM_SETTINGS], body)
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
            _pm(r, "test", "accuracy"),
            r["run_id"],
        ]
        for r in sorted(rows, key=lambda r: (r["method"], r["room_setting"], r["variant"]))
    ]
    return _grid(["method", "room", "variant", "val F1", "test F1", "test acc", "run_id"], body)


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
) -> int:
    root = Path(root)
    try:
        rows = refresh_index(root)
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
    comparison = [r for r in usable if r["role"] == "comparison"]
    reference = [r for r in usable if r["role"] == "reference"]
    out: list[str] = []
    try:
        hashes = sorted({r["split_hash"] for r in comparison})
        if len(hashes) > 1:
            raise ConfigError(
                f"runs use different split hashes ({', '.join(hashes)}); "
                "report them separately by moving the others out of runs/"
            )
        published = _load_reference(reference_path) if reference else None
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    if not usable:
        sys.stdout.write("no finished runs to report\n")
        return 0
    if comparison:
        out += ["Comparison (test score of the best-validation variant)", ""]
        out += _comparison(comparison, root, None)
        if all_variants:
            out += ["", "All variants", ""] + _variants(comparison)
        stability = _stability(comparison, root)
        if stability:
            out += ["", "Feature-selection stability (fraction of selections)", ""] + stability
    if reference:
        out += ["", "Reference runs (not comparable to the table above)", ""]
        out += _comparison(reference, root, published)
        if all_variants:
            out += [""] + _variants(reference)
    sys.stdout.write("\n".join(out).strip("\n") + "\n")
    return 0
