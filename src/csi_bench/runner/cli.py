"""The `csi-bench` command line."""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from csi_bench.core import registry
from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.results_schema import DATASET_TABLE_SCHEMA, SCOPES, dataset_table_path
from csi_bench.core.splitplan import PARTS, find_split, load_split_plan
from csi_bench.runner.discover import discover
from csi_bench.runner.report import DEFAULT_REFERENCE, report
from csi_bench.runner.run import run, run_sweep

_COUNTS = ("n_packets", *PARTS)
_SCOPE_RECORDING, _SCOPE_CLASS, _SCOPE_ROOM, _SCOPE_TOTAL = SCOPES


def build_table(
    recordings: list[Recording], plan: SplitPlan, classes: Sequence[str]
) -> list[dict[str, Any]]:
    """Recording rows sorted by id, then class, room and total rows summing them."""
    recs = sorted(recordings, key=lambda r: r.recording_id)
    rows: list[dict[str, Any]] = []
    for r in recs:
        blocks = plan.blocks[r.recording_id]
        row = {
            "scope": _SCOPE_RECORDING,
            "recording_id": r.recording_id,
            "room": r.room,
            "class": classes[r.label],
            "n_packets": r.meta.n_packets,
            "downsample_factor": r.meta.downsample_factor,
            **{p: blocks[p][1] - blocks[p][0] for p in PARTS},
            "split_hash": plan.split_hash,
        }
        rows.append(row)

    def total(scope: str, keep, room: str | None, label: str | None) -> dict[str, Any]:
        sel = [x for x in rows if keep(x)]
        return {
            "scope": scope,
            "recording_id": None,
            "room": room,
            "class": label,
            "downsample_factor": None,
            **{c: sum(x[c] for x in sel) for c in _COUNTS},
            "split_hash": plan.split_hash,
        }

    recs_rows = list(rows)
    present = {x["class"] for x in recs_rows}
    for name in (c for c in classes if c in present):
        rows.append(total(_SCOPE_CLASS, lambda x, n=name: x["class"] == n, None, name))
    for room in sorted({x["room"] for x in recs_rows}):
        rows.append(total(_SCOPE_ROOM, lambda x, rm=room: x["room"] == rm, room, None))
    rows.append(total(_SCOPE_TOTAL, lambda x: x["scope"] == _SCOPE_RECORDING, None, None))
    return rows


def _format(rows: list[dict[str, Any]]) -> str:
    cols = [f.name for f in DATASET_TABLE_SCHEMA]
    cells = [[("" if r[c] is None else str(r[c])) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(line.rstrip() for line in lines)


def describe(dataset: str, split: str | None = None) -> int:
    try:
        discover()
    except (ImportError, ConfigError) as e:
        sys.stderr.write(f"error: adapter discovery failed: {e}\n")
        return 2
    try:
        source_cls = registry.get("dataset", dataset)
    except ConfigError:
        known = ", ".join(registry.names("dataset")) or "none"
        sys.stderr.write(f"unknown dataset {dataset!r}; registered datasets: {known}\n")
        return 2
    try:
        cls: Any = source_cls
        split_path = find_split(dataset, split, hint="choose one with --split")
        source = cls(cls.Params())
        recordings = source.load()
        plan = load_split_plan(split_path, recordings)
        rows = build_table(recordings, plan, source.classes)
    except (DataError, ConfigError, ValidationError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    path = dataset_table_path(dataset)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=DATASET_TABLE_SCHEMA), path)
    except OSError as e:
        sys.stderr.write(f"error: cannot write {path}: {e}\n")
        return 2
    sys.stdout.write(_format(rows) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="csi-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("describe", help="print class balance and packet counts of a dataset")
    d.add_argument("dataset")
    d.add_argument("--split", help="split hash to use when several plans exist")
    r = sub.add_parser("run", help="run one experiment config and store its result")
    r.add_argument("config", help="path to an ExperimentConfig YAML file")
    w = sub.add_parser("sweep", help="expand a sweep file and run every config sequentially")
    w.add_argument("file", help="path to a base config YAML with a `sweep:` block")
    w.add_argument("--force", action="store_true", help="re-execute every run")
    p = sub.add_parser("report", help="refresh runs/index.parquet and print the comparison table")
    p.add_argument("--runs", default="runs", help="results directory")
    p.add_argument("--all-variants", action="store_true", help="list every run's variant")
    p.add_argument("--include-dirty", action="store_true", help="include runs made on a dirty tree")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="published reference values")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    if args.command == "run":
        return run(args.config)
    if args.command == "sweep":
        return run_sweep(args.file, args.force)
    if args.command == "report":
        return report(args.runs, args.all_variants, args.include_dirty, args.reference)
    return describe(args.dataset, args.split)


if __name__ == "__main__":
    sys.exit(main())
