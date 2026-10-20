#!/usr/bin/env python3
"""
Combine multiple AL_S1_FC maxmode histories into one constraints residual plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_codex_s1_history import draw_plot, parse_float, read_history


def sort_c_cols(cols):
    def key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**9
    return sorted(cols, key=key)


def pull(rows, key):
    return [parse_float(r.get(key, "")) for r in rows]


def parse_modes(modes_str):
    return [int(x.strip()) for x in modes_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot combined constraints for AL_S1_FC max modes.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="If set, histories are taken from this run folder using --modes.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="1,2,3",
        help="Comma-separated modes used only with --run-dir.",
    )
    parser.add_argument(
        "--histories",
        nargs="+",
        type=Path,
        default=[
            Path("examples/3_Advanced/auglag/outputs_s1/AL_S1_FC_maxmode1_AL_history.csv"),
            Path("examples/3_Advanced/auglag/outputs_s1/AL_S1_FC_maxmode2_AL_history.csv"),
            Path("examples/3_Advanced/auglag/outputs_s1/AL_S1_FC_maxmode3_AL_history.csv"),
        ],
        help="Ordered AL history CSV files to concatenate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/3_Advanced/auglag/outputs_s1/AL_S1_FC_combined_constraints.svg"),
    )
    args = parser.parse_args()

    histories = args.histories
    if args.run_dir is not None:
        modes = parse_modes(args.modes)
        histories = [args.run_dir / f"AL_S1_FC_maxmode{m}_AL_history.csv" for m in modes]

    all_rows = []
    row_counts = []
    kept_paths = []
    for p in histories:
        if not p.exists():
            print(f"Skipping missing history: {p}")
            continue
        rows_i = read_history(p)
        kept_paths.append(p)
        all_rows.append(rows_i)
        row_counts.append(len(rows_i))
    rows = [r for rows_i in all_rows for r in rows_i]
    if not rows:
        raise ValueError("No rows found in the provided histories.")

    c_cols = set()
    for r in rows:
        for k in r.keys():
            if k.startswith("c_") and k != "c_norm":
                c_cols.add(k)
    c_cols = sort_c_cols(c_cols)

    xvals = [float(i) for i in range(len(rows))]
    series = {"c_norm": pull(rows, "c_norm")}
    for c in c_cols:
        series[c] = pull(rows, c)

    draw_plot(
        xvals,
        series,
        args.out,
        title="Combined Constraint Residual History",
        y_label="constraint value",
        logy=False,
    )

    for p, n in zip(kept_paths, row_counts):
        print(f"{p}: {n} rows")
    print(f"combined rows: {len(rows)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
