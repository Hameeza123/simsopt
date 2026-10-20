#!/usr/bin/env python3
"""
Create quick SVG plots from an augmented-lagrangian history CSV without external deps.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_history(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if vmin == vmax:
        return [vmin - 1.0, vmin, vmin + 1.0]
    span = vmax - vmin
    raw_step = span / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(abs(raw_step)))
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    ticks = []
    val = start
    limit = vmax + step * 0.5
    while val <= limit:
        ticks.append(val)
        val += step
    if len(ticks) < 2:
        ticks = [vmin, vmax]
    return ticks


def draw_plot(
    xvals: List[float],
    series: Dict[str, List[Optional[float]]],
    out_path: Path,
    title: str,
    y_label: str,
    logy: bool = False,
) -> None:
    width = 1100
    height = 700
    left = 90
    right = 330
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#17becf",
    ]

    yvals = []
    for vals in series.values():
        for y in vals:
            if y is None:
                continue
            if logy and y <= 0:
                continue
            yvals.append(math.log10(y) if logy else y)

    if not yvals:
        yvals = [0.0, 1.0]
    y_min, y_max = min(yvals), max(yvals)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.06 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    x_min, x_max = min(xvals), max(xvals)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        yt = math.log10(y) if logy else y
        return top + (y_max - yt) / (y_max - y_min) * plot_h

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{left}" y="35" font-family="Helvetica,Arial,sans-serif" font-size="24" '
        f'font-weight="600">{title}</text>'
    )
    lines.append(
        f'<text x="22" y="{top + plot_h/2:.1f}" transform="rotate(-90 22 {top + plot_h/2:.1f})" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="16">{y_label}</text>'
    )
    lines.append(
        f'<text x="{left + plot_w/2:.1f}" y="{height - 25}" text-anchor="middle" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="16">AL iteration row</text>'
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>'
    )

    y_ticks = nice_ticks(y_min, y_max, count=6)
    for y_tick in y_ticks:
        ypix = top + (y_max - y_tick) / (y_max - y_min) * plot_h
        label_val = 10 ** y_tick if logy else y_tick
        lines.append(
            f'<line x1="{left}" y1="{ypix:.2f}" x2="{left + plot_w}" y2="{ypix:.2f}" '
            f'stroke="#e9e9e9" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{ypix + 4:.2f}" text-anchor="end" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="13">{label_val:.3g}</text>'
        )

    for xv in xvals:
        xpix = sx(xv)
        lines.append(
            f'<line x1="{xpix:.2f}" y1="{top}" x2="{xpix:.2f}" y2="{top + plot_h}" '
            f'stroke="#f3f3f3" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xpix:.2f}" y="{top + plot_h + 22}" text-anchor="middle" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="13">{int(xv)}</text>'
        )

    legend_x = left + plot_w + 20
    legend_y = top + 20
    for idx, (name, vals) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        path_parts = []
        drawing = False
        for x, y in zip(xvals, vals):
            if y is None or (logy and y <= 0):
                drawing = False
                continue
            cmd = "M" if not drawing else "L"
            path_parts.append(f"{cmd}{sx(x):.2f},{sy(y):.2f}")
            drawing = True
        if path_parts:
            lines.append(
                f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
        for x, y in zip(xvals, vals):
            if y is None or (logy and y <= 0):
                continue
            lines.append(
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.6" fill="{color}" stroke="white" stroke-width="1"/>'
            )
        ly = legend_y + idx * 28
        lines.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 26}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 34}" y="{ly + 4}" font-family="Helvetica,Arial,sans-serif" '
            f'font-size="14">{name}</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def pull(rows: List[Dict[str, str]], key: str) -> List[Optional[float]]:
    return [parse_float(r.get(key, "")) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AL history CSV into SVG files.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("examples/3_Advanced/auglag/outputs_s1/codex_s1_AL_history.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/3_Advanced/auglag/outputs_s1"),
    )
    args = parser.parse_args()

    rows = read_history(args.history)
    if not rows:
        raise ValueError(f"No rows in {args.history}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    xvals = [float(i) for i in range(len(rows))]

    c_cols = [k for k in rows[0].keys() if k.startswith("c_") and k != "c_norm"]
    mu_cols = [k for k in rows[0].keys() if k.startswith("mu_")]
    lambda_cols = [k for k in rows[0].keys() if k.startswith("lambda_")]

    draw_plot(
        xvals,
        {"c_norm": pull(rows, "c_norm"), **{c: pull(rows, c) for c in c_cols}},
        args.out_dir / "codex_s1_constraints.svg",
        "Constraint Residual History",
        "constraint value",
        logy=False,
    )
    draw_plot(
        xvals,
        {c: pull(rows, c) for c in mu_cols},
        args.out_dir / "codex_s1_penalties_log.svg",
        "Penalty Parameters",
        "mu_i",
        logy=True,
    )
    draw_plot(
        xvals,
        {c: pull(rows, c) for c in lambda_cols},
        args.out_dir / "codex_s1_lagrange_multipliers.svg",
        "Lagrange Multipliers",
        "lambda_i",
        logy=False,
    )
    draw_plot(
        xvals,
        {
            "time_sec": pull(rows, "time_sec"),
            "grad_norm": pull(rows, "grad_norm"),
            "delta_x_norm": pull(rows, "delta_x_norm"),
        },
        args.out_dir / "codex_s1_solver_metrics.svg",
        "Solver Metrics",
        "value",
        logy=True,
    )

    print(f"Plotted {len(rows)} rows from {args.history}")
    print(f"Wrote SVG files to {args.out_dir}")


if __name__ == "__main__":
    main()
