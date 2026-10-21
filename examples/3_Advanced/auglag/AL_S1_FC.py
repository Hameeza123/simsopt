#!/usr/bin/env python
"""
Augmented Lagrangian optimization for stellarator boundary shape with constraints.

Optimizes boundary Fourier modes to achieve:
1) Target aspect ratio
2) Quasi-symmetry (QS) at specific radii
3) Target mean rotational transform (iota)

Uses finite difference gradients (no analytical derivatives required).
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.solve.AL_nograd import augmented_lagrangian_method
from simsopt.util import MpiPartition, proc0_print
from AL_constraint_objects import AspectConstraint, QsConstraint, MeanIotaConstraint
import time

def _to_list(val):
    if val is None:
        return []
    if np.isscalar(val):
        return [float(val)]
    return [float(v) for v in np.ravel(np.asarray(val))]


def _build_log_callback(history_rows):
    def _logger(payload):
        row = {
            "stage": payload.get("stage"),
            "k": payload.get("k"),
            "c_norm": payload.get("c_norm"),
            "omega_k": payload.get("omega_k"),
            "eta_k": payload.get("eta_k"),
            "aug_lag": payload.get("aug_lag"),
            "time_sec": payload.get("time_sec"),
            "grad_norm": payload.get("grad_norm"),
            "minimize_success": payload.get("minimize_success"),
            "minimize_status": payload.get("minimize_status"),
            "minimize_message": payload.get("minimize_message"),
            "minimize_nit": payload.get("minimize_nit"),
            "delta_x_norm": payload.get("delta_x_norm"),
        }

        for i, v in enumerate(_to_list(payload.get("x"))):
            row[f"x_{i}"] = v
        for i, v in enumerate(_to_list(payload.get("c_vals"))):
            row[f"c_{i}"] = v
        for i, v in enumerate(_to_list(payload.get("lag_mul"))):
            row[f"lambda_{i}"] = v
        for i, v in enumerate(_to_list(payload.get("mu"))):
            row[f"mu_{i}"] = v

        history_rows.append(row)

    return _logger


def _write_history_csv(history_path, rows):
    if not rows:
        return

    x_count = max(len([k for k in r.keys() if k.startswith("x_") and k[2:].isdigit()]) for r in rows)
    c_count = max(len([k for k in r.keys() if k.startswith("c_") and k[2:].isdigit()]) for r in rows)
    lam_count = max(len([k for k in r.keys() if k.startswith("lambda_") and k[7:].isdigit()]) for r in rows)
    mu_count = max(len([k for k in r.keys() if k.startswith("mu_") and k[3:].isdigit()]) for r in rows)

    fieldnames = [
        "stage",
        "k",
        "c_norm",
        "omega_k",
        "eta_k",
        "aug_lag",
        "time_sec",
        "grad_norm",
        "minimize_success",
        "minimize_status",
        "minimize_message",
        "minimize_nit",
        "delta_x_norm",
    ]
    fieldnames += [f"x_{i}" for i in range(x_count)]
    fieldnames += [f"c_{i}" for i in range(c_count)]
    fieldnames += [f"lambda_{i}" for i in range(lam_count)]
    fieldnames += [f"mu_{i}" for i in range(mu_count)]

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_modes(modes_str):
    modes = [int(x.strip()) for x in modes_str.split(",") if x.strip()]
    if not modes:
        raise ValueError("At least one mode is required.")
    if any(m <= 0 for m in modes):
        raise ValueError("Modes must be positive integers.")
    return modes


def _build_parser():
    p = argparse.ArgumentParser(description="Run AL S1 Fourier-continuation by mode.")
    p.add_argument("--modes", type=str, default="1,2,3", help="Comma-separated max modes for continuation.")
    p.add_argument("--maxiter", type=int, default=5, help="Inner optimizer iteration limit.")
    p.add_argument("--maxiter-lag", type=int, default=2, help="Outer AL iteration limit.")
    p.add_argument("--mu-init", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--c-tol", type=float, default=1e-5)
    p.add_argument("--run-tag", type=str, default="", help="Optional run identifier for output folder.")
    p.add_argument(
        "--output-root",
        type=str,
        default="examples/3_Advanced/auglag/outputs_s1",
        help="Root folder where run subfolder is created.",
    )
    return p


def main():
    args = _build_parser().parse_args()
    modes = _parse_modes(args.modes)

    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    run_dir = output_root / f"AL_S1_FC_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc0_print("Running AL_S1_FC.py")
    proc0_print("=============================================")
    proc0_print(f"Modes: {modes}")
    proc0_print(f"Output run folder: {run_dir}")

    mpi = MpiPartition(25)
    filename = os.path.join(os.path.dirname(__file__), "../inputs", "input.nfp4_QH_warm_start")
    vmec = Vmec(filename, mpi=mpi, verbose=False)

    surf = vmec.boundary
    surf.fix_all()

    # Fixed targets for stage-1 constraints
    A_target = 6.8
    radii = np.array([0, 0.1, 1.01])
    iota_target = -0.42

    for max_mode in modes:
        surf.fix_all()
        vmec.indata.mpol = max_mode + 1
        vmec.indata.ntor = vmec.indata.mpol
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")

        c_aspect = AspectConstraint(surf, target=A_target)
        qs_obj = QuasisymmetryRatioResidual(vmec, radii, helicity_m=1, helicity_n=-1)
        vmec.run()
        qs_constraints = [QsConstraint(vmec, surf, qs_obj, k) for k in range(len(radii))]
        mean_iota_term = MeanIotaConstraint(vmec, target=iota_target)
        equality_constraints = [c_aspect, *qs_constraints, mean_iota_term]

        surf.to_vtk(filename=str(run_dir / f"initial_surf_maxmode{max_mode}"))
        proc0_print(f"Initial AR: A(x0) = {vmec.aspect():.6f}, target = {A_target:.6f}")
        proc0_print(f"Initial QS obj: {qs_obj.total():.6e}")
        proc0_print(f"QS profile at radii {radii}: {qs_obj.profile()}")
        proc0_print(f"Initial Mean iota: {vmec.mean_iota():.6f}, target = {mean_iota_term.target:.6f}")
        proc0_print(f"Starting AL optimization for max_mode={max_mode}")

        start_time = time.time()
        history_rows = []
        log_callback = _build_log_callback(history_rows)
        x, fnc, lag_mul = augmented_lagrangian_method(
            f=None,
            equality_constraints=equality_constraints,
            MAXITER=args.maxiter,
            MAXITER_lag=args.maxiter_lag,
            mu_init=args.mu_init,
            tau=args.tau,
            c_tol=args.c_tol,
            verbose=False,
            log_callback=log_callback,
        )
        end_time = time.time()
        proc0_print(f"Time taken: {end_time - start_time:.2f} seconds")

        surf.x = x
        vmec.run()
        proc0_print(f"Optimization finished for max_mode = {max_mode}")
        proc0_print(f"Final AR: A(x_final) = {vmec.aspect():.6f}, target = {A_target:.6f}")
        proc0_print(f"Final QS obj: {qs_obj.total():.6e}")
        proc0_print(f"QS profile at radii {radii}: {qs_obj.profile()}")
        proc0_print(f"Final Mean iota: {vmec.mean_iota():.6f}, target = {mean_iota_term.target:.6f}")

        surf.to_vtk(filename=str(run_dir / f"final_surf_maxmode{max_mode}_LBFGS{args.maxiter}_AL{args.maxiter_lag}_"))
        history_filename = run_dir / f"AL_S1_FC_maxmode{max_mode}_AL_history.csv"
        _write_history_csv(str(history_filename), history_rows)
        proc0_print(f"Saved AL history to {history_filename}")
        proc0_print("=============================================")


if __name__ == "__main__":
    main()
