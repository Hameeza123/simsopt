#!/usr/bin/env python3
"""
Simple 3-stage pipeline in examples/3_Advanced/auglag:
1) Stage 1: surface optimization
2) Stage 2: coil optimization (warm-started from stage 1)
3) Single-stage: joint coils + surface optimization (warm-started from stage 2)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveCurveDistance, CurveLength, LpCurveCurvature, create_equally_spaced_curves, curves_to_vtk
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.solve.AL_nograd import augmented_lagrangian_method
from simsopt.util import MpiPartition, proc0_print

from AL_constraint_objects import AspectConstraint, MeanIotaConstraint, QsConstraint

try:
    from simsopt.geo import CurveSurfaceDistance
except Exception:
    CurveSurfaceDistance = None

try:
    from simsopt.geo import LinkingNumber
except Exception:
    LinkingNumber = None


def parse_modes(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_parser():
    base = Path(__file__).resolve().parent
    default_vmec = base / "../inputs/input.nfp4_QH_warm_start"
    p = argparse.ArgumentParser(description="Simple 3-stage auglag workflow")
    p.add_argument("--vmec-input", type=str, default=str(default_vmec))
    p.add_argument("--output-dir", type=str, default=str(base / "outputs_codex_3stage"))
    p.add_argument("--max-modes", type=str, default="1")
    p.add_argument("--nphi", type=int, default=24)
    p.add_argument("--ntheta", type=int, default=24)
    p.add_argument("--stage1-iters", type=int, default=5)
    p.add_argument("--stage1-lag-iters", type=int, default=3)
    p.add_argument("--stage2-iters", type=int, default=20)
    p.add_argument("--stage2-lag-iters", type=int, default=8)
    p.add_argument("--single-iters", type=int, default=25)
    p.add_argument("--single-lag-iters", type=int, default=10)
    p.add_argument("--mu-init", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=4.0)
    p.add_argument("--c-tol", type=float, default=1e-5)
    p.add_argument("--aspect-target", type=float, default=6.8)
    p.add_argument("--iota-target", type=float, default=-0.42)
    p.add_argument("--qs-radii", type=float, nargs="+", default=[0.0, 0.1, 1.01])
    p.add_argument("--helicity-m", type=int, default=1)
    p.add_argument("--helicity-n", type=int, default=-1)
    p.add_argument("--ncoils", type=int, default=4)
    p.add_argument("--coil-order", type=int, default=5)
    p.add_argument("--coil-r1-factor", type=float, default=0.6)
    p.add_argument("--total-current", type=float, default=3e5)
    p.add_argument("--flux-threshold", type=float, default=1e-6)
    p.add_argument("--cc-threshold", type=float, default=0.1)
    p.add_argument("--cs-threshold", type=float, default=0.2)
    p.add_argument("--length-target", type=float, default=30.0)
    p.add_argument("--curvature-threshold", type=float, default=5.0)
    p.add_argument("--use-linking", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def stage1_constraints(vmec, surf, mode, args):
    surf.fix_all()
    vmec.indata.mpol = mode + 1
    vmec.indata.ntor = mode + 1
    surf.fixed_range(mmin=0, mmax=mode, nmin=-mode, nmax=mode, fixed=False)
    surf.fix("rc(0,0)")
    qs = QuasisymmetryRatioResidual(
        vmec,
        np.asarray(args.qs_radii),
        helicity_m=args.helicity_m,
        helicity_n=args.helicity_n,
    )
    vmec.run()
    c = [AspectConstraint(surf, target=args.aspect_target)]
    c += [QsConstraint(vmec, surf, qs, k) for k in range(len(args.qs_radii))]
    c.append(MeanIotaConstraint(vmec, target=args.iota_target))
    return c, qs


def build_initial_coils(surf, args):
    R0 = float(surf.get_rc(0, 0))
    R1 = args.coil_r1_factor * R0
    base_curves = create_equally_spaced_curves(
        args.ncoils,
        surf.nfp,
        stellsym=surf.stellsym,
        R0=R0,
        R1=R1,
        order=args.coil_order,
        numquadpoints=128,
    )
    base_currents = [Current(args.total_current / args.ncoils * 1e-5) * 1e5 for _ in range(args.ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
    curves = [c.curve for c in coils]
    return coils, curves, base_curves


def stage2_constraints(surf, bs, curves, base_curves, args):
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(surf, bs, definition="normalized", threshold=args.flux_threshold)
    Jls = [CurveLength(c) for c in base_curves]
    Jcs = [LpCurveCurvature(c, 2, args.curvature_threshold) for c in base_curves]
    c = [
        Jf,
        CurveCurveDistance(curves, args.cc_threshold, num_basecurves=args.ncoils),
        QuadraticPenalty(sum(Jls), args.length_target, "max"),
        sum(Jcs),
    ]
    if CurveSurfaceDistance is not None:
        c.append(CurveSurfaceDistance(curves, surf, args.cs_threshold))
    if LinkingNumber is not None and args.use_linking:
        c.append(LinkingNumber(curves, downsample=2))
    return c, Jf


def run_al(constraints, args, maxiter, maxiter_lag):
    return augmented_lagrangian_method(
        f=None,
        equality_constraints=constraints,
        MAXITER=maxiter,
        MAXITER_lag=maxiter_lag,
        mu_init=args.mu_init,
        tau=args.tau,
        c_tol=args.c_tol,
        verbose=args.verbose,
    )


class CombinedState:
    def __init__(self, coil_obj, surf, bs):
        self.coil_obj = coil_obj
        self.surf = surf
        self.bs = bs

    def get_x(self):
        return np.concatenate((np.asarray(self.coil_obj.x), np.asarray(self.surf.get_dofs())))

    def set_x(self, x):
        ncoil = len(self.coil_obj.x)
        self.coil_obj.x = np.asarray(x[:ncoil])
        self.surf.set_dofs(np.asarray(x[ncoil:]))
        self.bs.set_points(self.surf.gamma().reshape((-1, 3)))


class CombinedConstraint:
    def __init__(self, constraint, state):
        self.constraint = constraint
        self.state = state

    @property
    def x(self):
        return self.state.get_x()

    @x.setter
    def x(self, val):
        self.state.set_x(val)

    def J(self):
        return float(self.constraint.J())


def save_outputs(out_dir, tag, surf, curves, bs):
    bs.set_points(surf.gamma().reshape((-1, 3)))
    nphi = surf.quadpoints_phi.size
    ntheta = surf.quadpoints_theta.size
    B = bs.B().reshape((nphi, ntheta, 3))
    point_data = {"B_N": np.sum(B * surf.unitnormal(), axis=2)[:, :, None]}
    surf.to_vtk(str(out_dir / f"{tag}_surf"), extra_data=point_data)
    curves_to_vtk(curves, str(out_dir / f"{tag}_curves"))
    bs.save(str(out_dir / f"{tag}_bs.json"))


def main():
    args = make_parser().parse_args()
    args.max_modes = parse_modes(args.max_modes)
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mpi = MpiPartition()
    vmec = Vmec(
        str(Path(args.vmec_input).resolve()),
        mpi=mpi,
        verbose=False,
        nphi=args.nphi,
        ntheta=args.ntheta,
        range_surface="half period",
    )
    surf = vmec.boundary
    coils, curves, base_curves = build_initial_coils(surf, args)
    bs = BiotSavart(coils)

    proc0_print("Running simple 3-stage pipeline")
    proc0_print(f"output: {out_root}")

    for mode in args.max_modes:
        t0 = time.time()
        mode_dir = out_root / f"maxmode_{mode}"
        if mpi.proc0_world:
            mode_dir.mkdir(parents=True, exist_ok=True)

        proc0_print(f"=== max_mode={mode}: stage 1 ===")
        s1_constraints, qs = stage1_constraints(vmec, surf, mode, args)
        x1, _, _ = run_al(s1_constraints, args, args.stage1_iters, args.stage1_lag_iters)
        surf.set_dofs(x1)
        vmec.run()
        if mpi.proc0_world:
            surf.to_vtk(str(mode_dir / "stage1_surf"))
        proc0_print(f"stage1 aspect={vmec.aspect():.4f} iota={vmec.mean_iota():.4f} qs={qs.total():.3e}")

        proc0_print(f"=== max_mode={mode}: stage 2 ===")
        s2_constraints, Jf = stage2_constraints(surf, bs, curves, base_curves, args)
        x2, _, _ = run_al(s2_constraints, args, args.stage2_iters, args.stage2_lag_iters)
        Jf.x = x2
        if mpi.proc0_world:
            save_outputs(mode_dir, "stage2", surf, curves, bs)
        proc0_print(f"stage2 flux={Jf.J():.3e}")

        proc0_print(f"=== max_mode={mode}: single-stage ===")
        state = CombinedState(Jf, surf, bs)
        s3_constraints = [CombinedConstraint(c, state) for c in (s2_constraints + s1_constraints)]
        x0 = state.get_x()
        for c in s3_constraints:
            c.x = x0
        x3, _, _ = run_al(s3_constraints, args, args.single_iters, args.single_lag_iters)
        state.set_x(x3)
        vmec.run()
        if mpi.proc0_world:
            save_outputs(mode_dir, "single_stage", surf, curves, bs)
            with open(mode_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "max_mode": mode,
                        "aspect": float(vmec.aspect()),
                        "mean_iota": float(vmec.mean_iota()),
                        "qs_total": float(qs.total()),
                        "flux": float(Jf.J()),
                        "seconds": float(time.time() - t0),
                    },
                    f,
                    indent=2,
                )
        proc0_print(
            f"done mode={mode} aspect={vmec.aspect():.4f} iota={vmec.mean_iota():.4f} "
            f"qs={qs.total():.3e} flux={Jf.J():.3e} t={time.time()-t0:.1f}s"
        )

    proc0_print("All modes complete.")


if __name__ == "__main__":
    main()
