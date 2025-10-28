#!/usr/bin/env python
"""
Augmented Lagrangian optimization for stellarator boundary shape with constraints.

Optimizes boundary Fourier modes to achieve:
1. Target aspect ratio
2. Quasi-symmetry (QS) at specific radii
3. Target mean rotational transform (iota)

Uses finite difference gradients (no analytical derivatives required).
"""

import os
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.geo import AspectRatio
from simsopt.solve.AL_nograd import augmented_lagrangian_method
from simsopt.util import MpiPartition, proc0_print
from AL_constraint_objects import AspectConstraint, QsConstraint, MeanIotaConstraint

proc0_print("Running auglag_s1_hamza.py")
proc0_print("=============================================")

# Settings
nphi = 4
ntheta = 4
MAXITER = 10
MAXITER_lag = 3
mpi = MpiPartition(4)

# VMEC setup
filename = os.path.join(os.path.dirname(__file__), '../inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)

# Boundary parameter space
surf = vmec.boundary
surf.fix_all()
max_mode = 1
vmec.indata.mpol = max_mode + 1
vmec.indata.ntor = vmec.indata.mpol
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)

def _vmec_run(v):
    v.run()

# Build constraints
A_target = 6.8
ar_obj = AspectRatio(surf)
A0 = float(ar_obj.J())
scale_ar = max(1.0, abs(A0 - A_target))
c_aspect = AspectConstraint(surf, target=A_target, scale=scale_ar)

radii = np.array([0.2, 0.8])
keep = [0, 1]
qs_obj = QuasisymmetryRatioResidual(vmec, radii, helicity_m=1, helicity_n=0)

_vmec_run(vmec)
qs_vec0 = np.asarray(qs_obj.residuals())
qs_scales = {k: max(1.0, abs(qs_vec0[k])) for k in keep}
qs_constraints = [QsConstraint(vmec, surf, qs_obj, k, scale=qs_scales[k]) for k in keep]

mean_iota_term = MeanIotaConstraint(vmec, target=-0.42)
equality_constraints = [c_aspect, *qs_constraints, mean_iota_term]


surf.to_vtk(filename = "initial_surf")
# Print initial state
proc0_print(f"Initial AR: A(x0) = {A0:.6f}, target = {A_target:.6f}, scale = {scale_ar:.6f}")
for k in keep:
    proc0_print(f"Initial QS residual[k={k}] = {qs_vec0[k]:.6e}")
proc0_print(f"Initial Mean iota: {mean_iota_term.J() + mean_iota_term.target:.6f}, target = {mean_iota_term.target:.6f}")
proc0_print(f"Number of constraints: {len(equality_constraints)}")
proc0_print(f"FREE DOFs = {c_aspect.x.shape[0]}")
proc0_print("Using finite difference solver (no analytical gradients)")

proc0_print("Starting augmented Lagrangian optimization")
proc0_print("=============================================")

# Run optimization
x, fnc, lag_mul = augmented_lagrangian_method(
    f=None,
    equality_constraints=equality_constraints,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag,
    mu_init=3,
    tau=3,
    c_tol=1e-5,
    verbose=False,
)
surf.set_dofs(x)
_vmec_run(vmec)
A_final = float(AspectRatio(surf).J())
qs_final = np.asarray(qs_obj.residuals())

proc0_print("Optimization finished")
proc0_print("=============================================")
proc0_print(f"Final AR: A(x_final) = {A_final:.6f}, target = {A_target:.6f}")

for k in keep:
    proc0_print(f"Final QS residual[k={k}] = {qs_final[k]:.6e}")

print(f"Mean iota: {mean_iota_term.J() + mean_iota_term.target:.6f}, target = {mean_iota_term.target:.6f}")


surf.to_vtk(filename = "final_surf")
