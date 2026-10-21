#!/usr/bin/env python
"""
Augmented Lagrangian optimization for stellarator boundary shape with constraints.

Optimizes boundary Fourier modes to achieve:
1) Target aspect ratio
2) Quasi-symmetry (QS) at specific radii
3) Target mean rotational transform (iota)

Uses finite difference gradients (no analytical derivatives required).
"""

import os
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.solve.AL_nograd import augmented_lagrangian_method
from simsopt.util import MpiPartition, proc0_print
from AL_constraint_objects import AspectConstraint, QsConstraint, MeanIotaConstraint
import time

proc0_print("Running auglag_s1_hamza.py")
proc0_print("=============================================")

output_dir = "examples/3_Advanced/auglag/outputs_s1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Settings
MAXITER = 5
MAXITER_lag = 2
mpi = MpiPartition(25)

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
surf.fix("rc(0,0)")  # Major radius

# proc0_print('Parameter space:', surf.dof_names)

# Build constraints
A_target = 6.8
c_aspect = AspectConstraint(surf, target=A_target)

radii = np.array([0, 0.1, 1.01])
qs_obj = QuasisymmetryRatioResidual(vmec, radii, helicity_m=1, helicity_n=-1)

vmec.run()
qs_constraints = [QsConstraint(vmec, surf, qs_obj, k) for k in range(len(radii))]
mean_iota_term = MeanIotaConstraint(vmec, target=-0.42)
equality_constraints = [c_aspect, *qs_constraints, mean_iota_term]


surf.to_vtk(filename=os.path.join(output_dir, f"initial_surf_maxmode{max_mode}1"))
# Print initial state
proc0_print(f"Initial AR: A(x0) = {vmec.aspect():.6f}, target = {A_target:.6f}")
proc0_print(f"Initial QS obj: {qs_obj.total():.6e}")
proc0_print(f"Qs residuals at radii {radii}: {qs_obj.residuals()}")
proc0_print(f"Initial Mean iota: {vmec.mean_iota():.6f}, target = {mean_iota_term.target:.6f}")
proc0_print("Using finite difference solver (no analytical gradients)")

proc0_print("Starting augmented Lagrangian optimization")
proc0_print("=============================================")

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(
    f=None,
    equality_constraints=equality_constraints,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag,
    mu_init=10,
    tau=10,
    c_tol=1e-5,
    verbose=False,
)
end_time = time.time()
proc0_print(f"Time taken: {end_time - start_time:.2f} seconds")
surf.x = x
vmec.run()
qs_final = np.asarray(qs_obj.residuals())

proc0_print("Optimization finished")
proc0_print("=============================================")
proc0_print(f"Final AR: A(x_final) = {vmec.aspect():.6f}, target = {A_target:.6f}")
proc0_print(f"Final QS obj: {qs_obj.total():.6e}")
proc0_print(f"Qs residuals at radii {radii}: {qs_obj.residuals()}")
proc0_print(f"Final Mean iota: {vmec.mean_iota():.6f}, target = {mean_iota_term.target:.6f}")



surf.to_vtk(filename=os.path.join(output_dir, f"final_surf_maxmode{max_mode}_LBFGS{MAXITER}_AL{MAXITER_lag}_"))
