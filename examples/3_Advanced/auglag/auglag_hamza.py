#!/usr/bin/env python3
import os, time
import numpy as np
from pathlib import Path

from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.derivative import Derivative
from simsopt._core.finite_difference import MPIFiniteDifference

from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import (
    SquaredFlux, QuadraticPenalty, LeastSquaresProblem
)
from simsopt.field import (
    BiotSavart, Current, coils_via_symmetries
)
from simsopt.geo import (
    CurveLength, CurveCurveDistance, MeanSquaredCurvature,
    LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves
)

# Optional (if your branch has these):
try:
    from simsopt.objectives import CurveSurfaceDistance, LinkingNumber
except Exception:
    CurveSurfaceDistance = None
    LinkingNumber = None

# Optional (force model may require RegularizedCoil on your branch)
try:
    from simsopt.field.coil import RegularizedCoil
    from simsopt.field.force import LpCurveForce
    HAVE_FORCE = True
except Exception:
    HAVE_FORCE = False


from simsopt.solve import augmented_lagrangian_method 


# ----------------------------
# MPI/paths
# ----------------------------
mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
directory = 'optimization_AL_single_stage'
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

# ----------------------------
# Inputs / knobs
# ----------------------------
vmec_input_filename = os.path.join(parent_path, '../',  'inputs', 'input.nfp4_QH_warm_start')
ncoils = 3
R0, R1 = 1.0, 0.6
nmodes_coils = 7
nphi_VMEC, ntheta_VMEC = 34, 34
max_mode = 1

# Engineering thresholds (tune to taste)
CC_THRESHOLD       = 0.08
CS_THRESHOLD       = 0.12
LENGTH_TARGET      = 3.3
CURVATURE_THRESHOLD= 7.0
MSC_THRESHOLD      = 10.0
FORCE_THRESHOLD    = 5.0e3     # example units N/m per turn (tune)
FLUX_THRESHOLD     = 1e-6      # for normalized/local definitions

# FD settings for stage-1 Jacobian
diff_method = "forward"
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0.0

# Stage-1 targets (plasma shaping)
aspect_ratio_target = 7.0
quasisymmetry_target_surfaces = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# ----------------------------
# VMEC / surface
# ----------------------------
proc0_print(f' Using vmec input file {vmec_input_filename}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=False,
            nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary

# ----------------------------
# Coils: initial guess
# ----------------------------
base_curves = create_equally_spaced_curves(
    ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1,
    order=nmodes_coils, numquadpoints=128
)
base_currents = [Current(1) * 1e5 for _ in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]

# Optional regularization for force (if needed by your branch)
# def regularize_coils_if_needed(base_coils_list, coils_list,
#                                turns=200, width=0.02, height=0.02):
#     if not HAVE_FORCE:
#         return base_coils_list, coils_list
#     # If LpCurveForce requires RegularizedCoil objects:
#     # Detect by quick try/except:
#     try:
#         _ = LpCurveForce(base_coils_list, coils_list, p=2.0, threshold=FORCE_THRESHOLD)
#         return base_coils_list, coils_list
#     except Exception:
#         reg_base, reg_full = [], []
#         for c in base_coils_list:
#             reg_base.append(RegularizedCoil(curve=c.curve, current=c.current,
#                                             turns=turns, width=width, height=height))
#         for c in coils_list:
#             reg_full.append(RegularizedCoil(curve=c.curve, current=c.current,
#                                             turns=turns, width=width, height=height))
#         return reg_base, reg_full

base_coils, coils = coils, coils #regularize_coils_if_needed(coils, coils)

# ----------------------------
# Field sampler on boundary
# ----------------------------
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

# ----------------------------
# Stage-2 (engineering) SIMSOPT objectives
# ----------------------------
Jf      = SquaredFlux(surf, bs, definition="normalized", threshold=FLUX_THRESHOLD)
Jls     = [CurveLength(c) for c in base_curves]
Jl_cap  = QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max")
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD) if CurveSurfaceDistance else None
Jcs     = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jcurv   = sum(Jcs)
Jlink   = LinkingNumber(curves, downsample=2) if LinkingNumber else None
Jforce  = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD) if HAVE_FORCE else None

# ----------------------------
# Stage-1 (aspect + QS) as least squares
# ----------------------------
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
objective_tuple = [
    (vmec.aspect, aspect_ratio_target, 1.0),
    (qs.residuals, 0.0, 1.0)
]
prob = LeastSquaresProblem.from_tuples(objective_tuple)

# Surface DOFs control
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))

# ----------------------------
# Wrappers so AL sees one concatenated design vector
# ----------------------------

def concat_get_x():
    # Use any coil objective that carries coil DoFs; CurveLength sum works
    x_coils = sum(Jls).x
    x_surf  = vmec.x
    return np.concatenate((x_coils, x_surf))

def concat_set_x(x):
    ncoils_dofs = len(sum(Jls).x)
    x_coils = x[:ncoils_dofs]
    x_surf  = x[ncoils_dofs:]
    sum(Jls).x = x_coils  # broadcasts to the underlying coil variables
    vmec.x = x_surf
    # Refresh field sampling for constraints that use the boundary:
    bs.set_points(surf.gamma().reshape((-1,3)))

class CoilOnlyConstraint:
    """Wrap a coil-only SIMSOPT Objective as a combined-[coils|surface] constraint."""
    def __init__(self, obj):
        self.obj = obj
        self._x = concat_get_x()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val
        concat_set_x(val)

    def J(self):
        return float(self.obj.J())

    def dJ(self):
        ncoils = len(sum(Jls).x)
        g_coils = self.obj.dJ()
        g_surf  = np.zeros(len(vmec.x))
        return np.concatenate((g_coils, g_surf))

class SquaredFluxCoupledConstraint:
    """SquaredFlux with BOTH coil gradient (from Jf) and surface mixed derivative."""
    def __init__(self, Jf_obj):
        self.Jf = Jf_obj
        self._x = concat_get_x()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val
        concat_set_x(val)

    def J(self):
        return float(self.Jf.J())

    def dJ(self):
        # coil part:
        g_coils = self.Jf.dJ()

        # --- mixed derivative wrt surface (same math you already used) ---
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
        B_n = Bcoil_n
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (
            np.sum(dB_by_dX*(n - B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3)
        )
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + \
                surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        g_surf = np.ravel(Derivative({surf: deriv})(surf))
        # ---------------------------------------------------------------

        return np.concatenate((g_coils, g_surf))

class Stage1Constraint:
    """
    Wrap the VMEC LeastSquaresProblem so AL can treat aspect+QS as a single equality constraint.
    J() >= 0 and equals 0 at target. Gradient only wrt surface; coils part is zeros.
    """
    def __init__(self, prob, vmec, prob_jacobian_builder):
        self.prob = prob
        self.vmec = vmec
        self._build_jac = prob_jacobian_builder
        self._x = concat_get_x()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val
        concat_set_x(val)

    def J(self):
        os.chdir(vmec_results_path)  # keep VMEC output here like your original
        return float(self.prob.objective())

    def dJ(self):
        # Surface gradient from stage-1 LSQ; coils part is zeros
        with self._build_jac() as prob_jac:
            g_surf = np.ravel(prob_jac.jac(self.vmec.x))
        g_coils = np.zeros(len(sum(Jls).x))
        return np.concatenate((g_coils, g_surf))

def prob_jacobian_builder():
    return MPIFiniteDifference(
        prob.objective, mpi, diff_method=diff_method,
        abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step
    )

# ----------------------------
# Build constraint list (everything goes here)
# ----------------------------
c_list = []

# Squared flux (coupled)
c_list.append(SquaredFluxCoupledConstraint(Jf))

# Engineering coil-only constraints
c_list.append(CoilOnlyConstraint(Jccdist))
if Jcsdist is not None:
    c_list.append(CoilOnlyConstraint(Jcsdist))
c_list.append(CoilOnlyConstraint(Jl_cap))
c_list.append(CoilOnlyConstraint(Jcurv))
if Jlink is not None:
    c_list.append(CoilOnlyConstraint(Jlink))
if Jforce is not None:
    c_list.append(CoilOnlyConstraint(Jforce))

# Stage-1 (aspect + QS) as equality (0 at target)
c_list.append(Stage1Constraint(prob, vmec, prob_jacobian_builder))

# ----------------------------
# Run AL with dummy objective
# ----------------------------
start_time = time.time()

# Make sure concatenated x is set on all wrappers before the call
x0 = concat_get_x()
for c in c_list:
    c.x = x0.copy()

x, fnc, lag_mul = augmented_lagrangian_method(
    equality_constraints=c_list,
    MAXITER=20,
    MAXITER_lag=10
)

# Apply the solution back to objects (already done via setters, but ensure final sync)
concat_set_x(x)

# ----------------------------
# Save / report
# ----------------------------
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)

if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_AL_final"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_AL_final"), extra_data=pointData)

proc0_print(f"AL single-stage done in {time.time()-start_time:.1f}s")
proc0_print(f"Aspect: {vmec.aspect():.3f}, QS total: {qs.total():.3e}")
proc0_print(f"⟨|B·n|⟩: {np.mean(np.abs(BdotN_surf)):.3e}")