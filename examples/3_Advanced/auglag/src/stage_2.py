import os

import numpy as np
from scipy.optimize import minimize
from simsopt.geo import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    MeanSquaredCurvature,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

try:
    from simsopt.geo import CurveSurfaceDistance
except Exception:
    CurveSurfaceDistance = None


def _inactive_like(obj):
    return QuadraticPenalty(obj, 1e30, "max")


def form_stage_2_objective_function(surf, bs, base_curves, curves, inputs):
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(surf, bs, definition="local", threshold=inputs.FLUX_THRESHOLD)
    Jls = [CurveLength(c) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, inputs.CC_THRESHOLD, num_basecurves=inputs.ncoils)
    if CurveSurfaceDistance is not None:
        Jcsdist = CurveSurfaceDistance(curves, surf, inputs.CS_THRESHOLD)
    else:
        Jcsdist = _inactive_like(sum(Jls))
    Jcs = [LpCurveCurvature(c, 2, inputs.CURVATURE_THRESHOLD) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]

    J_LENGTH = QuadraticPenalty(sum(Jls), inputs.LENGTH_TARGET, "max")
    J_CC = Jccdist
    J_CS = Jcsdist
    J_CURVATURE = sum(Jcs)
    J_MSC = sum(QuadraticPenalty(J, inputs.MSC_THRESHOLD, "max") for J in Jmscs)
    J_ALS = sum(QuadraticPenalty(J, inputs.ALS_THRESHOLD, "max") for J in Jals)
    J_LENGTH_PENALTY = J_LENGTH

    JF_simple = (
        inputs.weight_flux * Jf
        + inputs.weight_cc * J_CC
        + inputs.weight_cs * J_CS
        + inputs.weight_length * J_LENGTH
        + inputs.weight_curvature * J_CURVATURE
    )
    JF = (
        inputs.weight_flux * Jf
        + inputs.weight_cc * J_CC
        + inputs.weight_cs * J_CS
        + inputs.weight_length * J_LENGTH
        + inputs.weight_curvature * J_CURVATURE
        + inputs.weight_msc * J_MSC
        + inputs.weight_als * J_ALS
    )
    return (
        JF_simple,
        JF,
        Jls,
        Jmscs,
        Jccdist,
        Jcsdist,
        Jf,
        J_LENGTH,
        J_CC,
        J_CS,
        J_CURVATURE,
        J_MSC,
        J_ALS,
        J_LENGTH_PENALTY,
    )


def inner_coil_loop(
    mpi,
    JF_simple,
    JF,
    Jls,
    Jmscs,
    Jccdist,
    Jcsdist,
    Jf,
    J_LENGTH,
    J_CC,
    J_CS,
    J_CURVATURE,
    J_MSC,
    J_ALS,
    J_LENGTH_PENALTY,
    vmec,
    curves,
    base_curves,
    surf,
    coils_results_path,
    number_vmec_dofs,
    bs,
    max_mode,
    inputs,
    figures_results_path,
    surf_full_boundary,
    write_coils=True,
):
    del Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS
    del J_LENGTH_PENALTY, base_curves, number_vmec_dofs, figures_results_path, surf_full_boundary

    x = np.copy(JF.x)
    if inputs.MAXITER_stage_2_simple > 0:
        res = minimize(
            fun=lambda y: (setattr(JF_simple, "x", y), JF_simple.J(), JF_simple.dJ())[1:3],
            x0=x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": inputs.MAXITER_stage_2_simple},
        )
        x = np.copy(res.x)

    if inputs.MAXITER_stage_2 > 0:
        res = minimize(
            fun=lambda y: (setattr(JF, "x", y), JF.J(), JF.dJ())[1:3],
            x0=x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": inputs.MAXITER_stage_2},
        )
        x = np.copy(res.x)

    JF.x = x
    bs.set_points(surf.gamma().reshape((-1, 3)))
    dofs = np.concatenate((x, vmec.x))

    if write_coils and mpi.proc0_world:
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_stage2_max_mode_{max_mode}"))
    return dofs, bs, JF

