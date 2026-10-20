import numpy as np
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem


def form_stage_1_objective_function(vmec, vmec_full_boundary, surf, surf_full_boundary, max_mode, inputs):
    surf = vmec.boundary
    surf_full_boundary = vmec_full_boundary.boundary

    surf.fix_all()
    vmec.indata.mpol = max_mode + 1
    vmec.indata.ntor = max_mode + 1
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.asarray(inputs.quasisymmetry_target_surfaces),
        helicity_m=inputs.qs_helicity_m,
        helicity_n=inputs.qs_helicity_n,
    )
    qi = None
    objective_tuple = [
        (vmec.aspect, inputs.aspect_ratio_target, 1.0),
        (vmec.mean_iota, inputs.iota_target, 1.0),
        (qs.residuals, 0.0, 1.0),
    ]
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    number_vmec_dofs = int(len(vmec.x))

    return vmec, vmec_full_boundary, surf, surf_full_boundary, qs, qi, number_vmec_dofs, prob

