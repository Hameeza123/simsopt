import argparse
import os
import time
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves, curves_to_vtk
from simsopt.util import proc0_print as pprint


def recalculate_inputs(parser, QAQHQICNTselected, QAorQHorQIorCNT, argv):
    start_idx = 2 if QAQHQICNTselected else 1

    parser.add_argument("--name", default=f"rogerio_simple_{QAorQHorQIorCNT.lower()}")
    parser.add_argument("--remove_previous_results", action="store_true")
    parser.add_argument("--vmec_input_start", default="input.nfp4_QH_warm_start")
    parser.add_argument("--use_half_period", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vmec_verbose", action="store_true")
    parser.add_argument("--nphi", type=int, default=24)
    parser.add_argument("--ntheta", type=int, default=24)

    parser.add_argument("--stage_1", action="store_true")
    parser.add_argument("--stage_2", action="store_true")
    parser.add_argument("--single_stage", action="store_true")
    parser.add_argument("--max_modes", nargs="+", type=int, default=[1])

    parser.add_argument("--MAXITER_stage_1", type=int, default=4)
    parser.add_argument("--MAXITER_stage_2_simple", type=int, default=4)
    parser.add_argument("--MAXITER_stage_2", type=int, default=8)
    parser.add_argument("--MAXITER_single_stage", type=int, default=12)
    parser.add_argument("--ftol", type=float, default=1e-9)

    parser.add_argument("--finite_difference_rel_step", type=float, default=0.0)
    parser.add_argument("--finite_difference_abs_step", type=float, default=1e-7)
    parser.add_argument("--diff_method", default="forward")
    parser.add_argument("--finite_beta", action="store_true")
    parser.add_argument("--JACOBIAN_THRESHOLD", type=float, default=1e12)
    parser.add_argument("--coils_objective_weight", type=float, default=1.0)

    parser.add_argument("--ncoils", type=int, default=4)
    parser.add_argument("--coil_order", type=int, default=6)
    parser.add_argument("--coil_numquadpoints", type=int, default=128)
    parser.add_argument("--current_scale", type=float, default=3e5)
    parser.add_argument("--R1_factor", type=float, default=0.6)

    parser.add_argument("--FLUX_THRESHOLD", type=float, default=1e-6)
    parser.add_argument("--CC_THRESHOLD", type=float, default=0.1)
    parser.add_argument("--CS_THRESHOLD", type=float, default=0.2)
    parser.add_argument("--LENGTH_TARGET", type=float, default=30.0)
    parser.add_argument("--CURVATURE_THRESHOLD", type=float, default=5.0)
    parser.add_argument("--MSC_THRESHOLD", type=float, default=5.0)
    parser.add_argument("--ALS_THRESHOLD", type=float, default=0.3)

    parser.add_argument("--weight_flux", type=float, default=1.0)
    parser.add_argument("--weight_cc", type=float, default=1.0)
    parser.add_argument("--weight_cs", type=float, default=1.0)
    parser.add_argument("--weight_length", type=float, default=1.0)
    parser.add_argument("--weight_curvature", type=float, default=1.0)
    parser.add_argument("--weight_msc", type=float, default=1.0)
    parser.add_argument("--weight_als", type=float, default=1.0)

    parser.add_argument("--aspect_ratio_target", type=float, default=6.8)
    parser.add_argument("--iota_target", type=float, default=-0.42)
    parser.add_argument("--qs_helicity_m", type=int, default=1)
    parser.add_argument("--qs_helicity_n", type=int, default=-1)
    parser.add_argument("--quasisymmetry_target_surfaces", nargs="+", type=float, default=[0.0, 0.1, 1.01])

    parser.add_argument("--output_interval", type=int, default=0)
    parser.add_argument("--debug_coils_outputtxt", action="store_true")
    parser.add_argument("--debug_output_file", default="debug_output.txt")

    parser.add_argument("--resulting_surface", default="surf_final_")
    parser.add_argument("--resulting_coils", default="coils_final_")
    parser.add_argument("--resulting_field_json", default="biotsavart_final_")

    parser.add_argument("--create_wout_final", action="store_true")
    parser.add_argument("--vmec_plot_result", action="store_true")
    parser.add_argument("--booz_xform_plot_result", action="store_true")
    parser.add_argument("--boozxform_nsurfaces", type=int, default=12)

    inputs = parser.parse_args(argv[start_idx:])

    if not isinstance(inputs.max_modes, list):
        inputs.max_modes = [inputs.max_modes]
    return inputs


def create_results_folders(inputs):
    base = os.getcwd()
    coils_results_path = os.path.join(base, "coils")
    vmec_results_path = os.path.join(base, "vmec")
    figures_results_path = os.path.join(base, "figures")
    for p in [coils_results_path, vmec_results_path, figures_results_path]:
        Path(p).mkdir(parents=True, exist_ok=True)

    if inputs.debug_output_file:
        debug_path = Path(inputs.debug_output_file)
        if not debug_path.is_absolute():
            debug_path = Path(base) / debug_path
        inputs.debug_output_file = str(debug_path)
        Path(inputs.debug_output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(inputs.debug_output_file, "w", encoding="utf-8") as f:
            f.write(f"run_start={time.time()}\n")
    return coils_results_path, vmec_results_path, figures_results_path


def create_initial_coils(vmec, parent_path, coils_results_path, inputs, surf_full_boundary, mpi):
    del parent_path, surf_full_boundary
    surf = vmec.boundary
    R0 = float(surf.get_rc(0, 0))
    R1 = inputs.R1_factor * R0
    base_curves = create_equally_spaced_curves(
        inputs.ncoils,
        surf.nfp,
        stellsym=surf.stellsym,
        R0=R0,
        R1=R1,
        order=inputs.coil_order,
        numquadpoints=inputs.coil_numquadpoints,
    )
    base_currents = [Current(inputs.current_scale / inputs.ncoils * 1e-5) * 1e5 for _ in range(inputs.ncoils)]
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    if mpi.proc0_world:
        curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pprint(f"Initialized {len(coils)} coils")
    return bs, coils, curves, base_curves

