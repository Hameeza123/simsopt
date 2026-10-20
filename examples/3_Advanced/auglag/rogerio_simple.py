#!/usr/bin/env python3
"""
Single stage stellarator optimization (3SO) code.
Simplified to the core pipeline for local examples/3_Advanced/auglag usage.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    import pandas as pd
except Exception:
    pd = None

from simsopt._core.derivative import Derivative
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.geo import curves_to_vtk
from simsopt.mhd import Vmec
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition
from simsopt.util import proc0_print as pprint

from src.initialization_functions import recalculate_inputs, create_initial_coils, create_results_folders
from src.stage_1 import form_stage_1_objective_function
from src.stage_2 import form_stage_2_objective_function, inner_coil_loop


def pick_qa(argv):
    if len(argv) > 1 and argv[1] in {"QA", "QH", "QI", "CNT"}:
        return argv[1], True
    pprint("First line argument (QA, QH, QI or CNT) not selected. Defaulting to QA.")
    return "QA", False


def pick_vmec_input(current_path, parent_path, vmec_results_path, inputs):
    vmec_files_list = os.listdir(vmec_results_path) if os.path.isdir(vmec_results_path) else []
    if os.path.exists(os.path.join(current_path, "input.final")):
        return os.path.join(current_path, "input.final")
    if len(vmec_files_list) == 0:
        return os.path.join(parent_path, "..", "inputs", inputs.vmec_input_start)
    vmec_input_files = [f for f in vmec_files_list if f.startswith("input.")]
    vmec_input_files.sort(key=lambda item: (len(item), item))
    return os.path.join(vmec_results_path, vmec_input_files[-1])


def append_debug(inputs, txt):
    if not inputs.debug_output_file:
        return
    with open(inputs.debug_output_file, "a", encoding="utf-8") as f:
        f.write(txt)


class single_stage_obj_and_der:
    def __init__(
        self,
        inputs,
        vmec_results_path,
        current_path,
        bs,
        surf,
        surf_full_boundary,
        curves,
        base_curves,
        prob,
        number_vmec_dofs,
        JF,
        Jf,
        Jls,
        Jmscs,
        Jccdist,
        J_LENGTH,
        J_CC,
        J_CURVATURE,
        J_MSC,
        J_ALS,
        J_LENGTH_PENALTY,
        qs,
        qa_type,
        coils_results_path,
    ):
        self.inputs = inputs
        self.vmec_results_path = vmec_results_path
        self.current_path = current_path
        self.bs = bs
        self.surf = surf
        self.surf_full_boundary = surf_full_boundary
        self.curves = curves
        self.base_curves = base_curves
        self.prob = prob
        self.number_vmec_dofs = number_vmec_dofs
        self.JF = JF
        self.Jf = Jf
        self.Jls = Jls
        self.Jmscs = Jmscs
        self.Jccdist = Jccdist
        self.J_LENGTH = J_LENGTH
        self.J_CC = J_CC
        self.J_CURVATURE = J_CURVATURE
        self.J_MSC = J_MSC
        self.J_ALS = J_ALS
        self.J_LENGTH_PENALTY = J_LENGTH_PENALTY
        self.qs = qs
        self.qa_type = qa_type
        self.coils_results_path = coils_results_path

    def fun(self, dofs, prob_jacobian=None, info={"Nfeval": 0}, max_mode=1, oustr_dict=[]):
        info["Nfeval"] += 1
        self.JF.x = dofs[:-self.number_vmec_dofs]
        self.prob.x = dofs[-self.number_vmec_dofs:]
        self.bs.set_points(self.surf.gamma().reshape((-1, 3)))

        os.chdir(self.vmec_results_path)
        J_stage_1 = self.prob.objective()
        J_stage_2 = self.inputs.coils_objective_weight * self.JF.J()
        J = J_stage_1 + J_stage_2
        if J > self.inputs.JACOBIAN_THRESHOLD or np.isnan(J):
            J = self.inputs.JACOBIAN_THRESHOLD

        jf = self.Jf.J()
        BdotN = np.mean(np.abs(np.sum(self.bs.B().reshape((self.inputs.nphi, self.inputs.ntheta, 3)) * self.surf.unitnormal(), axis=2)))
        row = {
            "Nfeval": info["Nfeval"],
            "J": float(J),
            "Jf": float(jf),
            "J_length": float(self.J_LENGTH.J()),
            "J_CC": float(self.J_CC.J()),
            "J_CURVATURE": float(self.J_CURVATURE.J()),
            "J_MSC": float(self.J_MSC.J()),
            "J_ALS": float(self.J_ALS.J()),
            "J_LENGTH_PENALTY": float(self.J_LENGTH_PENALTY.J()),
            "Lengths": float(sum(j.J() for j in self.Jls)),
            "curvatures": float(np.sum([np.max(c.kappa()) for c in self.base_curves])),
            "msc": float(np.sum([j.J() for j in self.Jmscs])),
            "B.n": float(BdotN),
            "C-C-Sep": float(self.Jccdist.shortest_distance()),
        }
        try:
            row["Jquasisymmetry"] = float(self.qs.total()) if self.qa_type != "QI" else 0.0
            row["Jiota"] = float((self.prob.parent_return_fns_no_args[1]() - self.inputs.iota_target) ** 2)
            row["Jaspect"] = float((self.prob.parent_return_fns_no_args[0]() - self.inputs.aspect_ratio_target) ** 2)
        except Exception:
            row["Jquasisymmetry"] = 0.0
            row["Jiota"] = 0.0
            row["Jaspect"] = 0.0

        if J < self.inputs.JACOBIAN_THRESHOLD:
            prob_dJ = prob_jacobian.jac(self.prob.x)
            if self.inputs.finite_beta:
                raise RuntimeError("finite_beta not implemented")
            coils_dJ = self.JF.dJ()

            n = self.surf.normal()
            absn = np.linalg.norm(n, axis=2)
            B = self.bs.B().reshape((self.inputs.nphi, self.inputs.ntheta, 3))
            dB_by_dX = self.bs.dB_by_dX().reshape((self.inputs.nphi, self.inputs.ntheta, 3, 3))
            Bcoil = self.bs.B().reshape(n.shape)
            unitn = n * (1.0 / absn)[:, :, None]
            Bcoil_n = np.sum(Bcoil * unitn, axis=2)
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            B_N = np.sum(Bcoil * n, axis=2)
            assert self.Jf.definition == "local"
            dJdx = (Bcoil_n / mod_Bcoil**2)[:, :, None] * (
                np.sum(dB_by_dX * (n - B * (B_N / mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3)
            )
            dJdN = (Bcoil_n / mod_Bcoil**2)[:, :, None] * Bcoil - 0.5 * (B_N**2 / absn**3 / mod_Bcoil**2)[:, :, None] * n
            deriv = self.surf.dnormal_by_dcoeff_vjp(dJdN / (self.inputs.nphi * self.inputs.ntheta)) + self.surf.dgamma_by_dcoeff_vjp(
                dJdx / (self.inputs.nphi * self.inputs.ntheta)
            )
            mixed_dJ = Derivative({self.surf: deriv})(self.surf)
            grad = np.concatenate(
                (
                    self.inputs.coils_objective_weight * coils_dJ,
                    np.ravel(prob_dJ) + self.inputs.coils_objective_weight * mixed_dJ,
                )
            )
        else:
            grad = np.zeros_like(dofs)

        os.chdir(self.current_path)
        append_debug(self.inputs, f"\nfun#{info['Nfeval']} J={J:.3e} Jf={jf:.3e} <B.n>={BdotN:.3e}")
        oustr_dict.append(row)

        if self.inputs.output_interval > 0 and np.mod(info["Nfeval"], self.inputs.output_interval) == 0:
            pointData = {
                "B_N": np.sum(self.bs.B().reshape((self.inputs.nphi, self.inputs.ntheta, 3)) * self.surf.unitnormal(), axis=2)[:, :, None]
            }
            self.surf_full_boundary.x = self.surf.x
            self.surf_full_boundary.to_vtk(
                os.path.join(self.coils_results_path, f"surf_intermediate_max_mode_{max_mode}_{info['Nfeval']}"),
                extra_data=pointData,
            )
            curves_to_vtk(self.curves, os.path.join(self.coils_results_path, f"curves_intermediate_max_mode_{max_mode}_{info['Nfeval']}"))
        return J, grad


def main(argv):
    start = time.time()
    mpi = MpiPartition()
    qa_type, qa_selected = pick_qa(argv)

    parser = argparse.ArgumentParser()
    inputs = recalculate_inputs(parser, qa_selected, qa_type, argv)
    if (not inputs.stage_1) and (not inputs.stage_2) and (not inputs.single_stage):
        inputs.single_stage = True

    parent_path = str(Path(__file__).parent.resolve())
    current_path = os.path.join(parent_path, "results", f"{inputs.name}")
    if mpi.proc0_world and inputs.remove_previous_results and os.path.isdir(current_path):
        shutil.copytree(current_path, current_path + "_backup", dirs_exist_ok=True)
        shutil.rmtree(current_path)
    if mpi.proc0_world:
        Path(current_path).mkdir(parents=True, exist_ok=True)

    current_path = str(Path(current_path).resolve())
    src_inputs = os.path.join(parent_path, "src", "inputs.py")
    if mpi.proc0_world and os.path.isfile(src_inputs):
        shutil.copy(src_inputs, os.path.join(current_path, f"{inputs.name}.py"))

    os.chdir(current_path)
    coils_results_path, vmec_results_path, figures_results_path = create_results_folders(inputs)
    del figures_results_path
    with open(os.path.join(current_path, f"inputs_{inputs.name}.json"), "w", encoding="utf-8") as f:
        json.dump({k: getattr(inputs, k) for k in dir(inputs) if "__" not in k}, f, ensure_ascii=False, indent=2)

    pprint("============================================")
    pprint("Starting single stage optimization")
    pprint("============================================")

    vmec_input_filename = pick_vmec_input(current_path, parent_path, vmec_results_path, inputs)
    pprint(f"Using vmec input file {vmec_input_filename}")
    if inputs.use_half_period:
        vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta, range_surface="half period")
        vmec_full_boundary = Vmec(vmec_input_filename, mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta)
    else:
        vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta)
        vmec_full_boundary = vmec

    surf = vmec.boundary
    surf_full_boundary = vmec_full_boundary.boundary
    bs, coils, curves, base_curves = create_initial_coils(vmec, parent_path, coils_results_path, inputs, surf_full_boundary, mpi)
    (
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
    ) = form_stage_2_objective_function(surf, bs, base_curves, curves, inputs)
    del Jcsdist, J_CS

    oustr_dict_outer = []
    for max_mode in inputs.max_modes:
        pprint(f"Optimization with max_mode={max_mode}")
        oustr_dict_inner = []

        vmec, vmec_full_boundary, surf, surf_full_boundary, qs, qi, number_vmec_dofs, prob = form_stage_1_objective_function(
            vmec, vmec_full_boundary, surf, surf_full_boundary, max_mode, inputs
        )
        del qi
        dofs = np.concatenate((JF.x, vmec.x))

        if inputs.stage_1:
            os.chdir(vmec_results_path)
            pprint(f"Performing Stage 1 optimization with {inputs.MAXITER_stage_1} iterations")
            least_squares_mpi_solve(
                prob,
                mpi,
                grad=True,
                rel_step=inputs.finite_difference_rel_step,
                abs_step=inputs.finite_difference_abs_step,
                max_nfev=inputs.MAXITER_stage_1,
                ftol=inputs.ftol,
            )
            os.chdir(current_path)
            vmec.write_input(os.path.join(current_path, f"input.stage1_maxmode{max_mode}"))

        if inputs.stage_2:
            pprint(f"Performing Stage 2 optimization with {inputs.MAXITER_stage_2+inputs.MAXITER_stage_2_simple} iterations")
            surf = vmec.boundary
            bs.set_points(surf.gamma().reshape((-1, 3)))
            if mpi.proc0_world:
                dofs, bs, JF = inner_coil_loop(
                    mpi,
                    JF_simple,
                    JF,
                    Jls,
                    Jmscs,
                    Jccdist,
                    None,
                    Jf,
                    J_LENGTH,
                    J_CC,
                    None,
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
                    "",
                    surf_full_boundary,
                )
            mpi.comm_world.Bcast(dofs, root=0)

        if inputs.single_stage:
            pprint(f"Performing Single Stage optimization with {inputs.MAXITER_single_stage} iterations")
            dofs = np.concatenate((JF.x, vmec.x))
            obj_and_der = single_stage_obj_and_der(
                inputs=inputs,
                vmec_results_path=vmec_results_path,
                current_path=current_path,
                bs=bs,
                surf=surf,
                surf_full_boundary=surf_full_boundary,
                curves=curves,
                base_curves=base_curves,
                prob=prob,
                number_vmec_dofs=number_vmec_dofs,
                JF=JF,
                Jf=Jf,
                Jls=Jls,
                Jmscs=Jmscs,
                Jccdist=Jccdist,
                J_LENGTH=J_LENGTH,
                J_CC=J_CC,
                J_CURVATURE=J_CURVATURE,
                J_MSC=J_MSC,
                J_ALS=J_ALS,
                J_LENGTH_PENALTY=J_LENGTH_PENALTY,
                qs=qs,
                qa_type=qa_type,
                coils_results_path=coils_results_path,
            )
            with MPIFiniteDifference(
                prob.objective,
                mpi,
                rel_step=inputs.finite_difference_rel_step,
                abs_step=inputs.finite_difference_abs_step,
                diff_method=inputs.diff_method,
            ) as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(
                        obj_and_der.fun,
                        dofs,
                        args=(prob_jacobian, {"Nfeval": 0}, max_mode, oustr_dict_inner),
                        jac=True,
                        method="BFGS",
                        options={"maxiter": inputs.MAXITER_single_stage},
                        tol=1e-15,
                    )
                    dofs = np.copy(res.x)
            mpi.comm_world.Bcast(dofs, root=0)

        if mpi.proc0_world:
            pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf_full_boundary.x = surf.x
            surf_full_boundary.to_vtk(os.path.join(coils_results_path, inputs.resulting_surface + "max_mode_" + str(max_mode)), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path, inputs.resulting_coils + "max_mode_" + str(max_mode)))
            bs.save(os.path.join(coils_results_path, inputs.resulting_field_json + "max_mode_" + str(max_mode) + ".json"))
            vmec.write_input(os.path.join(vmec_results_path, f"input.{inputs.name}_maxmode{max_mode}"))

        os.chdir(vmec_results_path)
        try:
            pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
            pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
            if qa_type != "QI":
                pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
            pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
        except Exception as e:
            pprint(e)
        os.chdir(current_path)

        if inputs.single_stage and mpi.proc0_world and (pd is not None):
            try:
                pd.DataFrame(oustr_dict_inner).to_csv(os.path.join(current_path, f"output_max_mode_{max_mode}.csv"), index_label="index")
            except Exception as e:
                pprint(e)

        oustr_dict_outer.append(oustr_dict_inner)

    if mpi.proc0_world:
        pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf_full_boundary.x = surf.x
        surf_full_boundary.to_vtk(os.path.join(coils_results_path, inputs.resulting_surface), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path, inputs.resulting_coils))
        bs.save(os.path.join(coils_results_path, inputs.resulting_field_json))
        vmec.write_input(os.path.join(current_path, "input.final"))
        if inputs.single_stage and (pd is not None):
            try:
                pd.DataFrame(oustr_dict_outer).to_csv(os.path.join(current_path, "output_final.csv"), index_label="index")
            except Exception as e:
                pprint(e)

    os.chdir(parent_path)
    if mpi.proc0_world:
        for pattern in ["jac_*"]:
            for f in glob.glob(os.path.join(current_path, pattern)):
                os.remove(f)
        for pattern in ["jac_*", "objective_*", "residuals_*", "*000_*", "parvmec*", "threed*"]:
            for f in glob.glob(os.path.join(vmec_results_path, pattern)):
                os.remove(f)

    stop = time.time()
    pprint("============================================")
    pprint("End of single stage optimization")
    pprint(f"Took {stop-start} seconds")
    pprint("============================================")


if __name__ == "__main__":
    main(sys.argv)

