import os
import numpy as np

from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.geo import AspectRatio
from simsopt.solve.AL_nograd import augmented_lagrangian_method
from simsopt.util import MpiPartition, proc0_print



def _free_get(s):
    return np.asarray(s.get_dofs())

def _free_set(s, v):
    s.set_dofs(np.asarray(v))

def _vmec_run(v):
    v.run()  # suppress VMEC output

# -------------------------------
# Aspect-ratio constraint (A(x) - A_target = 0)
# Uses analytic gradient; sliced to FREE DOFs if necessary.
# Includes per-constraint scaling.
# -------------------------------
class AspectConstraint:
    def __init__(self, surf, target=7.0, scale=1.0):
        self.surf = surf
        self.ar = AspectRatio(surf)   # J() -> A(x), dJ() -> dA/dx(full)
        self.target = float(target)
        self.scale = float(scale)

        # Cache free-index map if available
        self._free_idx = getattr(surf, "_free_dof_idx", None)
        if self._free_idx is not None:
            self._free_idx = np.asarray(self._free_idx, dtype=int)

    @property
    def x(self):  # FREE DOFs
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def J(self):
        from simsopt._core.util import ObjectiveFailure
        try:
            # Residual (signed) / scale
            return (float(self.ar.J()) - self.target) # / self.scale
        except ObjectiveFailure:
            # Return a large penalty when VMEC fails
            return 1e10 # / self.scale

    # No dJ() method - finite difference solver will compute gradients automatically


# -------------------------------
# QS constraint: one scalar per chosen radius index k
# residual = QS_ratio_residuals[k]
# Uses central FD with VMEC runs; includes per-constraint scaling.
# -------------------------------
class QsConstraint:
    def __init__(self, vmec, surf, qs_obj, k, scale=1.0, eps=None):
        self.vmec = vmec
        self.surf = surf
        self.qs = qs_obj
        self.k = int(k)
        self.scale = float(scale)
        self.eps = eps  # if None, computed per component

    @property
    def x(self):
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def _current_residual_k(self):
        # Assumes VMEC already run for current x
        return float(np.asarray(self.qs.residuals())[self.k])

    def J(self):
        from simsopt._core.util import ObjectiveFailure
        _vmec_run(self.vmec)
        return self._current_residual_k()
        # try:
        #     _vmec_run(self.vmec)
        #     return self._current_residual_k() # / self.scale
        # except ObjectiveFailure:
        #     # Return a large penalty when VMEC fails
        #     return 1e10 # / self.scale

    # No dJ() method - finite difference solver will compute gradients automatically

# --- Iota constraint(s) for VMEC + Simsopt (no gradients) ---

# --- Mean iota (rotational transform) constraint: scalar, no gradients ---
import numpy as np

class MeanIotaConstraint:
    """
    Residual: mean_iota(x) - target
    - One scalar constraint suitable for your AL loop.
    - Gradient-free: only J() is used.
    """
    def __init__(self, vmec, target, scale=None, grid='half'):
        """
        vmec   : simsopt.mhd.Vmec instance
        target : desired mean iota (float), e.g. 0.42
        scale  : optional scalar to keep residual ~ O(1). If None, set on first call.
        grid   : 'half' uses wout.iotas (default), 'full' uses wout.iotaf if needed
        """
        self.vmec = vmec
        self.surf = vmec.boundary   # access FREE boundary DOFs
        self.target = float(target)
        self.scale = None if scale is None else float(scale)
        self.grid = grid

    # --- FREE-DOF interface expected by your AL code ---
    @property
    def x(self):
        return np.asarray(self.surf.get_dofs())

    @x.setter
    def x(self, v):
        self.surf.set_dofs(np.asarray(v, dtype=float))

    # --- helpers ---
    def _mean_iota(self):
        """
        Try vmec.mean_iota if available; otherwise average iota from wout.
        """
        # Prefer a provided scalar if VMEC exposes it:
        # mi = getattr(self.vmec, "mean_iota", None)
        self.vmec.run()
        return float(self.vmec.mean_iota())
        # try:
        #     self.vmec.run()
        #     return float(self.vmec.mean_iota())
        # except Exception:
        #     return 1e10


    # --- scalar residual for AL ---
    def J(self):
        """
        Returns (mean_iota - target)/scale as a float.
        On VMEC failure, returns a large penalty.
        """
        try:
            self.vmec.run()
            val = self._mean_iota()
        except Exception:
            return 1e10

        resid = val - self.target
        return resid 
        # Add scale later
        # if self.scale is None:
        #     self.scale = max(1.0, abs(resid))
        # return float(resid / self.scale)
