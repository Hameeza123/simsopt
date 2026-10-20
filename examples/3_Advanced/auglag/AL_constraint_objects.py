import numpy as np
from simsopt.geo import AspectRatio

def _free_get(s):
    return np.asarray(s.get_dofs())

def _free_set(s, v):
    s.set_dofs(np.asarray(v))

# Aspect-ratio constraint: residual = A(x) - target
class AspectConstraint:
    def __init__(self, surf, target=7.0):
        self.surf = surf
        self.ar = AspectRatio(surf)
        self.target = float(target)

        self._free_idx = getattr(surf, "_free_dof_idx", None)
        if self._free_idx is not None:
            self._free_idx = np.asarray(self._free_idx, dtype=int)

    @property
    def x(self):
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def J(self):
        from simsopt._core.util import ObjectiveFailure
        try:
            return float(self.ar.J()) - self.target
        except ObjectiveFailure:
            return 1e10


# QS constraint: one scalar per radius index k, residual = QS_ratio_residuals[k]
class QsConstraint:
    def __init__(self, vmec, surf, qs_obj, k):
        self.vmec = vmec
        self.surf = surf
        self.qs = qs_obj
        self.k = int(k)

    @property
    def x(self):
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def _current_residual_k(self):
        return float(np.asarray(self.qs.residuals())[self.k])

    def J(self):
        try:
            self.vmec.run()
        except Exception:
            return 1e10
        return self._current_residual_k()
        # possibly add a threshold here
    # No dJ() method - finite difference solver will compute gradients automatically

class MeanIotaConstraint:
    """Scalar constraint for mean iota without gradients."""
    def __init__(self, vmec, target, grid='half'):
        self.vmec = vmec
        self.surf = vmec.boundary
        self.target = float(target)
        self.grid = grid

    @property
    def x(self):
        return np.asarray(self.surf.get_dofs())

    @x.setter
    def x(self, v):
        self.surf.set_dofs(np.asarray(v, dtype=float))

    def _mean_iota(self):
        return float(self.vmec.mean_iota())


    def J(self):
        try:
            self.vmec.run()
            val = self._mean_iota()
        except Exception:
            return 1e10
        return val - self.target
