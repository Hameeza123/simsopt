import numpy as np
from simsopt.geo import AspectRatio

# Use .x (free DOFs only). get_dofs()/set_dofs() act on all DOFs and ignore fix().
def _free_get(s):
    return np.asarray(s.x)

def _free_set(s, v):
    s.x = np.asarray(v, dtype=float)

# Aspect-ratio constraint: residual = (A(x) - target) / scale
class AspectConstraint:
    def __init__(self, surf, target=7.0, scale=1.0):
        self.surf = surf
        self.ar = AspectRatio(surf)
        self.target = float(target)
        self.scale = float(scale)

    @property
    def x(self):
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def J(self):
        from simsopt._core.util import ObjectiveFailure
        try:
            return (float(self.ar.J()) - self.target) / self.scale
        except ObjectiveFailure:
            return 1e10


# QS constraint: one scalar per radius index k, residual = QS profile at radii[k] / scale.
# profile() sums the squared residuals over theta and phi for each radius;
# residuals() is the flattened (ns * ntheta * nphi) vector, so indexing it by k is wrong.
class QsConstraint:
    def __init__(self, vmec, surf, qs_obj, k, scale=1.0):
        self.vmec = vmec
        self.surf = surf
        self.qs = qs_obj
        self.k = int(k)
        self.scale = float(scale)

    @property
    def x(self):
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def _current_residual_k(self):
        return float(np.asarray(self.qs.profile())[self.k])

    def J(self):
        try:
            self.vmec.run()
        except Exception:
            return 1e10
        return self._current_residual_k() / self.scale
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
        return _free_get(self.surf)

    @x.setter
    def x(self, v):
        _free_set(self.surf, v)

    def _mean_iota(self):
        return float(self.vmec.mean_iota())


    def J(self):
        try:
            self.vmec.run()
            val = self._mean_iota()
        except Exception:
            return 1e10
        return val - self.target
