import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.objectives import SquaredFlux
from threadpoolctl import threadpool_limits

__all__ = ['augmented_lagrangian_objective', 'augmented_lagrangian_method']


class dummyObjective:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
    def J(self):
        return 0.0


def _safe_J(obj):
    """Evaluate obj.J() with OpenMP limited; return large penalty on failure."""
    try:
        with threadpool_limits(limits=1, user_api='openmp'):
            return float(obj.J())
    except Exception:
        try:
            return float(obj.J())
        except Exception:
            return 1e10


def augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu):
    """Compute augmented Lagrangian: L_A = f(x) - λ^T g(x) + 0.5 * sum(μ_i * g_i(x)^2)"""
    dofs = np.asarray(dofs, dtype=float)
    f.x = dofs
    f_val = _safe_J(f)

    c_vals = []
    for c_i in equality_constraints:
        c_i.x = dofs
        c_vals.append(_safe_J(c_i))
    c_vals = np.asarray(c_vals, dtype=float)

    if np.ndim(mu) == 0:
        penalty = 0.5 * float(mu) * np.dot(c_vals, c_vals)
    else:
        penalty = 0.5 * np.dot(mu, c_vals**2)

    L = f_val - np.dot(lag_mul, c_vals) + penalty

    return L


def augmented_lagrangian_method(
        f=None,
        equality_constraints=[],
        mu_init=10.0,
        grad_tol=1e-15,
        c_tol=1e-15,
        tau=4,
        MAXITER=50,
        argmin_tol=1e-15,
        minimize_method='L-BFGS-B',
        MAXITER_lag=10,
        OUT_DIR='',
        verbose=False,
        penalty_type=None,
    ):
    """
    Augmented Lagrangian Method (ALM) for constrained optimization using finite differences.
    
    Solves min f(x) subject to g_i(x) = 0 by iteratively minimizing the augmented Lagrangian
    and updating Lagrange multipliers and penalty parameters. Uses gradient-free optimization
    where SciPy computes finite difference gradients automatically.
    
    Args:
        f: Objective function (with .J(), .x attributes). If None, pure feasibility problem.
        equality_constraints: List of constraint objects (each with .J(), .x methods).
        mu_init: Initial penalty parameter(s).
        c_tol: Constraint violation tolerance.
        tau: Penalty increase factor.
        MAXITER: Max iterations for inner minimization.
        MAXITER_lag: Max outer ALM iterations.
        minimize_method: Inner solver ('L-BFGS-B', 'Nelder-Mead', 'Powell', etc.).
        verbose: Print detailed progress.
        
    Returns:
        x: Optimized degrees of freedom
        fun: Final Lagrangian value
        lag_mul: Final Lagrange multipliers
    """
    from simsopt.geo import CurveLength

    if equality_constraints is None:
        equality_constraints = []
    m_eq = len(equality_constraints)

    if mu_init <= 0 or c_tol <= 0:
        raise ValueError("mu_init and c_tol must be strictly positive.")

    if f is not None and hasattr(f, 'x'):
        x = np.asarray(f.x, dtype=float)
    elif m_eq > 0 and hasattr(equality_constraints[0], 'x'):
        x = np.asarray(equality_constraints[0].x, dtype=float)
    else:
        raise ValueError("Provide f.x or equality_constraints[0].x for initialization.")

    if f is None:
        f = dummyObjective(x)

    if penalty_type is None:
        if np.isscalar(mu_init):
            mu_k = np.full(m_eq, float(mu_init), dtype=float)
        else:
            mu_k = np.asarray(mu_init, dtype=float)
            if mu_k.size != m_eq:
                raise ValueError("mu_init vector length must match number of constraints")
        if np.any(mu_k <= 1.0):
            raise ValueError("All components of mu_init must be > 1 for stability.")
    elif penalty_type == 'scalar':
        mu_k = float(mu_init)
    else:
        raise ValueError("Invalid penalty_type")

    lag_mul = np.zeros(m_eq, dtype=float)
    c_vals = np.array([_safe_J(J) for J in equality_constraints], dtype=float) if m_eq > 0 else np.array([], float)
    c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0.0

    mu_scalar = float(np.mean(mu_k)) if not np.isscalar(mu_k) else float(mu_k)
    omega_k = max(1e-3 / mu_scalar, c_tol)
    eta_k = max(1e-2 / np.sqrt(mu_scalar), c_tol)

    if verbose:
        print('='*80)
        print(f'METHOD: {minimize_method.upper()}')
        print(f'Number of constraints: {m_eq}')
        print(f'Number of DOFs: {len(x)}')
        try:
            print(f'Initial objective: {f.J():0.6e}')
        except Exception:
            print(f'Initial objective: (dummy) 0.0')
        print(f'Initial ||c||_inf = {c_norm:0.6e}')
        print('Initial constraint values:', c_vals)
        print('='*80)

    options = {'maxiter': MAXITER, 'disp': False}
    mth = minimize_method.lower()
    if mth == 'nelder-mead':
        options.update({'xatol': omega_k, 'fatol': omega_k, 'maxfev': max(2000, 10*len(x))})
    elif mth == 'powell':
        options.update({'xtol': omega_k, 'ftol': omega_k, 'maxfev': max(2000, 10*len(x))})
    elif minimize_method.upper() == 'L-BFGS-B':
        options.update({'gtol': omega_k, 'maxcor': 100, 'eps': 1e-6})

    k = 0
    last_res = None
    while (c_norm > c_tol) and (k < MAXITER_lag):
        dofs_before = x.copy()

        def fun(dofs):
            return augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu_k)

        res = minimize(fun, x, method=minimize_method, jac=None, options=options, tol=argmin_tol)
        last_res = res
        x = np.asarray(res.x, dtype=float)

        try:
            if isinstance(f, SquaredFlux):
                curves = [c.curve for c in f.field.coils]
                curves_to_vtk(curves, OUT_DIR + f"last_optimized_coils_auglag{k}")
        except Exception:
            pass

        c_vals = np.array([_safe_J(J) for J in equality_constraints], dtype=float) if m_eq > 0 else np.array([], float)
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0.0

        print(f"ALM iteration {k}: ||c||_inf = {c_norm:.6e} (target: {c_tol:.6e})")
        if verbose:
            try:
                f_val_now = float(f.J())
            except Exception:
                f_val_now = 0.0
            print(f"  Objective: {f_val_now:.6e}, nfev={getattr(res, 'nfev', 'NA')}, ||Δx||={np.linalg.norm(x - dofs_before):.3e}")
            print(f"  Constraints: {c_vals}")

        if c_norm < eta_k:
            if m_eq > 0:
                if np.isscalar(mu_k):
                    lag_mul -= mu_k * c_vals
                else:
                    lag_mul -= mu_k * c_vals
            mu_scalar = float(np.mean(mu_k)) if not np.isscalar(mu_k) else float(mu_k)
            omega_k = max(omega_k / max(mu_scalar, 2.0), c_tol)
            eta_k = max(eta_k / max(np.sqrt(mu_scalar), 2.0), c_tol)
        else:
            if penalty_type is None and m_eq > 0:
                if np.isscalar(mu_k):
                    mu_k = min(1e8, tau * mu_k)
                else:
                    for i in range(m_eq):
                        if abs(c_vals[i]) > c_tol:
                            mu_k[i] = min(1e8, tau * mu_k[i])
            else:
                mu_k = min(1e8, tau * float(mu_k))

            mu_scalar = float(np.mean(mu_k)) if not np.isscalar(mu_k) else float(mu_k)
            if mth == 'nelder-mead':
                options['xatol'] = options['fatol'] = max(1e-3 / mu_scalar, c_tol)
            elif mth == 'powell':
                options['xtol'] = options['ftol'] = max(1e-3 / mu_scalar, c_tol)
            elif minimize_method.upper() == 'L-BFGS-B':
                options['gtol'] = max(1e-3 / mu_scalar, c_tol)

            omega_k = max(1e-3 / mu_scalar, c_tol)
            eta_k = max(5e-3 / np.sqrt(mu_scalar), c_tol)

        if verbose:
            if np.isscalar(mu_k):
                mu_str = f"{mu_k:.2e}"
            else:
                mu_str = "[" + ", ".join(f"{m:.2e}" for m in np.atleast_1d(mu_k)[:3]) + "..." if len(mu_k) > 3 else "]"
            print(f"    Lagrange multipliers: {lag_mul}")
            print(f"    Penalty μ = {mu_str}, ω={omega_k:.2e}, η={eta_k:.2e}")

        k += 1

    if verbose:
        print('='*80)
        print(f'Final ||c||_inf = {c_norm:.6e}')
        print('Final constraint values:', c_vals)
        print('='*80)
    
    try:
        return x, (last_res.fun if last_res is not None else None), lag_mul
    except Exception:
        return x, None, lag_mul
