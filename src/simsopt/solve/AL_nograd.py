import numpy as np
import time
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.objectives import SquaredFlux
from threadpoolctl import threadpool_limits

__all__ = ['augmented_lagrangian_objective', 'augmented_lagrangian_method']

class dummyObjective:
    def __init__(self, x):
        self.x = x
    def J(self):
        return 0.0
    def dJ(self):
        return np.zeros_like(self.x)


def augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu):
    """
    Compute the value of the augmented Lagrangian for the current optimization variables.

    The standard augmented Lagrangian is:

        .. math::

            L_A(x, \lambda, \mu) = f(x) - \lambda^T g(x) + \frac{\mu}{2} \|g(x)\|^2

    where:
        - :math:`f(x)`: objective function (e.g., squared flux)
        - :math:`g(x)`: vector of constraint functions
        - :math:`\lambda`: vector of Lagrange multipliers
        - :math:`\mu`: penalty parameter

    If ``option == 'least-squares'``, uses the least-squares form:

        .. math::

            L_{LS}(x, \lambda, \mu) = \frac{1}{2} \|f(x)\|^2 + \frac{1}{2} \left\| -\frac{\lambda}{\sqrt{\mu}} + \sqrt{\mu} g(x) \right\|^2

    Args:
        f: Objective function object (with .J() and .x attributes).
        dofs (np.ndarray): Current degrees of freedom.
        equality_constraints (list): List of constraint objects (with .J() method).
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        float: Value of the augmented Lagrangian.
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        f_val = f.J()
    else:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                f_val = f.J()
        except Exception:
            f_val = f.J()

    # collect constraint values
    c_vals_list = []
    for c_i in equality_constraints:
        dof_dif = len(dofs) - len(c_i.x)
        c_i.x = dofs[dof_dif:]
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                val_ci = float(c_i.J())
        except Exception:
            val_ci = float(c_i.J())
        c_vals_list.append(val_ci)
    c_vals = np.array(c_vals_list, dtype=np.float64)

    # build Lagrangian penalty term supporting scalar or vector mu
    if np.ndim(mu) == 0:
        penalty = 0.5 * float(mu) * np.dot(c_vals, c_vals)
    else:
        penalty = 0.5 * np.dot(mu, c_vals**2)
    L = f_val - np.dot(lag_mul, c_vals) + penalty

    return L


def augmented_lagrangian_method(
    f=None,
    equality_constraints=None,
        mu_init=10.0,
        c_tol=1e-15,
        tau=4,
        MAXITER=50,
        argmin_tol=1e-15,
        minimize_method='L-BFGS-B',
        MAXITER_lag=10,
        OUT_DIR='',
        verbose=False,
        penalty_type=None,
        log_callback=None,
        ):
    """
    Run the Augmented Lagrangian Method (ALM) for constrained optimization.

    This method solves:

        .. math::

            \min_x f(x) \quad \text{subject to} \quad g_i(x) = 0

    by iteratively minimizing the augmented Lagrangian and updating multipliers and penalty parameters.

    The main loop alternates between minimizing the augmented Lagrangian and updating the multipliers/penalty:
        - Minimize :math:`L_A(x, \lambda, \mu)` with respect to :math:`x`
        - Update :math:`\lambda` and :math:`\mu` based on constraint satisfaction

    Args:
        f (Optimizable, optional): Main objective function (with .J(), .x attributes).
            If not provided, only constraints are used in the optimization.
        equality_constraints (list of Optimizable): List of constraint objects corresponding to equality constraints.
            These are equivalent to inequality constraints if the constraint is set up so that if the value is below
            some threshold, the value is set to zero.
        mu_init (float, optional, default=10.0): Initial penalty parameter (must be > 0).
        c_tol (float, optional, default=1e-15): Tolerance for constraint norm.
        MAXITER (int, optional, default=50): Max iterations for inner minimization.
        argmin_tol (float, optional, default=1e-15): Tolerance for inner minimization.
        minimize_method (str, optional, default='L-BFGS-B'): Optimization method for inner loop.
        MAXITER_lag (int, optional, default=10): Max outer ALM iterations.
        lagrangian_form (str, optional, default=None): If 'least-squares', use least-squares form.
        OUT_DIR (str, optional, default=''): Directory for output files.
        verbose (bool, optional, default=False): Whether to print verbose output.

    Returns:
        tuple: (x, final_Lagrangian_value, lagrange_multipliers)
            - x (np.ndarray): Optimized degrees of freedom
            - final_Lagrangian_value (float): Final value of the augmented Lagrangian
            - lagrange_multipliers (np.ndarray): Final Lagrange multipliers
    """
    from simsopt.geo import CurveLength
    np.random.seed(1)
    start_time = time.time()
    if equality_constraints is None:
        equality_constraints = []
    m_eq = len(equality_constraints)

    if mu_init <= 0 or c_tol <= 0:
        raise ValueError(
            "eta_init, mu_init and omega_init  must be strictly positive")
    if not penalty_type:
        if np.isscalar(mu_init):
            mu_k = np.ones(m_eq, dtype=float) * mu_init
        else:
            mu_k = np.array(mu_init, dtype=float)
            if mu_k.size != m_eq:
                raise ValueError("mu_init vector length must match number of constraints")
        if np.any(mu_k <= 1):
            raise ValueError("All components of mu_init must be > 1")
    elif penalty_type == 'scalar':
        mu_k = mu_init
    else:
        raise ValueError("Invalid penalty type")
    
    k = 1

    # Picks the most dofs from the objective function or the first equality constraint
    try:
        x = f.x
    except Exception:
        if m_eq == 0:
            raise ValueError("At least one of f or equality_constraints must provide an initial x")
        x = equality_constraints[0].x
    
    if verbose:
        print('----------------------------------------------------------------')
        print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
        try:
            print(f'INITIAL SQUARED FLUX: {f.J():0.6f}')
        except:
            print(f'INITIAL SQUARED FLUX: {equality_constraints[0].J():0.6f}')
        print('----------------------------------------------------------------')

    if f is None:
        f = dummyObjective(x)

    # Initialize multipliers randomly -- DO NOT INITIALIZE TO ZEROS since then the
    # first iteration will not be able to improve the constraints and DO NOT
    # INITIALIZE TO -mu_init * c_vals since then some of the lagrange multipliers
    # will be set to exactly zero, turning off the corresponding constraint for 
    # all remaining iterations. However, need to make sure lag_mul matches c_vals sign
    c_vals = np.array([J.J() for J in equality_constraints], dtype=float) if m_eq > 0 else np.array([], dtype=float)
    lag_mul = -np.random.rand(m_eq) * np.sign(c_vals)

    # Evaluate initial lagrangian
    c_norm = np.linalg.norm(c_vals) if m_eq > 0 else 0.0

    mu_k_scalar = np.mean(mu_k) if not np.isscalar(mu_k) else float(mu_k)
    omega_k = 1.0 / mu_k_scalar
    eta_k = 1.0 / (mu_k_scalar ** 0.1)
    aug_lag = augmented_lagrangian_objective(x, f, equality_constraints, lag_mul, mu_k)

    def maybe_log(stage, extra=None):
        if log_callback is None:
            return
        payload = {
            'stage': stage,
            'k': k,
            'x': np.array(x, copy=True),
            'c_vals': np.array(c_vals, copy=True),
            'c_norm': float(c_norm),
            'lag_mul': np.array(lag_mul, copy=True),
            'mu': np.array(mu_k, copy=True) if not np.isscalar(mu_k) else float(mu_k),
            'omega_k': float(omega_k),
            'eta_k': float(eta_k),
            'aug_lag': float(aug_lag),
            'time_sec': time.time() - start_time,
        }
        if extra:
            payload.update(extra)
        log_callback(payload)

    maybe_log('init')
    
    # Note: No gradient evaluation - scipy will use finite differences automatically

    if verbose:
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        # Handle mu_k formatting - it can be a scalar or array
        if np.isscalar(mu_k):
            mu_k_str = f"{mu_k:.2e}"
        else:
            mu_k_str = f"[{', '.join([f'{m:.2e}' for m in mu_k])}]"
        print(f"Iteration {0}, \u03BC_k={mu_k_str}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║g║ = {c_norm:.2e}")
        print(f'L_A value: {aug_lag:0.5f}')
        print("Using finite differences (no analytical gradients)")
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

    options = {
        'disp': False,
        'maxiter': MAXITER,
    }
    if minimize_method == 'L-BFGS-B':
        options['maxcor'] = 100

    # If the penalty parameter is too large, stop the optimization
    while (c_norm > c_tol) and k < MAXITER_lag:
        # Solve arg min of the augmented lagrangian
        dofs_before = x.copy()
        def fun(dofs):
            # Objective function only - scipy will use finite differences for gradient
            aug_lag = augmented_lagrangian_objective(
                dofs, f, equality_constraints, lag_mul, mu_k)
            return aug_lag
        options['gtol'] = omega_k

        # Note: No jac=True parameter means scipy uses finite differences automatically
        res = minimize(fun, x, method=minimize_method, options=options, tol=argmin_tol)
        grad_norm = np.linalg.norm(res.jac, np.inf) if res.jac is not None else np.inf
        if grad_norm < 1e-5 and c_norm < c_tol:
            break  # tighten your outer-loop condition to include grad_norm
        dofs_after = res.x
        if verbose:
            print('||Δx||:', np.linalg.norm(dofs_after - dofs_before))
        x = res.x
        
        if isinstance(f, SquaredFlux):
            curves = [c.curve for c in f.field.coils]  # f is assumed to be a SquaredFlux object 
            curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
        else:
            try:
                curves = [c.curve for c in equality_constraints[0].Jobj.field.coils]  # equality_constraints[0] is assumed to be a SquaredFlux object
                curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
            except:
                # print("WARNING: equality_constraints[0] is not a SquaredFlux object, so no coils will be visualized")
                pass

        # Evaluate constraints
        c_vals = np.array([J.J() for J in equality_constraints], dtype=float) if m_eq > 0 else np.array([], dtype=float)

        # Check convergence
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0.0

        # Print detailed progress
        if verbose:
            print(f"Iteration {k}")
            # print(f"  x = {x}")
            print(f"  Objective f(x) = {f.J()}")
            print(f"  Constraints g(x) = {c_vals}")
            print(f"  Constraint norm = {c_norm}")
            # No gradient norm in no-grad mode
            print(f"  Lagrange multipliers = {lag_mul}")
            # Handle mu_k formatting - it can be a scalar or array
            if np.isscalar(mu_k):
                print(f"  Penalty parameter mu_k = {mu_k}")
            else:
                print(f"  Penalty parameter mu_k = [{', '.join([f'{m:.2e}' for m in mu_k])}]")
            print(f"  Penalty parameter omega_k = {omega_k}")
            print(f"  Penalty parameter eta_k = {eta_k}")
            print(f"  Change in x = {np.linalg.norm(x - dofs_before)}")
            print("--------------------------------------------------")

        if m_eq > 0:
            # try:
            #### Requires the ordering of constraints to be in the order below
            if verbose:
                try:
                    ncoils = len(equality_constraints[0].field.coils) // (equality_constraints[0].surface.stellsym + 1) // (equality_constraints[0].surface.nfp)
                    print(f"  Normalized flux: {equality_constraints[0].J():.2e}")
                    print(f"  CS separation: {equality_constraints[1].J():.2e} (min distance: {equality_constraints[1].shortest_distance():.3f})")
                    print(f"  CC separation: {equality_constraints[2].J():.2e} (min distance: {equality_constraints[2].shortest_distance():.3f})")
                    print(f"  Length constraint: {equality_constraints[3].J():.2e}")
                    print(f"  Curvature constraint: {equality_constraints[4].J():.2e}")
                    print(f"  MSC Curvature constraint: {equality_constraints[5].J():.2e}")
                    print(f"  Linking number: {equality_constraints[6].J():.2e}")
                    print(f"  Force constraint: {equality_constraints[7].J():.2e}")
                    print(f"  Max curvatures: {[np.max(c.curve.kappa()) for c in equality_constraints[0].field.coils[:ncoils]]}")
                    print(f"  Lengths: {[CurveLength(c.curve).J() for c in equality_constraints[0].field.coils[:ncoils]]}")
                    print(f"  Total length: {sum([CurveLength(c.curve).J() for c in equality_constraints[0].field.coils[:ncoils]]):.2e}")
            #     print(f"Iteration {k}")
                # print('Deviation from target: Jf = {:.2e}, CC-Sep-min = {:.2e}, CS-Sep-min = {:.2e}, Len = {:.2e}, Curv = {:.2e}, Link = {:.2e}'.format(
                #     abs(c_vals[0]), abs(c_vals[1]), abs(c_vals[2]), abs(c_vals[3]), abs(c_vals[4]), abs(c_vals[5])))
            #     if verbose:
            #         print('Contributions to the objective:', lag_mul * c_vals, mu_k / 2 * np.linalg.norm(c_vals)**2)
                except:
                    pass
            c_str = f"Iter {k}: " + 'Jf = {:.2e}'.format(f.J()) + ', '
            for i, c in enumerate(c_vals):
                c_str += 'c{:d} = {:.2e}, '.format(i, abs(c))
            print(c_str)
        if verbose:
            try:
                print('Max curvatures:', [np.max(c.kappa()) for c in equality_constraints[1].Jobj.curves])
            except:
                pass
        # increase penalty only for violated constraints

        if not penalty_type and m_eq > 0:
            for i in range(m_eq):
                if abs(c_vals[i]) > c_tol:
                    mu_k[i] *= tau
        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            # constraints sufficiently small: update multipliers
            if verbose:
                print("*Constraints are satisfied*")
            if m_eq > 0:
                lag_mul += -mu_k * c_vals
            # tighten tolerances based on current worst‐case mu
            omega_k = max(omega_k / mu_k_scalar, c_tol)
            eta_k = max(eta_k / mu_k_scalar, c_tol)
        else:
            if verbose:
                print("*Constraints are not satisfied*")
            if not penalty_type:
                mu_k_scalar = np.mean(mu_k)
            else:
                mu_k = tau * mu_k
                mu_k_scalar = float(mu_k)
            omega_k = max(1.0 / mu_k_scalar, c_tol)
            eta_k = max(1.0 / (mu_k_scalar ** 0.1), c_tol)
        if verbose:
            print("LAGRANGE MULTIPLIERS:", lag_mul)
            print("--------------------------------------------------------------------------------------------------------------------------------------------")

        maybe_log('outer', extra={
            'grad_norm': float(grad_norm) if np.isfinite(grad_norm) else None,
            'minimize_success': bool(res.success),
            'minimize_status': int(res.status),
            'minimize_message': str(res.message),
            'minimize_nit': getattr(res, 'nit', None),
            'delta_x_norm': float(np.linalg.norm(x - dofs_before)),
        })

        k += 1

    if verbose:
        print('While loop finished because something became false: \n',
            # 'grad_tol_check = ', grad_aug_lag_norm > grad_tol,
            ', c_tol_check = ', c_norm > c_tol,
            ', maxiter_check = ', k < MAXITER_lag,
        )
    try:
        return x, res.fun, lag_mul
    except:  # while loop did not run a single time
        return x, None, lag_mul
