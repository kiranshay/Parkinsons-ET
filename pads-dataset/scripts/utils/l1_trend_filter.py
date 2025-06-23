import numpy as np
import scipy
import cvxpy as cp


def l1_trend_filter(y, vlambda=50, verbose=True):
    """
    Apply l1-trend-filtering to time series y.

    Parameters
    ----------
    vlambda : int, default = 50
        Regularization parameter.
    verbose : bool, default = True
        Whether to print out information.
    """
    n = y.size

    # Form second difference matrix
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)

    # Solve l1 trend filtering problem
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                      + vlambda * cp.norm(D * x, 1))
    prob = cp.Problem(obj)

    # ECOS and SCS solvers fail to converge before the iteration limit. Use CVXOPT instead.
    prob.solve(solver=cp.CVXOPT, verbose=verbose)
    print(f'Solver status: {prob.status}')

    # Check for error.
    if prob.status != cp.OPTIMAL:
        raise Exception('Solver did not converge!')

    print(f'optimal objective value: {obj.value}')

    return x.value
