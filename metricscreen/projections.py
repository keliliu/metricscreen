import numpy as np


def threshold(x, t):
    return np.sign(x) * np.clip(np.abs(x) - t, a_min=0, a_max=np.Inf)


def binary_solver(f, lower, upper, tol=1e-10):
    """
    Returns x such that f(x) = 0
    :param f: Univariate, monotone decreasing function.
    :param lower: Lower bound on solution x.
    :param upper: Upper bound on solution x.
    :return: Numeric solving f(x) = 0
    """

    x = (lower + upper) / 2
    val = f(x)
    if np.abs(val) < tol:
        return x
    elif val > 0:
        return binary_solver(f, x, upper)
    else:
        return binary_solver(f, lower, x)


def _clipped_l1_norm(x, c, lower, upper):
    return np.sum(np.clip(x - c, a_min=lower, a_max=upper))


def project_simplex_with_box_constraints(x, l1_norm=1, linf=np.Infinity):

    assert l1_norm > 0, "Norm constraint must be positive."
    assert linf > 0, "Norm constraint must be positive."

    if linf >= l1_norm: return project_positive_simplex(x, l1_norm)

    xclip = np.clip(x, a_min=0, a_max=linf)
    if np.sum(xclip) <= l1_norm:
        return xclip

    MM = np.max(x)

    mu = binary_solver(lambda u: _clipped_l1_norm(x, u, 0, linf) - l1_norm, 0, MM)
    return np.clip(x - mu, a_min=0, a_max=linf)


def project_simplex_with_subgroup_l1(x, l1_norm=1, subgroup=None, subgroup_l1=np.Infinity):
    """
    Projection of x onto set where z_i >=0, sum_i z_i <= l1_norm, and sum_subgroup z_i <= subgroup_l1
    :param x: np.ndarray
    :param l1_norm: positive real
    Constraint on l1 norm of x.
    :param subgroup: np.ndarray of indices
    Indices for a subset of elements of x.
    :param subgroup_l1: positive real
    Constraint on l1 norm of a subset of elements of x.
    :return: np.ndarray
    """

    if subgroup is None:
        return project_positive_simplex(x, l1_norm=l1_norm)

    xplus = np.clip(x, a_min=0, a_max=np.Infinity)

    a = np.sum(xplus)
    b = np.sum(xplus[subgroup])

    # Case 1: Both constraints satisfied.
    if a <= l1_norm and b <= subgroup_l1:
        return xplus

    # Case 2: Constraint on subgroup is tight but constraint on x is loose.
    if (subgroup_l1 + (a - b)) <= l1_norm:  # a-b is sum of elements not in the subgroup
        xplus[subgroup] = project_positive_simplex(x[subgroup], l1_norm=subgroup_l1)
        return xplus


    # Case 3: Constraint on x is tight but subgroup constraint is loose.
    xhat = project_positive_simplex(x, l1_norm=l1_norm)
    if np.sum(xhat[subgroup]) <= subgroup_l1:
        return xhat

    not_subgroup = np.delete(np.arange(0, len(x), 1), subgroup)
    # Case 4: Neither constraint satisfied. At solution both constraints are tight.
    xplus[subgroup] = project_positive_simplex(x[subgroup], l1_norm=subgroup_l1)
    xplus[not_subgroup] = project_positive_simplex(x[not_subgroup], l1_norm=l1_norm - subgroup_l1)
    return xplus


def project_positive_simplex(x, l1_norm=1):

    assert type(x) is np.ndarray, "Input should be numeric numpy array."

    xplus = np.clip(x, a_min=0, a_max=np.Infinity)
    if np.sum(xplus) <= l1_norm:
        return xplus

    ind = np.argsort(x)
    xsort = x[ind]

    n = len(x)
    i = n

    tail_sum = 0
    while i > 0:
        i = i - 1
        tail_sum += xsort[i]
        mu = (tail_sum - l1_norm) / (n - i)
        if (i == 0) or (mu >= xsort[i - 1]):
            break

    return np.clip(x - mu, a_min=0, a_max=np.Infinity)
