import numpy as np


def F(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2), ord=2)


def heaviside(x: float):
    if x >= 0:
        return 1
    else:
        return 0


def delta(x, err):
    if abs(x) <= err:
        return 1
    else:
        return 0


def float_equal(a: float, b: float, accuracy: float = 1e-10):
    return abs(a - b) <= accuracy


def interpolate_coefs(fd_depth, md_depth):
    """Calculate coefficients for linear interpolation FROM md_depth TO fd_depth depth grid.

    Parameters
    ----------
    fd_depth : numpy.ndarray, shape(m1,)
        Array of depth grid we make interpolation TO.
    md_time : numpy.ndarray, shape(m2,)
        Array of depth grid we make interpolation FROM.

    Returns
    ----------
    coeffs : list of point_coeff, where
        point_coeff : list[ind1, ind2, k1, k2]
            Coefficients for each point d in fd_depth: ind1 and ind2 are the indexes of points in md_depth around d, k1 and k2 are linear interpolation coefficients.

    """
    coeffs = []
    if len(fd_depth) == 1:
        return coeffs

    for d in fd_depth:
        for i in range(len(md_depth) - 1):
            if (md_depth[i] <= d < md_depth[i + 1]) or (md_depth[i] < d <= md_depth[i + 1]):
                ind1 = i
                ind2 = ind1 + 1
                k2 = (d - md_depth[ind1]) / (md_depth[ind2] - md_depth[ind1])
                k1 = 1 - k2
                coeffs.append([ind1, ind2, k1, k2])
                break
    return coeffs


def interpolate(md, coeffs_t):
    """Make linear interpolation for md (2D data array) using coeffs_t (interpolation along time grid) and coeffs_d (interpolation along depth grid).

    Parameters
    ----------
    md : numpy.ndarray, shape(n, m)
        2D array of data values before interpolation.
    coeffs_t : list, length is n1
        Coefficients for interpolation. See interpolateTime.
    coeffs_d : list, length is m1
        Coefficients for interpolation. See interpolateDepth.

    Returns
    ----------
    data : numpy.ndarray, shape(n1, m1)
        2D array of data values after interpolation.

    """
    coeffs_t = np.array(coeffs_t)

    t_i1 = coeffs_t.T[0].astype('int32')
    t_i2 = coeffs_t.T[1].astype('int32')
    t_k1 = coeffs_t.T[2]
    t_k2 = coeffs_t.T[3]

    data = md[t_i1] + np.multiply(t_k2.reshape(t_k2.shape[0], 1), (md[t_i2] - md[t_i1]))

    return data
