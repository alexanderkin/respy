"""This module contains fast routines or replacements of numpy functions."""

from numba import njit
import numpy as np


@njit
def clip(a, a_min=None, a_max=None):
    """Replacement for np.clip as long it is not supported by Numba.

    Mirrors functionality from
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html.

    Support is currently planned for Numba version 0.43.0. The PR on Github is
    https://github.com/numba/numba/pull/3468.

    """
    a_min_is_none = a_min is None
    a_max_is_none = a_max is None

    if a_min_is_none and a_max_is_none:
        raise ValueError("array_clip: must set either max or min")

    elif a_min_is_none:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val > a_max:
                ret[index] = a_max
            else:
                ret[index] = val
        return ret

    elif a_max_is_none:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val < a_min:
                ret[index] = a_min
            else:
                ret[index] = val
        return ret

    else:
        ret = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val < a_min:
                ret[index] = a_min
            elif val > a_max:
                ret[index] = a_max
            else:
                ret[index] = val
        return ret
