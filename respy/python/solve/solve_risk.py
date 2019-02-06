import numpy as np
from numba import njit
from respy.python.shared.shared_auxiliary import get_total_values


def construct_emax_risk(
    num_periods,
    num_draws_emax,
    period,
    k,
    draws_emax_risk,
    rewards_systematic,
    periods_emax,
    states_all,
    mapping_state_idx,
    edu_spec,
    optim_paras,
):
    """ Simulate expected future value for a given distribution of the unobservables.

    Parameters
    ----------
    num_periods : int
        Number of periods.
    num_draws_emax : int
        Number of draws.
    period : int
        Number of period.
    k : int
        ???
    draws_emax_risk : np.array
        Array with shape (num_draws_emax, num_rewards)
    rewards_systematic : np.array
        Array with shape (num_rewards)
    periods_emax : np.array
        Array with shape (num_periods, num_individuals???)
    states_all : np.array
        Array with shape (num_periods, num_individuals, num_rewards + 1)
    mapping_state_idx : np.array
        Array with shape (num_periods, num_periods, num_periods, 21, num_rewards, 1)
    edu_spec : namedtuple
        Keys are lagged, start, share, max
    optim_paras : namedtuple


    Returns
    -------
    float
        ???

    """
    # Antibugging
    assert np.all(draws_emax_risk[:, :2] >= 0)

    # Calculate maximum value
    emax = 0.0
    for i in range(num_draws_emax):

        # Select draws for this draw
        draws = draws_emax_risk[i, :]

        # Get total value of admissible states
        total_values, _ = get_total_values(
            period,
            num_periods,
            optim_paras,
            rewards_systematic,
            draws,
            edu_spec,
            mapping_state_idx,
            periods_emax,
            k,
            states_all,
        )

        # Determine optimal choice
        maximum = np.max(total_values)

        # Recording expected future value
        emax += maximum

    # Scaling
    emax /= num_draws_emax

    return emax
