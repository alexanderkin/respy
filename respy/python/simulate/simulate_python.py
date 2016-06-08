# standard library
import numpy as np
import logging

# project library
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_auxiliary import get_total_value
from respy.python.solve.solve_python import pyth_solve

logger = logging.getLogger('RESPY_SIMULATE')

''' Main function
'''


def pyth_simulate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
        periods_draws_emax, num_agents_sim, periods_draws_sims):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """

    # First we need to solve the underlying model.
    periods_payoffs_systematic, states_number_period, mapping_state_idx, \
        periods_emax, states_all = \
        pyth_solve(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
            is_interpolated, num_draws_emax, num_periods, num_points_interp,
            is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
            periods_draws_emax)


    # Standard deviates transformed to the distributions relevant for
    # the agents actual decision making as traversing the tree.
    periods_draws_sims_transformed = np.tile(np.nan,
        (num_periods, num_agents_sim, 4))

    for period in range(num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :], shocks_cholesky)

    # Simulate agent experiences
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents_sim * num_periods, 8))

    for i in range(num_agents_sim):

        current_state = states_all[0, 0, :].copy()

        dataset[count, 0] = i

        # Logging
        if (i != 0) and (i % 100 == 0):
            logger.info('... simulated ' + str(i) + ' agents')

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            payoffs_systematic = periods_payoffs_systematic[period, k, :]
            draws = periods_draws_sims_transformed[period, i, :]

            # Get total value of admissible states
            total_payoffs = get_total_value(period,
                num_periods, delta, payoffs_systematic, draws, edu_max,
                edu_start, mapping_state_idx, periods_emax, k, states_all)

            # Determine optimal choice
            max_idx = np.argmax(total_payoffs)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record earnings
            dataset[count, 3] = MISSING_FLOAT
            if max_idx in [0, 1]:
                dataset[count, 3] = payoffs_systematic[max_idx] * draws[max_idx]

            # Write relevant state space for period to data frame
            dataset[count, 4:8] = current_state

            # Special treatment for education
            dataset[count, 6] += edu_start

            # Update work experiences and education
            if max_idx == 0:
                current_state[0] += 1
            elif max_idx == 1:
                current_state[1] += 1
            elif max_idx == 2:
                current_state[2] += 1

            # Update lagged education
            current_state[3] = 0

            if max_idx == 2:
                current_state[3] = 1

            # Update row indicator
            count += 1

    # Finishing
    return dataset
