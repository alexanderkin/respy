""" This module contains the functions related to the incorporation of
    ambiguity in the model.
"""

# standard library
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# project library
from robupy.checks._checks_ambiguity import _checks
from robupy.shared import *


''' Public functions
'''


def simulate_emax_ambiguity(num_draws, period_payoffs_ex_post, eps_standard,
        period, k, payoffs_ex_ante, edu_max, edu_start,
        mapping_state_idx, states_all, future_payoffs, num_periods, emax,
        ambiguity, true_cholesky, delta):
    """ Get worst case
    """

    # Initialize options.
    options = dict()

    options['maxiter'] = 10000

    debug = ambiguity['debug']

    # Initialize optimization problem.
    x0 = _get_start(ambiguity)

    bounds = _get_bounds(ambiguity)

    # Collect arguments
    args = (num_draws, period_payoffs_ex_post, eps_standard, period,
            k, payoffs_ex_ante, edu_max, edu_start,
            mapping_state_idx, states_all, future_payoffs,
            num_periods, emax, true_cholesky, delta)

    # Run optimization
    opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                   bounds=bounds)

    # Write result to file
    if debug:
        _write_result(period, k, opt)


    print('\n\n')
    print(opt['message'])
    print(x0)
    print(bounds)
    # Extract information
    fun = opt['fun']

    # Finishing
    return fun


''' Private functions
'''

def _write_result(period, k, opt):
    """ Write result of optimization problem to loggging file.
    """

    file_ = open('ambiguity.robupy.log', 'a')

    file_.write('Period ' + str(period) + '    State ' + str(k) + '\n' +
                    '-------------------\n\n')

    file_.write('Result  ' + str(opt['x']) + '\n')
    file_.write('Success ' + str(opt['success']) + '\n')
    file_.write('Message ' + opt['message'] + '\n\n\n')

    file_.close()

def _criterion(x, num_draws, period_payoffs_ex_post, eps_standard, period,
                   k, payoffs_ex_ante, edu_max, edu_start,
                   mapping_state_idx, states_all, future_payoffs,
                   num_periods, emax, true_cholesky, delta):
    """ Simulate expected future value for alternative shock distributions.
    """
    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = np.dot(true_cholesky, eps_standard.T).T + x
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Simulate the expected future value for a given parameterization.
    simulated, _, _ = \
        simulate_emax(num_draws, period_payoffs_ex_post, period, k,
                      eps_relevant, payoffs_ex_ante, edu_max,
                      edu_start, num_periods, emax, states_all, future_payoffs,
                      mapping_state_idx, delta)

    assert (np.isfinite(simulated))

    # Finishing
    return simulated


def _get_start(ambiguity):
    """ Get starting value.
    """
    # Distribute information
    measure = ambiguity['measure']
    level = ambiguity['level']
    para = ambiguity['para']

    if measure == 'absolute':

        x0 = [0.00, 0.00, 0.00, 0.00]

    return x0


def _get_bounds(ambiguity):
    """ Get bounds.
    """
    # Distribute information
    measure = ambiguity['measure']
    level = ambiguity['level']
    para = ambiguity['para']

    if measure == 'absolute':

        bounds = [[-level, level], [-level, level],
                      [-level, level], [-level, level]]

    return bounds

