""" This modules contains the tests for the continuous integration efforts.
"""

# standard library
import numpy as np

import sys
import os

# project library
from modules._random_init import generate_init

# robustToolbox
sys.path.insert(0, os.environ['ROBUPY'])

from robupy import *

''' Main
'''
def test_A():
    """ Testing whether ten random initialization file can be
    solved and simulated.
    """
    for i in range(10):

        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        simulate(robupy_obj)

        # Distribute class attributes
        num_periods = robupy_obj.get_attr('num_periods')

        emax = robupy_obj.get_attr('emax')

        # Check that the expected future value is always
        # zero in the last period.
        assert (np.all(emax[(num_periods - 1), :] == 0.00))

    # Finishing
    return True

def test_B():
    """ Testing ten admissible realizations of state space
    for the first three periods.
    """
    for i in range(10):

        # Generate constraint periods
        constraints = dict()
        constraints['periods'] = np.random.randint(3, 10)

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        simulate(robupy_obj)

        # Distribute class attributes
        states_number_period = robupy_obj.get_attr('states_number_period')

        states_all = robupy_obj.get_attr('states_all')

        # The next harded-coded results assume that at least two more
        # years of education are admissible.
        edu_max = robupy_obj.get_attr('edu_max')
        edu_start = robupy_obj.get_attr('edu_start')

        if edu_max - edu_start < 2:
            continue

        # The number of admissible states in the first three periods
        for j, number_period in enumerate([1, 4, 13]):
            assert (states_number_period[j] == number_period)

        # The actual realizations of admissible states in period one
        assert ((states_all[0, 0, :] == [0, 0, 0, 1]).all())

        # The actual realizations of admissible states in period two
        states = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]]
        states += [[1, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[1, j, :] == state).all())

        # The actual realizations of admissible states in period three
        states = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
        states += [[0, 0, 2, 1], [0, 1, 0, 0], [0, 1, 1, 0]]
        states += [[0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 0]]
        states += [[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]]
        states += [[2, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[2, j, :] == state).all())

    # Finishing
    return True

def test_C():
    """ Testing whether the ex ante and ex post payoffs
    are identical if there is no random variation
    in the payoffs
    """
    for i in range(10):

        # Generate constraint periods
        constraints = dict()
        constraints['eps_zero'] = True

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Distribute class attributes
        ex_ante = robupy_obj.get_attr('period_payoffs_ex_ante')
        ex_post = robupy_obj.get_attr('period_payoffs_ex_post')

        # Check
        assert (np.ma.all(np.ma.masked_invalid(ex_ante) == np.ma.masked_invalid(ex_post)))

    # Finishing
    return True

def test_D():
    """ If there is no random variation in payoffs
    then the number of draws to simulate the
    expected future value should have no effect.
    """
    # Generate constraint periods
    constraints = dict()
    constraints['eps_zero'] = True

    # Generate random initialization file
    generate_init(constraints)

    # Initialize auxiliary objects
    base = None

    for _ in range(10):

        # Draw a random number of draws for
        # expected future value calculations.
        num_draws = np.random.randint(1, 100)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj.unlock()

        robupy_obj.set_attr('num_draws', num_draws)

        robupy_obj.lock()

        robupy_obj = solve(robupy_obj)

        # Distribute class attributes
        emax = robupy_obj.get_attr('emax')

        if base is None:
            base = emax.copy()

        # Statistic
        diff = np.max(abs(np.ma.masked_invalid(base) - np.ma.masked_invalid(
            emax)))

        # Checks
        assert (np.isfinite(diff))
        assert (diff < 10e-15)

    # Finishing
    return True