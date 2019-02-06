import os
import shlex

import numpy as np
import statsmodels.api as sm
from numba import njit, objmode

from respy.python.record.record_solution import record_solution_prediction
from respy.python.record.record_solution import record_solution_progress
from respy.python.shared.shared_auxiliary import calculate_rewards_common
from respy.python.shared.shared_auxiliary import calculate_rewards_general
from respy.python.shared.shared_auxiliary import construct_covariates
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import MISSING_INT
from respy.python.solve.solve_risk import construct_emax_risk
import respy.python.shared.fast_routines as fr
from numba import jit


def pyth_create_state_space(num_periods, num_types, edu_spec):
    """ Create grid for state space.

    Parameters
    ----------
    num_periods : int
    num_types
    edu_spec : namedtuple
        Contains educational specification with keys lagged, start, share and max.

    Returns
    -------
    states_all : np.array
        Array with shape (num_periods, 100000, 5)
    states_number_period
    mapping_state_idx
    max_states_period

    # TODO: Cannot be jitted until edu_spec is not dict anymore.

    # TODO: Deprecate return argument max_states_period as it can be derived from
    # states_number_period.

    Examples
    --------
    >>> from respy.python.shared.data_classes import education_specification
    >>> num_periods = 40
    >>> num_types = 1
    >>> edu_spec = education_specification(
    ...     lagged=[1.0], start=[10], share=[1.0], max=20
    ... )
    >>> res = pyth_create_state_space(num_periods, num_types, edu_spec)
    >>> assert res[0].shape == (40, 100000, 5)
    >>> assert res[1].shape == (40,)
    >>> assert res[2].shape == (40, 40, 40, 21, 4, 1)
    >>> assert res[3] == 26348

    """
    # Auxiliary information
    min_idx = edu_spec.max + 1

    # TODO: We should not hardcode dimensions. Especially 100000
    # Array for possible realization of state space by period
    states_all = np.full((num_periods, 100000, 5), MISSING_INT)

    # Array for the mapping of state space values to indices in variety of matrices.
    shape = (num_periods, num_periods, num_periods, min_idx, 4, num_types)
    mapping_state_idx = np.full(shape, MISSING_INT)

    # Array for maximum number of realizations of state space by period
    states_number_period = np.full(num_periods, MISSING_INT)

    # Construct state space by periods
    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all unobserved types
        for type_ in range(num_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_spec.start:

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(num_periods + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(num_periods + 1):

                        # Loop over all admissible additional education levels
                        for edu_add in range(num_periods + 1):

                            # Check if admissible for time constraints. Note that the
                            # total number of activities does not have is less or equal
                            # to the total possible number of activities as the rest is
                            # implicitly filled with leisure.
                            if edu_add + exp_a + exp_b > period:
                                continue

                            # Agent cannot attain more additional education than
                            # (EDU_MAX - EDU_START).
                            if edu_add > (edu_spec.max - edu_start):
                                continue

                            # Loop over all admissible values for the lagged activity:
                            # (1) Occupation A, (2) Occupation B, (3) Education, and (4)
                            # Home.
                            for choice_lagged in [1, 2, 3, 4]:

                                if period > 0:

                                    # (0, 1) Whenever an agent has only worked in
                                    # Occupation A, then the lagged choice cannot be
                                    # anything other than one.
                                    if (choice_lagged != 1) and (exp_a == period):
                                        continue

                                    # (0, 2) Whenever an agent has only worked in
                                    # Occupation B, then the lagged choice cannot be
                                    # anything other than two
                                    if (choice_lagged != 2) and (exp_b == period):
                                        continue

                                    # (0, 3) Whenever an agent has only acquired
                                    # additional education, then the lagged choice
                                    # cannot be anything other than three..
                                    if (choice_lagged != 3) and (edu_add == period):
                                        continue

                                    # (0, 4) Whenever an agent has not acquired any
                                    # additional education and we are not in the first
                                    # period, then lagged activity cannot take a value
                                    # of three.
                                    if (choice_lagged == 3) and (edu_add == 0):
                                        continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. The
                                # cannot have any work experience.
                                if period == 0:
                                    if choice_lagged in [1, 2]:
                                        continue

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if (choice_lagged == 1) and (exp_a == 0):
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if (choice_lagged == 2) and (exp_b == 0):
                                    continue

                                # If we have multiple initial conditions it might well
                                # be the case that we have a duplicate state, i.e. the
                                # same state is possible with other initial condition
                                # that period.
                                if (
                                    mapping_state_idx[
                                        period,
                                        exp_a,
                                        exp_b,
                                        edu_start + edu_add,
                                        choice_lagged - 1,
                                        type_,
                                    ]
                                    != MISSING_INT
                                ):
                                    continue

                                # Collect mapping of state space to array index.
                                mapping_state_idx[
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged - 1,
                                    type_,
                                ] = k

                                # Collect all possible realizations of state space
                                states_all[period, k, :] = [
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged,
                                    type_,
                                ]

                                # Update count
                                k += 1

        # Record maximum number of state space realizations by time period
        states_number_period[period] = k

    max_states_period = np.max(states_number_period)

    return (states_all, states_number_period, mapping_state_idx, max_states_period)


def pyth_calculate_rewards_systematic(
    num_periods, states_number_period, states_all, max_states_period, optim_paras
):
    """ Calculate ex systematic rewards.

    Parameters
    ----------
    num_periods : int
        Number of periods
    states_number_period : np.array
        Number of states per period.
    states_all
        Array with shape (num_periods, max_states_period, num_state_space)
    max_states_period : int
        Maximum number states in period
    optim_paras : dict
        ???

    """
    # Initialize
    shape = (num_periods, max_states_period, 4)
    periods_rewards_systematic = np.full(shape, MISSING_FLOAT)

    # Calculate systematic instantaneous rewards
    for period in range(num_periods - 1, -1, -1):

        # Loop over all possible states
        for k in range(states_number_period[period]):

            # Distribute state space
            exp_a, exp_b, edu, choice_lagged, type_ = states_all[period, k, :]

            # Initialize container
            rewards = np.full(4, np.nan)

            # Construct auxiliary information
            covariates = construct_covariates(
                exp_a, exp_b, edu, choice_lagged, type_, period
            )

            # Calculate common and general rewards component.
            rewards_general = calculate_rewards_general(covariates, optim_paras)
            rewards_common = calculate_rewards_common(
                covariates, optim_paras.coeffs_common
            )

            # Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards.
            # These are defined in a general sense, where not only wages matter.
            wages = calculate_wages_systematic(
                covariates,
                optim_paras.coeffs_a,
                optim_paras.coeffs_b,
                optim_paras.type_shifts,
            )

            rewards[:2] = wages[:2] + rewards_general[:2]

            # Calculate systematic part of SCHOOL rewards
            covars_edu = []
            covars_edu += [1]
            covars_edu += list(covariates[15:19])  # hs or co graduate and returns
            covars_edu += [covariates.period]
            covars_edu += [covariates.is_minor]

            rewards[2] = optim_paras.coeffs_edu.dot(covars_edu)

            # Calculate systematic part of HOME
            covars_home = []
            covars_home += [1]
            covars_home += list(covariates[20:22])  # Age 18 - 20 or over 21

            rewards[3] = optim_paras.coeffs_home.dot(covars_home)

            # Now we add the type-specific deviation for SCHOOL and HOME.
            rewards[2:4] = rewards[2:4] + optim_paras.type_shifts[type_, 2:4]

            # We can now also added the common component of rewards.
            rewards[:4] = rewards[:4] + rewards_common

            periods_rewards_systematic[period, k, :] = rewards

    return periods_rewards_systematic


def pyth_backward_induction(
    num_periods,
    is_myopic,
    max_states_period,
    periods_draws_emax,
    num_draws_emax,
    states_number_period,
    periods_rewards_systematic,
    mapping_state_idx,
    states_all,
    is_debug,
    is_interpolated,
    num_points_interp,
    edu_spec,
    optim_paras,
    file_sim,
    is_write,
):
    """ Backward induction procedure.

    There are two main threads to this function depending on whether interpolation is
    requested or not.

    # TODO: Many 4's are hardcoded.

    """
    # Initialize containers, which contain a lot of missing values as we capture the
    # tree structure in arrays of fixed dimension.
    periods_emax = np.full((num_periods, max_states_period), MISSING_FLOAT)

    if is_myopic:
        with objmode():
            record_solution_progress(-2, file_sim)

        # TODO: Find beautiful way to mask different lengths of rows and set them to 0.
        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

        return periods_emax

    # Construct auxiliary objects
    shocks_cov = optim_paras.shocks_cholesky.dot(optim_paras.shocks_cholesky.T)

    # Auxiliary objects. These shifts are used to determine the expected values of the
    # two labor market alternatives. These are log normal distributed and thus the draws
    # cannot simply set to zero.
    shifts = np.zeros(4)
    shifts[:2] = np.clip(np.exp(shocks_cov[np.diag_indices(2)] / 2.0), 0.0, HUGE_FLOAT)

    # Initialize containers with missing values
    periods_emax = np.full((num_periods, max_states_period), MISSING_FLOAT)

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        # Extract auxiliary objects
        draws_emax_standard = periods_draws_emax[period, :, :]
        num_states = states_number_period[period]

        # Treatment of the disturbances for the risk-only case is straightforward. Their
        # distribution is fixed once and for all.
        draws_emax_risk = transform_disturbances(
            draws_emax_standard, np.zeros(4), optim_paras.shocks_cholesky
        )

        if is_write:
            with objmode():
                record_solution_progress(4, file_sim, period, num_states)

        # The number of interpolation points is the same for all periods. Thus, for some
        # periods the number of interpolation points is larger than the actual number of
        # states. In that case no interpolation is needed.
        any_interpolated = (num_points_interp <= num_states) and is_interpolated

        # Case distinction
        if any_interpolated:
            # Get indicator for interpolation and simulation of states
            is_simulated = get_simulated_indicator(
                num_points_interp, num_states, period, is_debug
            )

            # Constructing the exogenous variable for all states, including the ones
            # where simulation will take place. All information will be used in either
            # the construction of the prediction model or the prediction step.
            exogenous, maxe = get_exogenous_variables(
                period,
                num_periods,
                num_states,
                periods_rewards_systematic,
                shifts,
                mapping_state_idx,
                periods_emax,
                states_all,
                edu_spec,
                optim_paras,
            )

            # Constructing the dependent variables for at the random subset of points
            # where the EMAX is actually calculated.
            endogenous = get_endogenous_variable(
                period,
                num_periods,
                num_states,
                periods_rewards_systematic,
                mapping_state_idx,
                periods_emax,
                states_all,
                is_simulated,
                num_draws_emax,
                maxe,
                draws_emax_risk,
                edu_spec,
                optim_paras,
            )

            # Create prediction model based on the random subset of points where the
            # EMAX is actually simulated and thus dependent and independent variables
            # are available. For the interpolation points, the actual values are used.
            predictions = get_predictions(
                endogenous, exogenous, maxe, is_simulated, file_sim, is_write
            )

            # Store results
            periods_emax[period, :num_states] = predictions

        else:
            # Loop over all possible states
            for k in range(states_number_period[period]):

                rewards_systematic = periods_rewards_systematic[period, k, :]

                emax = construct_emax_risk(
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
                )

                periods_emax[period, k] = emax

    return periods_emax


def get_simulated_indicator(num_points_interp, num_candidates, period, is_debug):
    """ Get the indicator for points of interpolation and simulation.

    """
    # Drawing random interpolation points
    interpolation_points = np.random.choice(
        range(num_candidates), size=num_points_interp, replace=False
    )

    # Constructing an indicator whether a state will be simulated or interpolated.
    is_simulated = np.full(num_candidates, False)
    is_simulated[interpolation_points] = True

    # Check for debugging cases.
    with objmode("bool"):
        is_standardized = is_debug and os.path.exists(".interpolation.respy.test")
        if is_standardized:
            with open(".interpolation.respy.test", "r") as file_:
                indicators = []
                for line in file_:
                    indicators += [(shlex.split(line)[period] == "True")]
            is_simulated = indicators[:num_candidates]

    is_simulated = np.array(is_simulated)

    return is_simulated


def get_exogenous_variables(
    period,
    num_periods,
    num_states,
    periods_rewards_systematic,
    shifts,
    mapping_state_idx,
    periods_emax,
    states_all,
    edu_spec,
    optim_paras,
):
    """ Get exogenous variables for interpolation scheme. The unused argument is present
    to align the interface between the PYTHON and FORTRAN implementations.

    """
    # Construct auxiliary objects
    exogenous = np.full((num_states, 9), np.nan)
    maxe = np.full(num_states, np.nan)

    # Iterate over all states.
    for k in range(num_states):

        # Extract systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Get total value
        total_values, _ = get_total_values(
            period,
            num_periods,
            optim_paras,
            rewards_systematic,
            shifts,
            edu_spec,
            mapping_state_idx,
            periods_emax,
            k,
            states_all,
        )

        # Implement level shifts
        maxe[k] = np.max(total_values)

        diff = maxe[k] - total_values

        exogenous[k, :8] = np.hstack((diff, np.sqrt(diff)))

        # Add intercept to set of independent variables and replace infinite values.
        exogenous[:, 8] = 1

    return exogenous, maxe


def get_endogenous_variable(
    period,
    num_periods,
    num_states,
    periods_rewards_systematic,
    mapping_state_idx,
    periods_emax,
    states_all,
    is_simulated,
    num_draws_emax,
    maxe,
    draws_emax_risk,
    edu_spec,
    optim_paras,
):
    """ Construct endogenous variable for the subset of interpolation points.

    """
    # Construct auxiliary objects
    endogenous_variable = np.full(num_states, np.nan)

    for k in range(num_states):

        # Skip over points that will be interpolated and not simulated.
        if not is_simulated[k]:
            continue

        # Extract rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Simulate the expected future value.
        emax = construct_emax_risk(
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
        )

        # Construct dependent variable
        endogenous_variable[k] = emax - maxe[k]

    return endogenous_variable


@jit(nopython=False)
def get_predictions(endogenous, exogenous, maxe, is_simulated, file_sim, is_write):
    """ Fit an OLS regression of the exogenous variables on the endogenous variables and
    use the results to predict the endogenous variables for all points in the state
    space.

    """
    # Define ordinary least squares model and fit to the data.
    model = sm.OLS(endogenous[is_simulated], exogenous[is_simulated])
    results = model.fit()

    # Use the model to predict EMAX for all states. As in Keane & Wolpin (1994),
    # negative predictions are truncated to zero.
    endogenous_predicted = results.predict(exogenous)
    endogenous_predicted = fr.clip(endogenous_predicted, 0.00, None)

    # Construct predicted EMAX for all states and the replace interpolation points with
    # simulated values.
    predictions = endogenous_predicted + maxe
    predictions[is_simulated] = endogenous[is_simulated] + maxe[is_simulated]

    check_prediction_model(endogenous_predicted, model)

    # Write out some basic information to spot problems easily.
    if is_write:
        record_solution_prediction(results, file_sim)

    return predictions


def check_prediction_model(predictions_diff, model):
    """ Perform some basic consistency checks for the prediction model.
    """
    # Construct auxiliary object
    results = model.fit()
    # Perform basic checks
    assert np.all(predictions_diff >= 0.00)
    assert results.params.shape == (9,)
    assert np.all(np.isfinite(results.params))


def check_input(respy_obj):
    """ Check input arguments.

    """
    # Check that class instance is locked.
    assert respy_obj.get_attr("is_locked")

    # Check for previous solution attempt.
    if respy_obj.get_attr("is_solved"):
        respy_obj.reset()

    return True


def calculate_wages_systematic(covariates, coeffs_a, coeffs_b, type_shifts):
    """ Calculate the systematic component of wages.

    Examples
    --------
    >>> from respy.python.shared.data_classes import Covariates
    >>> covariates = Covariates(*[
    ...     0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 39, 0, 0, 0, 10, 0, 0, 1, 0, 0, 0, 1
    ... ])
    >>> coeffs_a = np.array([
    ...     9.21e+00,  3.80e-02,  3.30e-02, -5.00e-04,  0.00e+00, -0.00e+00,
    ...     0.00e+00, -0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
    ...     0.00e+00,  0.00e+00,  0.00e+00
    ... ])
    >>> coeffs_b = np.array([
    ...     8.48e+00,  7.00e-02,  2.20e-02, -5.00e-04,  6.70e-02, -1.00e-03,
    ...     0.00e+00, -0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
    ...     0.00e+00,  0.00e+00,  0.00e+00
    ... ])
    >>> type_shifts = np.zeros((1, 4))
    >>> calculate_wages_systematic(covariates, coeffs_a, coeffs_b, type_shifts)
    array([14617.86953434,  9701.15277293])

    """
    # Collect all relevant covariates
    covs = np.array(
        [
            [
                1,
                covariates.edu,
                covariates.exp_a,
                covariates.exp_a ** 2 / 100,
                covariates.exp_b,
                covariates.exp_b ** 2 / 100,
                covariates.hs_graduate,
                covariates.co_graduate,
                covariates.period,
                covariates.is_minor,
            ]
        ]
        * 2
    )

    # TODO: OBJECT_MODE, performance critical

    # Revert the scaling. This used for testing purposes, where we compare the results
    # from the RESPY package to the original RESTUD program.
    with objmode():
        if os.path.exists(".restud.respy.scratch"):
            covs[:, 3] *= 100.00
            covs[:, 5] *= 100.00

    covs_occ = np.array(
        [
            [covariates.any_exp_a, covariates.work_a_lagged],
            [covariates.any_exp_b, covariates.work_a_lagged],
        ]
    )
    covs = np.hstack((covs, covs_occ))

    wages = np.exp(np.diag(covs.dot(np.vstack((coeffs_a[:12], coeffs_b[:12])).T)))

    wages = fr.clip(wages, 0.0, 1000000.0)

    # We need to add the type-specific deviations here as these are part of
    # skill-function component.
    wages = wages * np.exp(type_shifts[covariates.type, :2])

    return wages
