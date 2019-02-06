import linecache
import os
import shlex
from collections import namedtuple

import numpy as np
from numba import njit, jit

from respy.custom_exceptions import MaxfunError
from respy.custom_exceptions import UserError
from respy.python.record.record_warning import record_warning
from respy.python.shared import fast_routines as fr
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.python.shared.shared_constants import TINY_FLOAT
from respy.python.shared.data_classes import optimization_parameters, Covariates


def get_log_likl(contribs):
    """Aggregate contributions to the likelihood value.

    Parameters
    ----------
    contribs : np.array ???
        Individual contributions to the log likelihood value.

    Returns
    -------
    float
        Value of log likelihood function.

    """
    if (np.abs(contribs) > HUGE_FLOAT).sum() > 0:
        record_warning(5)

    return -np.mean(fr.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT))


def distribute_parameters(paras_vec, is_debug=False, info=None, paras_type="optim"):
    """Parse the parameter vector into a dictionary of model quantities.

    Parameters
    ----------
    paras_vec : np.ndarray
        1d numpy array with the parameters
    is_debug : bool
        If true, the parameters are checked for validity
    info :
        ???
    paras_type : str
        One of ['econ', 'optim']. A paras_vec of type 'econ' contains the the standard
        deviations and covariances of the shock distribution. This is how parameters are
        represented in the .ini file and the output of .fit(). A paras_vec of type
        'optim' contains the elements of the cholesky factors of the covariance matrix
        of the shock distribution. This type is used internally during the likelihood
        estimation. The default value is 'optim' in order to make the function more
        aligned with Fortran, where we never have to parse 'econ' parameters.

    TODO: Transform optim_paras eventually

    """
    paras_vec = paras_vec.copy()
    assert paras_type in ["econ", "optim"], "paras_type must be econ or optim."

    if is_debug and paras_type == "optim":
        _check_optimization_parameters(paras_vec)

    pinfo = paras_parsing_information(len(paras_vec))
    optim_paras = {}

    # basic extraction
    for quantity in pinfo:
        start = pinfo[quantity]["start"]
        stop = pinfo[quantity]["stop"]
        optim_paras[quantity] = paras_vec[start:stop]

    # modify the shock_coeffs
    if paras_type == "econ":
        shocks_cholesky = coeffs_to_cholesky(optim_paras["shocks_coeffs"])
    else:
        shocks_cholesky, info = extract_cholesky(paras_vec, info)
    optim_paras["shocks_cholesky"] = shocks_cholesky
    del optim_paras["shocks_coeffs"]

    # overwrite the type information
    type_shares, type_shifts = extract_type_information(paras_vec)
    optim_paras["type_shares"] = type_shares
    optim_paras["type_shifts"] = type_shifts

    if is_debug:
        optim_paras = optimization_parameters(**optim_paras)
        assert check_model_parameters(optim_paras)

    return optim_paras._asdict()


def get_optim_paras(optim_paras, num_paras, which, is_debug):
    """Stack optimization parameters from a dictionary into a vector of type 'optim'.

    Parameters
    ----------
    optim_paras : namedtuple
        namedtuple with quantities from which the parameters can be extracted
    num_paras : int
        number of parameters in the model (not only free parameters)
    which : str
        one of ['free', 'all'], determines whether the resulting parameter vector
        contains only free parameters or all parameters.
    is_debug : bool
        If True, inputs and outputs are checked for consistency.

    """
    if is_debug:
        assert which in ["free", "all"], 'which must be in ["free", "all"]'

        # TODO: Delete if compatibility is achieved.
        if isinstance(optim_paras, dict):
            optim_paras = optimization_parameters(**optim_paras)

        assert check_model_parameters(optim_paras)

    pinfo = paras_parsing_information(num_paras)
    x = np.full(num_paras, np.nan)

    start, stop = pinfo["delta"]["start"], pinfo["delta"]["stop"]
    x[start:stop] = optim_paras.delta

    start, stop = (pinfo["coeffs_common"]["start"], pinfo["coeffs_common"]["stop"])
    x[start:stop] = optim_paras.coeffs_common

    start, stop = pinfo["coeffs_a"]["start"], pinfo["coeffs_a"]["stop"]
    x[start:stop] = optim_paras.coeffs_a

    start, stop = pinfo["coeffs_b"]["start"], pinfo["coeffs_b"]["stop"]
    x[start:stop] = optim_paras.coeffs_b

    start, stop = pinfo["coeffs_edu"]["start"], pinfo["coeffs_edu"]["stop"]
    x[start:stop] = optim_paras.coeffs_edu

    start, stop = pinfo["coeffs_home"]["start"], pinfo["coeffs_home"]["stop"]
    x[start:stop] = optim_paras.coeffs_home

    start, stop = (pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"])
    x[start:stop] = optim_paras.shocks_cholesky[np.tril_indices(4)]

    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    x[start:stop] = optim_paras.type_shares[2:]

    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    x[start:stop] = optim_paras.type_shifts.flatten()[4:]

    if is_debug:
        _check_optimization_parameters(x)

    if which == "free":
        x = [x[i] for i in range(num_paras) if not optim_paras.paras_fixed[i]]
        x = np.array(x)

    return x


def paras_parsing_information(num_paras):
    """Dictionary with the start and stop indices of each quantity.

    TODO: Maybe replace with namedtuple.

    """
    num_types = int((num_paras - 53) / 6) + 1
    num_shares = (num_types - 1) * 2
    pinfo = {
        "delta": {"start": 0, "stop": 1},
        "coeffs_common": {"start": 1, "stop": 3},
        "coeffs_a": {"start": 3, "stop": 18},
        "coeffs_b": {"start": 18, "stop": 33},
        "coeffs_edu": {"start": 33, "stop": 40},
        "coeffs_home": {"start": 40, "stop": 43},
        "shocks_coeffs": {"start": 43, "stop": 53},
        "type_shares": {"start": 53, "stop": 53 + num_shares},
        "type_shifts": {"start": 53 + num_shares, "stop": num_paras},
    }
    return pinfo


def _check_optimization_parameters(x):
    """Check optimization parameters."""
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    assert np.all(np.isfinite(x))
    return True


def get_conditional_probabilities(type_shares, edu_start):
    """Calculate the conditional choice probabilities.

    The calculation is based on the multinomial logit model for one particular initial
    condition.

    Parameters
    ----------
    type_shares : ???
    edu_start : ???

    TODO: Easy to make faster.

    """
    num_types = int(len(type_shares) / 2)
    probs = np.full(num_types, np.nan)
    for i in range(num_types):
        lower, upper = i * 2, (i + 1) * 2
        covariate = edu_start > 9
        probs[i] = np.exp(np.sum(type_shares[lower:upper] * [1.0, covariate]))

    probs = probs / sum(probs)

    return probs


def extract_type_information(x):
    """Extract the information about types from a parameter vector of type 'optim'.

    Parameters
    ----------
    x : ???

    Returns
    -------
    ???

    """
    pinfo = paras_parsing_information(len(x))

    # Type shares
    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    num_types = int(len(x[start:]) / 6) + 1
    type_shares = x[start:stop]
    type_shares = np.concatenate((np.zeros(2), type_shares), axis=0)

    # Type shifts
    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    type_shifts = x[start:stop]
    type_shifts = np.reshape(type_shifts, (num_types - 1, 4))
    type_shifts = np.concatenate((np.zeros((1, 4)), type_shifts), axis=0)

    return type_shares, type_shifts


def extract_cholesky(x, info=None):
    """Extract the cholesky factor of the shock covariance from parameters of type
    'optim.

    Parameters
    ----------
    x : ???

    """
    pinfo = paras_parsing_information(len(x))
    start, stop = (pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"])
    shocks_coeffs = x[start:stop]
    dim = number_of_triangular_elements_to_dimensio(shocks_coeffs.shape[0])
    shocks_cholesky = np.zeros((dim, dim))
    shocks_cholesky[np.tril_indices(dim)] = shocks_coeffs

    # Stabilization
    info = 0 if info is not None else info

    # We need to ensure that the diagonal elements are larger than zero during
    # estimation. However, we want to allow for the special case of total absence of
    # randomness for testing with simulated datasets.
    if not (fr.count_nonzero(shocks_cholesky) == 0):
        shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
        for i in range(len(shocks_cov)):
            if np.abs(shocks_cov[i, i]) < TINY_FLOAT:
                shocks_cholesky[i, i] = np.sqrt(TINY_FLOAT)
                if info is not None:
                    info = 1

    return shocks_cholesky, info


def coeffs_to_cholesky(coeffs):
    """Return the cholesky factor of a covariance matrix described by coeffs.

    The function can handle the case of a deterministic model (i.e. where all coeffs =
    0)

    Parameters
    ----------
    coeffs : np.ndarray
        1d numpy array that contains the upper triangular elements of a covariance
        matrix whose diagonal elements have been replaced by their square roots.

    Returns
    -------
    ???

    Example
    -------
    >>> import numpy as np
    >>> coefficients = np.array([
    ...     0.2, 0,    0, 0,    0.25,
    ...     0,   0, 1500, 0, 1500,
    ... ])
    >>> cholesky_factor = np.array([
    ...     [0.2, 0,       0,    0],
    ...     [0,   0.25,    0,    0],
    ...     [0,   0,    1500,    0],
    ...     [0,   0,       0, 1500],
    ... ])
    >>> assert np.allclose(cholesky_factor, coeffs_to_cholesky(coefficients))

    References
    ----------
    - `Cholesky decomposition (Wikipedia)
      <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_

    TODO: Use lineprofiling. Maybe np.linalg.cholesky can be refined.

    """
    dim = number_of_triangular_elements_to_dimensio(coeffs.shape[0])
    shocks = np.zeros((dim, dim))
    # Populate upper triangle of matrix
    for i, (row, col) in enumerate(zip(*fr.triu_indices(4))):
        shocks[row, col] = coeffs[i]
    # Square diagonal
    for i in range(dim):
        shocks[i, i] **= 2

    shocks_cov = shocks + shocks.T - np.diag(np.diag(shocks))

    return (
        np.zeros((dim, dim))
        if fr.count_nonzero(shocks_cov) == 0
        else np.linalg.cholesky(shocks_cov)
    )


def cholesky_to_coeffs(shocks_cholesky):
    """ Map the Cholesky factor into the coefficients from the .ini file.

    Parameters
    ----------
    shocks_cholesky : np.array
        Cholesky factor of shock covariance matrix.

    Returns
    -------
    list
        List of coefficients in the upper triangle of the shock covariance matrix.

    Example
    -------
    >>> import numpy as np
    >>> cholesky_factor = np.array([
    ...     [0.2, 0,       0,    0],
    ...     [0,   0.25,    0,    0],
    ...     [0,   0,    1500,    0],
    ...     [0,   0,       0, 1500],
    ... ])
    >>> coefficients = np.array([
    ...     0.2, 0,    0, 0,    0.25,
    ...     0,   0, 1500, 0, 1500,
    ... ])
    >>> assert np.allclose(coefficients, cholesky_to_coeffs(cholesky_factor))

    TODO: Why does it have to be a list?

    """
    # Recover the covariance matrix from the Cholesky factor.
    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
    for i in range(shocks_cov.shape[0]):
        shocks_cov[i, i] **= 0.5

    # Extract values of upper triangle
    shocks_coeffs = np.full(fr.tri_n_with_diag(shocks_cov.shape[0]), np.nan)
    for i, (row, col) in enumerate(zip(*fr.triu_indices(shocks_cov.shape[0]))):
        shocks_coeffs[i] = shocks_cov[row, col]

    return list(shocks_coeffs)


def get_total_values(
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
):
    """Get value function of all possible states.

    This is called total value because it is the sum of immediate rewards, including
    realized shocks, and expected future rewards.

    Parameters
    ----------
    period : int
        Number of period
    num_periods : int
        Total number of periods
    optim_paras : dict
        Dictionary containing shocks, types, coefficients, etc..
    rewards_systematic : np.array
        Four-dimensional array
    draws : np.array
        Four-dimensional array
    edu_spec : dict
        Dictionary with lagged, start, share, max
    mapping_state_idx : np.array
        Six-dimensional array
    periods_emax : np.array
        Array with shape (num_periods, ???num_individuals)
    k : int
        ???
    states_all : np.array
        Array with shape (num_periods, ???num_individuals, ???num_states)

    Returns
    -------
    total_values : ???
    rewards_ex_post : ???

    """
    # We need to back out the wages from the total systematic rewards to working in the
    # labor market to add the shock properly.
    exp_a, exp_b, edu, choice_lagged, type_ = states_all[period, k, :]
    wages_systematic = back_out_systematic_wages(
        rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras
    )

    rewards_ex_post = np.full(4, np.nan)

    # Calculate ex post rewards
    total_increment = rewards_systematic[:2] - wages_systematic[:2]
    rewards_ex_post[:2] = wages_systematic[:2] * draws[:2] + total_increment

    # TODO(tobiasraabe): I do not understand what is going on here. Is it + or *?
    rewards_ex_post[2:4] = rewards_systematic[2:4] + draws[2:4]

    # Get future values
    if period != (num_periods - 1):
        emaxs = get_emaxs(
            edu_spec.max, mapping_state_idx, period, periods_emax, k, states_all
        )
    else:
        emaxs = np.zeros(4)

    total_values = rewards_ex_post + optim_paras.delta * emaxs

    # This is required to ensure that the agent does not choose any inadmissible states.
    # If the state is inadmissible, emaxs takes value zero.
    total_values[2] += (
        INADMISSIBILITY_PENALTY if states_all[period, k, 2] >= edu_spec.max else 0
    )

    return total_values, rewards_ex_post


def get_emaxs(edu_spec, mapping_state_idx, period, periods_emax, k, states_all):
    """Get emaxs for additional choices.

    Parameters
    ----------
    edu_spec
    mapping_state_idx
    period
    periods_emax
    k
    states_all

    Returns
    -------
    emaxs : ???

    TODO: Write test.

    """
    # Distribute state space
    exp_a, exp_b, edu, _, type_ = states_all[period, k, :]

    # Future utilities
    emaxs = np.full(4, np.nan)

    # Working in Occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 1 - 1, type_]
    emaxs[0] = periods_emax[period + 1, future_idx]

    # Working in Occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 2 - 1, type_]
    emaxs[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year of schooling is only
    # possible for those that have strictly less than the maximum level of additional
    # education allowed.
    is_inadmissible = edu >= edu_spec.max
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 3 - 1, type_]

    emaxs[2] = 0.0 if is_inadmissible else periods_emax[period + 1, future_idx]

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 4 - 1, type_]
    emaxs[3] = periods_emax[period + 1, future_idx]

    return emaxs


def create_draws(num_periods, num_draws, seed, is_debug):
    """Create draws from a standard multivariate normal distribution.

    Handle special case of zero variances as this case is useful for testing. The draws
    are from a standard normal distribution and transformed later in the code.

    Parameters
    ----------
    num_periods : int
        Number of periods.
    num_draws : int
        Number of draws.
    seed : int
        Seed for randomness.
    is_debug: bool
        Flag for debugging.

    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution or read it from disk. The
    # latter is available to allow for testing across implementations.
    if is_debug and os.path.exists(".draws.respy.test"):
        draws = read_draws(num_periods, num_draws)
    else:
        draws = np.random.multivariate_normal(
            np.zeros(4), np.identity(4), (num_periods, num_draws)
        )

    return draws


def add_solution(
    respy_obj,
    periods_rewards_systematic,
    states_number_period,
    mapping_state_idx,
    periods_emax,
    states_all,
):
    """Add solution to class instance."""
    respy_obj.unlock()
    respy_obj.set_attr("periods_rewards_systematic", periods_rewards_systematic)
    respy_obj.set_attr("states_number_period", states_number_period)
    respy_obj.set_attr("mapping_state_idx", mapping_state_idx)
    respy_obj.set_attr("periods_emax", periods_emax)
    respy_obj.set_attr("states_all", states_all)
    respy_obj.set_attr("is_solved", True)
    respy_obj.lock()
    return respy_obj


def replace_missing_values(arguments):
    """Replace MISSING_FLOAT with NAN.

    Note that the output argument is of type float in the case missing values
    are found.

    """
    # Antibugging
    assert isinstance(arguments, tuple) or isinstance(arguments, np.ndarray)

    if isinstance(arguments, np.ndarray):
        arguments = (arguments,)

    rslt = tuple()

    for argument in arguments:
        # Transform to float array to evaluate missing values.
        argument_internal = np.asfarray(argument)

        # Determine missing values
        is_missing = argument_internal == MISSING_FLOAT
        if np.any(is_missing):
            # Replace missing values
            argument = np.asfarray(argument)
            argument[is_missing] = np.nan

        rslt += (argument,)

    # Aligning interface.
    if len(rslt) == 1:
        rslt = rslt[0]

    return rslt


def check_model_parameters(optim_paras):
    """Check the integrity of all model parameters."""
    # Auxiliary objects
    num_types = optim_paras.type_shifts.shape[0]

    # Checks for all arguments
    for field in optim_paras._fields:
        if field in ["paras_fixed", "paras_bounds"]:
            continue

        assert isinstance(getattr(optim_paras, field), np.ndarray), field
        assert np.all(np.isfinite(getattr(optim_paras, field)))
        assert getattr(optim_paras, field).dtype == "float"
        assert np.all(abs(getattr(optim_paras, field)) < PRINT_FLOAT)

    # Check for discount rate
    assert optim_paras.delta >= 0

    # Checks for common returns
    assert optim_paras.coeffs_common.size == 2

    # Checks for occupations
    assert optim_paras.coeffs_a.size == 15
    assert optim_paras.coeffs_b.size == 15
    assert optim_paras.coeffs_edu.size == 7
    assert optim_paras.coeffs_home.size == 3

    # Checks shock matrix
    assert optim_paras.shocks_cholesky.shape == (4, 4)
    np.allclose(optim_paras.shocks_cholesky, np.tril(optim_paras.shocks_cholesky))

    # Checks for type shares
    assert optim_paras.type_shares.size == num_types * 2

    # Checks for type shifts
    assert optim_paras.type_shifts.shape == (num_types, 4)

    return True


def dist_class_attributes(respy_obj, *args):
    """Distribute class class attributes.

    Parameters
    ----------
    respy_obj : clsRespy
    args :
        Any number of strings that are keys in :meth:`clsRespy.attr`.

    Returns
    -------
    list
        List of values from :meth:`clsRespy.attr` or single value from
        :meth:`clsRespy.attr`

    """
    ret = [respy_obj.get_attr(arg) for arg in args]
    if len(ret) == 1:
        ret = ret[0]

    return ret


def read_draws(num_periods, num_draws):
    """Read the draws from disk.

    This is only used in the development process.

    """
    # Initialize containers
    periods_draws = np.full((num_periods, num_draws, 4), np.nan)

    # Read and distribute draws
    draws = np.array(np.genfromtxt(".draws.respy.test"), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws * period
        upper = lower + num_draws
        periods_draws[period, :, :] = draws[lower:upper, :]

    return periods_draws


def transform_disturbances(draws, shocks_mean, shocks_cholesky):
    """Transform the standard normal deviates to the relevant distribution.

    Parameters
    ----------
    draws : np.array
        Array with shape (num_draws, num_shocks)
    shocks_mean : np.array
        Array with shape (num_shocks,)
    shocks_cholesky : np.array
        Array with shape (num_shocks, num_shocks)

    """
    draws_transformed = draws.dot(shocks_cholesky.T)

    draws_transformed[:, :4] = draws_transformed[:, :4] + shocks_mean[:4]

    draws_transformed[:, :2] = fr.clip(
        np.exp(draws_transformed[:, :2]), 0.0, HUGE_FLOAT
    )

    return draws_transformed


def format_opt_parameters(dict_, pos):
    """Format the values depending on whether they are fixed or estimated."""
    # Initialize baseline line
    val = dict_["coeffs"][pos]
    is_fixed = dict_["fixed"][pos]
    bounds = dict_["bounds"][pos]

    line = ["coeff", val, " ", " "]
    if is_fixed:
        line[-2] = "!"

    # Check if any bounds defined
    if any(x is not None for x in bounds):
        line[-1] = "(" + str(bounds[0]) + "," + str(bounds[1]) + ")"

    return line


def apply_scaling(x, precond_matrix, request):
    """Apply or revert the preconditioning step."""
    if request == "do":
        out = np.dot(precond_matrix, x)
    elif request == "undo":
        out = np.dot(np.linalg.pinv(precond_matrix), x)
    else:
        raise AssertionError

    return out


def get_est_info():
    """Read the parameters from the last step of a previous estimation run."""

    def _process_value(input_, type_):
        try:
            if type_ == "float":
                value = float(input_)
            elif type_ == "int":
                value = int(input_)
        except ValueError:
            value = "---"

        return value

    # We need to make sure that the updating file actually exists.
    if not os.path.exists("est.respy.info"):
        msg = "Parameter update impossible as "
        msg += "file est.respy.info does not exist"
        raise UserError(msg)

    # Initialize container and ensure a fresh start processing the file
    linecache.clearcache()
    rslt = dict()

    # Value of the criterion function
    line = shlex.split(linecache.getline("est.respy.info", 6))
    for key_ in ["start", "step", "current"]:
        rslt["value_" + key_] = _process_value(line.pop(0), "float")

    # Total number of evaluations and steps
    line = shlex.split(linecache.getline("est.respy.info", 49))
    rslt["num_step"] = _process_value(line[3], "int")

    line = shlex.split(linecache.getline("est.respy.info", 51))
    rslt["num_eval"] = _process_value(line[3], "int")

    # Parameter values
    for i, key_ in enumerate(["start", "step", "current"]):
        rslt["paras_" + key_] = []
        for j in range(13, 99):
            line = shlex.split(linecache.getline("est.respy.info", j))
            if not line:
                break
            rslt["paras_" + key_] += [_process_value(line[i + 1], "float")]
        rslt["paras_" + key_] = np.array(rslt["paras_" + key_])

    return rslt


def remove_scratch(fname):
    """Remove scratch files."""
    if os.path.exists(fname):
        os.unlink(fname)


def check_early_termination(maxfun, num_eval):
    """Check for reasons that require early termination of the optimization.

    We want early termination if the number of function evaluations is already
    at maxfun. This is not strictly enforced in some of the SCIPY algorithms.

    The user can also stop the optimization immediate, but gently by putting
    a file called '.stop.respy.scratch' in the working directory.

    """
    if maxfun == num_eval:
        raise MaxfunError

    if os.path.exists(".stop.respy.scratch"):
        raise MaxfunError


def get_num_obs_agent(data_array, num_agents_est):
    """Get a list with the number of observations for each agent."""
    num_obs_agent = np.zeros(num_agents_est)
    agent_number = data_array[0, 0]
    num_rows = data_array.shape[0]

    q = 0
    for i in range(num_rows):
        # We need to check whether we are faced with a new agent.
        if data_array[i, 0] != agent_number:
            q += 1
            agent_number = data_array[i, 0]

        num_obs_agent[q] += 1

    return num_obs_agent


def back_out_systematic_wages(
    rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras
):
    """Construct the wage component for the labor market rewards.

    Parameters
    ----------
    rewards_systematic : np.array
        Array with shape (num_rewards).
    exp_a : int
        Years of experience in occupation A
    exp_b : int
        Years of experience in occupation B
    edu : int
        Years of education.
    choice_lagged : int
        ???
    optim_paras : dict
        ???

    TODO: Cannot be jitted until optim_paras is no dict anymore.

    Example
    -------
    >>> import numpy as np
    >>> rewards_systematic = np.array([14617.86953434, 9701.15277293, -4000, 17750])
    >>> optim_paras = {
    ...     "coeffs_a": np.array([
    ...         9.21e+00,  3.80e-02,  3.30e-02, -5.00e-04,  0.00e+00, -0.00e+00,
    ...         0.00e+00, -0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
    ...         0.00e+00,  0.00e+00,  0.00e+00
    ...     ]),
    ...     "coeffs_b": np.array([
    ...         8.48e+00,  7.00e-02,  2.20e-02, -5.00e-04, 6.70e-02,  -1.00e-03,
    ...         0.00e+00, -0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
    ...         0.00e+00,  0.00e+00,  0.00e+00
    ...     ]),
    ...     "coeffs_common": np.array([0., 0.])
    ... }
    >>> wages_systematic = back_out_systematic_wages(
    ...     rewards_systematic, 0, 0, 10, 4, optim_paras
    ... )
    >>> assert np.allclose(wages_systematic, np.array([14617.86953434,  9701.15277293]))

    """
    # Construct covariates needed for the general part of labor market rewards.
    covariates = construct_covariates(exp_a, exp_b, edu, choice_lagged, np.nan, np.nan)

    # First we calculate the general component.
    general, wages_systematic = (np.full(2, np.nan), np.full(2, np.nan))

    covars_general = [1.0, covariates.not_exp_a_lagged, covariates.not_any_exp_a]
    general[0] = optim_paras.coeffs_a[12:].dot(covars_general)

    covars_general = [1.0, covariates.not_exp_b_lagged, covariates.not_any_exp_b]
    general[1] = optim_paras.coeffs_b[12:].dot(covars_general)

    # Second we do the same with the common component.
    covars_common = [covariates.hs_graduate, covariates.co_graduate]
    rewards_common = optim_paras.coeffs_common.dot(covars_common)
    wages_systematic = rewards_systematic[:2] - general - rewards_common

    return wages_systematic


def construct_covariates(exp_a, exp_b, edu, choice_lagged, type_, period):
    """ Construction of some additional covariates for the reward calculations.

    The problem is that namedtuples cannot be created in jitted functions. This helper
    function generates the data.

    There are two cases where the function receives np.nan as inputs.

    1. In :func:`back_out_systematic_wages` type_ and period are set to np.nan.
    2. TODO: Find when edu is set to nan.

    Notes
    -----
    - Generating the covariates is in the top three of computational costly functions
    - The jitted function is four times faster than its Python version.
    - Creating the namedtuple is costly as the jitted version of generating data +
      creating the namedtuple is as fast as only generating data with the Python
      function. Maybe jitclasses are more efficient, but they do not support slicing.
    - Use only array? Supports slicing but makes lookup harder.
    - np.nans are returned in contrast to None.

    Parameters
    ----------
    exp_a : int
        Years of experience in occupation A.
    exp_b : int
        Years of experience in occupation B.
    edu : int
        Years of education.
    choice_lagged : int
        ???
    type_ : int
        ???
    period : int
        Number of period.

    Returns
    -------
    dict
        Dictionary with covariates

    Example
    -------
    >>> covariates = construct_covariates(0, 0, 10, 4, 0, 39)
    >>> assert covariates == Covariates(
    ...     not_exp_a_lagged=0, not_exp_b_lagged=0, work_a_lagged=0, work_b_lagged=0,
    ...     edu_lagged=0, choice_lagged=4, not_any_exp_a=1, not_any_exp_b=1,
    ...     any_exp_a=0, any_exp_b=0, period=39, exp_a=0, exp_b=0, type=0, edu=10,
    ...     hs_graduate=0, co_graduate=0, is_return_not_high_school=1,
    ...     is_return_high_school=0, is_minor=0, is_young_adult=0, is_adult=1
    ... )

    TODO: DISCUSSION_ON_WEDNESDAY
    TODO: What about the case where edu is None? Was implemented but when does it
    happen?

    """
    return Covariates(
        int((exp_a > 0) and (choice_lagged != 1)),
        int((exp_b > 0) and (choice_lagged != 2)),
        int(choice_lagged == 1),
        int(choice_lagged == 2),
        int(choice_lagged == 3),
        int(choice_lagged),
        int(exp_a == 0),
        int(exp_b == 0),
        int(exp_a > 0),
        int(exp_b > 0),
        int(period) if not np.isnan(period) else np.nan,
        int(exp_a),
        int(exp_b),
        int(type_) if not np.isnan(type_) else np.nan,
        int(edu),
        # High school and/or college graduate
        int(edu >= 12),
        int(edu >= 16),
        # Return or not high school
        int((not choice_lagged == 3) and (not edu >= 12)),
        int((not choice_lagged == 3) and edu >= 12),
        # Age group: minor, young adult or adult
        int(period < 2) if not np.isnan(period) else np.nan,
        int(period in [2, 3, 4]) if not np.isnan(period) else np.nan,
        int(period >= 5) if not np.isnan(period) else np.nan,
    )


def calculate_rewards_common(covariates, coeffs_common):
    """ Calculate the reward component that is common to all alternatives.

    Parameters
    ----------
    covariates : namedtuple
        Contains covariates of individual.
    coeffs_common : dict
        ???

    """
    # Cast to float
    covars_common = np.array([covariates.hs_graduate, covariates.co_graduate]) * 1.0
    rewards_common = coeffs_common.dot(covars_common)

    return rewards_common


def calculate_rewards_general(covariates, optim_paras):
    """ Calculate the non-skill related reward components.

    Parameters
    ----------
    covariates : namedtuple
        Contains covariates of individual.
    optim_paras : dict
        ???

    Returns
    -------
    np.array
        Returns general rewards.

    TODO: Cannot be jitted until optim_paras is no dict.

    """
    rewards_general = np.full(2, np.nan)
    covars_general = np.array(
        [1.0, covariates.not_exp_a_lagged, covariates.not_any_exp_a]
    )
    rewards_general[0] = optim_paras.coeffs_a[12:].dot(covars_general)

    covars_general = np.array(
        [1.0, covariates.not_exp_b_lagged, covariates.not_any_exp_b]
    )
    rewards_general[1] = optim_paras.coeffs_b[12:].dot(covars_general)

    return rewards_general


def get_valid_bounds(which, value):
    """ Simply get a valid set of bounds.
    """
    assert which in ["cov", "coeff", "delta", "share"]

    # The bounds cannot be too tight as otherwise the BOBYQA might not start
    # properly.
    if which in ["delta"]:
        upper = np.random.choice([None, value + np.random.uniform(low=0.1)])
        bounds = [max(0.0, value - np.random.uniform(low=0.1)), upper]
    elif which in ["coeff"]:
        upper = np.random.choice([None, value + np.random.uniform(low=0.1)])
        lower = np.random.choice([None, value - np.random.uniform(low=0.1)])
        bounds = [lower, upper]
    elif which in ["cov"]:
        bounds = [None, None]
    elif which in ["share"]:
        bounds = [0.0, None]
    return bounds


def number_of_triangular_elements_to_dimensio(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Parameters
    ----------
    num : int
        The number of upper or lower triangular elements in the matrix.

    Returns
    -------
    int
        Number of dimensions.

    # TODO: Replace with one of the fast routines.

    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
