from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_log_likl
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic


def pyth_criterion(
    x,
    is_interpolated,
    num_draws_emax,
    num_periods,
    num_points_interp,
    is_myopic,
    is_debug,
    data_array,
    num_draws_prob,
    tau,
    periods_draws_emax,
    periods_draws_prob,
    states_all,
    states_number_period,
    mapping_state_idx,
    max_states_period,
    num_agents_est,
    num_obs_agent,
    num_types,
    edu_spec,
):
    """Criterion function for the likelihood maximization."""
    optim_paras = distribute_parameters(x, is_debug)

    # Calculate all systematic rewards
    periods_rewards_systematic = pyth_calculate_rewards_systematic(
        num_periods, states_number_period, states_all, max_states_period, optim_paras
    )

    periods_emax = pyth_backward_induction(
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
        file_sim="",
        is_write=False,
    )

    contribs = pyth_contributions(
        periods_rewards_systematic,
        mapping_state_idx,
        periods_emax,
        states_all,
        data_array,
        periods_draws_prob,
        tau,
        num_periods,
        num_draws_prob,
        num_agents_est,
        num_obs_agent,
        num_types,
        edu_spec,
        optim_paras,
    )

    crit_val = get_log_likl(contribs)

    return crit_val
