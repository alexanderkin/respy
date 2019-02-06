from collections import namedtuple


education_specification = namedtuple(
    "education_specification", ["lagged", "start", "share", "max"]
)


optimization_parameters = namedtuple(
    "OptimParas",
    [
        "shocks_cholesky",
        "type_shifts",
        "type_shares",
        "coeffs_a",
        "coeffs_b",
        "coeffs_common",
        "coeffs_edu",
        "coeffs_home",
        "delta",
        "paras_fixed",
        "paras_bounds",
    ],
)
"""
namedtuple
    Wrapper for the optim_paras as dicts cannot be used as arguments for jitted
    functions.

"""


Covariates = namedtuple(
    "Covariates",
    [
        "not_exp_a_lagged",
        "not_exp_b_lagged",
        "work_a_lagged",
        "work_b_lagged",
        "edu_lagged",
        "choice_lagged",
        "not_any_exp_a",
        "not_any_exp_b",
        "any_exp_a",
        "any_exp_b",
        "period",
        "exp_a",
        "exp_b",
        "type",
        "edu",
        "hs_graduate",
        "co_graduate",
        "is_return_not_high_school",
        "is_return_high_school",
        "is_minor",
        "is_young_adult",
        "is_adult",
    ],
)
"""namedtuple: Container for covariates.

The former approach was to use a dictionary which lacks the support from Numba. Also,
namedtuples are not malleable and support slicing.

# TODO: Refactor if more datatypes exist. E.g. optim_paras.

"""
