"""Functions for the adaptive assigner
"""
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hemlock import User
from hemlock.app import db
from hemlock_ax.assign import Assigner, get_data as get_data_base
from hemlock_ax.models import ModelReturnType
from scipy.stats import multivariate_normal

from .questions import questions
from .scoring import crps
from .treatments import CONTROL_ARM, treatments
from .utils import (
    OUTCOME_VARIABLE,
    QUESTION_VARIABLE,
    START_TIME_VARIABLE,
    TARGET,
    TREATMENT_VARIABLE,
)

CUTOFF_STD = -3


def get_data(assigner: Assigner) -> pd.DataFrame:
    """Get users' data.

    Args:
        assigner (Assigner): Adaptive assigner.

    Returns:
        pd.DataFrame: Users' data.
    """
    # update outcomes and targets from the previous day's outcomes
    def get_outcome_df(question):
        df = question["get_outcome"]()
        df[QUESTION_VARIABLE] = question["name"]
        return df

    outcome_df = pd.concat([get_outcome_df(q) for q in questions])
    users = User.query.filter(User.has_target != True).all()
    for user in users:
        df = user.get_data()
        if OUTCOME_VARIABLE in df and df[OUTCOME_VARIABLE].isna().any():
            df = df.drop(columns=OUTCOME_VARIABLE).merge(
                outcome_df, on=[START_TIME_VARIABLE, QUESTION_VARIABLE], how="left"
            )
            if df[OUTCOME_VARIABLE].count() > 0:
                for data in user.data:
                    if data.variable == OUTCOME_VARIABLE:
                        outcome_data = data
                    if data.variable == TARGET:
                        target_data = data

                outcome_data.data = list(df[OUTCOME_VARIABLE].replace(np.nan, None))
                treatment = treatments[user.get_meta_data()[TREATMENT_VARIABLE]]
                loss = partial(
                    crps,
                    convert_to_distribution=treatment["convert_to_distribution"],
                )
                target_data.data = list(df.apply(loss, axis=1))
                user.cache_data()
                if not df[OUTCOME_VARIABLE].isna().any():
                    user.has_target = True

    db.session.commit()

    # get the data
    df = get_data_base(assigner)
    if TARGET not in df:
        return pd.DataFrame()

    # if too few users have been assigned to the control condition
    # standardization will be impossible
    if not enough_control_users(df):
        return df

    df = standardize_target(df)
    # exclude users with exceptionally poor scores
    # likely weren't paying attention or didn't understand what to do
    df = df[df[TARGET] > CUTOFF_STD]
    if not enough_control_users(df):
        return df

    return standardize_target(df)


def enough_control_users(df: pd.DataFrame, min_control_users: int = 2) -> bool:
    """Indicates that there are enough users in the control condition.

    Args:
        df (pd.DataFrame): User's data.
        min_control_users (int, optional): Minimum number of users that need to be in
            the control condition for the model to fit. Defaults to 2.

    Returns:
        bool: Indicator.
    """
    return (
        df[df[TREATMENT_VARIABLE] == CONTROL_ARM].groupby("question").target.count()
        < min_control_users
    ).any()


def standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes the target variable.

    For each day and question, standardize the target using the control group mean and
    standard error.

    Args:
        df (pd.DataFrame): User data with unstandardized target.

    Returns:
        pd.DataFrame: User data with standardized target.
    """
    gb_variables = [START_TIME_VARIABLE, QUESTION_VARIABLE]
    gb = df[df[TREATMENT_VARIABLE] == CONTROL_ARM].groupby(gb_variables)[TARGET]
    control_df = gb.mean().reset_index().rename(columns={TARGET: "control_mean"})
    control_df["control_std"] = gb.std().values
    df = df.merge(control_df, on=gb_variables)
    df[TARGET] = (df[TARGET] - df.control_mean) / df.control_std
    return df.drop(columns=["control_mean", "control_std"])


def fixed_effects_regression(df: pd.DataFrame, **kwargs: Any) -> ModelReturnType:
    """Run a fixed effects regression.

    Fixed effects are question x date. Cluster standard errors by date.

    Args:
        df (pd.DataFrame): User data.

    Returns:
        ModelReturnType: List of possible assignments, distribution of treatment
            effects.
    """
    df = df.dropna(subset=[TREATMENT_VARIABLE, TARGET])
    df["fixed_effects"] = df[[START_TIME_VARIABLE, QUESTION_VARIABLE]].apply(
        lambda x: tuple(x), axis=1
    )
    X = pd.get_dummies(df[[TREATMENT_VARIABLE, "fixed_effects"]]).drop(
        columns=f"{TREATMENT_VARIABLE}_{CONTROL_ARM}"
    )
    model = sm.OLS(df.target, X)
    results = model.fit().get_robustcov_results(
        "cluster", groups=df[START_TIME_VARIABLE]
    )
    mask = [col.startswith(TREATMENT_VARIABLE) for col in model.exog_names]
    exog_names = [
        (col[len(f"{TREATMENT_VARIABLE}_") :],)
        for col in np.array(model.exog_names)[mask]
    ]
    dist = multivariate_normal(
        results.params[mask], results.cov_params()[mask][:, mask], allow_singular=True
    )
    return exog_names, dist
