"""Defines scoring rules and convenience functions.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous

from src.utils import OUTCOME_VARIABLE


def crps(
    row: pd.Series, convert_to_distribution: Callable[[pd.Series], rv_continuous]
) -> Optional[float]:
    """Continuous ranked probability score.

    Technically, the negative CRPS so the higher (closer to 0) is better.

    Args:
        row (pd.Series): Row of a user's dataframe corresponding to one forecast.
        convert_to_distribution (Callable[[pd.Series], rv_continuous]): Function that
            takes the row and returns the forecast distribution.

    Returns:
        Optional[float]: Score.
    """
    try:
        dist = convert_to_distribution(row)
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99))
        return -((dist.cdf(x) - (x > row[OUTCOME_VARIABLE])) ** 2).sum() * (
            x[-1] - x[0]
        )
    except ValueError:
        return None


def convert_to_nonparametric_elicitation(row: pd.Series) -> nonparametric_elecitation:
    """Converts forecasts to distribution based on nonparametric elicitation.

    Args:
        row (pd.Series): A row of a user's data with their forecasts.

    Returns:
        nonparametric_elecitation: Maximum entropy distribtuion from nonparametric
            elicitation.
    """
    cdf, values = [], []
    for col in row.index:
        if col.startswith("ppf"):
            cdf.append(float(col.lstrip("ppf_")))
            values.append(row[col])

    return nonparametric_elecitation(cdf, values)


class nonparametric_elecitation(rv_continuous):
    def __init__(
        self, cdf: list[float], values: list[float], *args: Any, **kwargs: Any
    ):
        """Maximum entropy distribution based on nonparametric elicitation.

        Args:
            cdf (list[float]): Value of the CDF (must be between 0 and 1).
            values (list[float]): Values at which the CDF is evaluated.

        Raises:
            ValueError: If the values are not weakly increasing with the CDF.
        """
        super().__init__(*args, **kwargs)
        argsort = np.array(cdf).argsort()
        cdf, self.values = np.array(cdf)[argsort], np.array(values)[argsort]
        if (np.diff(self.values) < 0).any():
            raise ValueError(
                f"Values must be weakly increasing with the cdf, got {values}."
            )
        pdf = np.insert(np.diff(cdf) / np.diff(self.values), 0, 0)
        self.pdf_values = np.append(pdf, 0)
        self.cdf_values = np.insert(cdf, 0, 0)

    def _get_index(self, x):
        return (
            self.values < np.repeat(np.atleast_2d(x), len(self.values), axis=0).T
        ).sum(axis=1)

    def _pdf(self, x):
        return self.pdf_values[self._get_index(x)]

    def _cdf(self, x):
        index = self._get_index(x)
        return self.cdf_values[index] + self.pdf_values[index] * np.clip(
            (x - self.values[index - 1]), 0, np.inf
        )
