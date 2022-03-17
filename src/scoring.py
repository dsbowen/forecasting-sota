import numpy as np
from scipy.stats import rv_continuous

from src.utils import OUTCOME_VARIABLE


def crps(row, convert_to_distribution):
    try:
        dist = convert_to_distribution(row)
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99))
        return ((dist.cdf(x) - (x > row[OUTCOME_VARIABLE])) ** 2).sum() * (x[1] - x[0])
    except ValueError:
        return None


def convert_to_nonparametric_elicitation(row):
    cdf, values = [], []
    for col in row.index:
        if col.startswith("ppf"):
            cdf.append(float(col.lstrip("ppf_")))
            values.append(row[col])

    return nonparametric_elecitation(cdf, values)


class nonparametric_elecitation(rv_continuous):
    def __init__(self, cdf, values, *args, **kwargs):
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
