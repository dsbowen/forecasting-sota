"""Questions to forecast.
"""
from __future__ import annotations

import pytz
from datetime import datetime
from typing import Callable, Dict, Union

import pandas as pd
import yfinance as yf

from .utils import DATE_FORMAT, START_TIME_VARIABLE, OUTCOME_VARIABLE

QuestionType = Dict[str, Union[str, Callable[[], pd.DataFrame]]]


def get_today() -> str:
    """Get today's date in YYYY-MM-DD format.

    Returns:
        str: Today's date in Easter Standard time.
    """
    now = datetime.utcnow()
    return (
        datetime(now.year, now.month, now.day, now.hour, tzinfo=pytz.utc)
        .astimezone(pytz.timezone("US/Eastern"))
        .strftime(DATE_FORMAT)
    )


def get_crude_oil_price() -> pd.DatatFrame:
    """Get the crude oil prices for the last 5 days.

    Returns:
        pd.DatatFrame: Dataframe with dates and closing prices for crude oil.
    """
    df = yf.Ticker("CL=F").history("5d").reset_index()
    df[START_TIME_VARIABLE] = df.Date.astype(str)
    return df[df[START_TIME_VARIABLE] != get_today()].rename(
        columns={"Close": OUTCOME_VARIABLE}
    )


def get_s_and_p() -> pd.DataFrame:
    """Get the S&P 500 index value for the last 5 days.

    Returns:
        pd.DataFrame: Dataframe with dates and closing values for the S&P 500.
    """
    df = yf.Ticker("^GSPC").history("5d").reset_index()
    df[START_TIME_VARIABLE] = df.Date.astype(str)
    return df[df[START_TIME_VARIABLE] != get_today()].rename(
        columns={"Close": OUTCOME_VARIABLE}
    )


# each question is a dictionary with the keys "name", "url", "get_outcome".
# name is the question name.
# url is a URL with more information about the forecasted variable.
# get_outcome is a function that returns a dataframe with variable outcomes. It must
# contain a column for the date (as a string) and the outcome (as a float).
questions: list[QuestionType] = [
    {
        "name": "crude",
        "url": "https://finance.yahoo.com/chart/CL%3DF",
        "get_outcome": get_crude_oil_price,
    },
    {
        "name": "s_and_p",
        "url": "https://finance.yahoo.com/chart/%5EGSPC",
        "get_outcome": get_s_and_p,
    },
]
