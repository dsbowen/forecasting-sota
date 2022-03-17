import pytz
from datetime import datetime

import yfinance as yf

from .utils import DATE_FORMAT, START_TIME_VARIABLE, OUTCOME_VARIABLE


def get_today():
    now = datetime.utcnow()
    return (
        datetime(now.year, now.month, now.day, now.hour, tzinfo=pytz.utc)
        .astimezone(pytz.timezone("US/Eastern"))
        .strftime(DATE_FORMAT)
    )


def get_crude_oil_price():
    df = yf.Ticker("CL=F").history("5d").reset_index()
    df[START_TIME_VARIABLE] = df.Date.astype(str)
    return df[df.start_time_et != get_today()].rename(
        columns={"Close": OUTCOME_VARIABLE}
    )


def get_s_and_p():
    df = yf.Ticker("^GSPC").history("5d").reset_index()
    df[START_TIME_VARIABLE] = df.Date.astype(str)
    return df[df.start_time_et != get_today()].rename(
        columns={"Close": OUTCOME_VARIABLE}
    )


questions = [
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
