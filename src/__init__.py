"""Main survey file.
"""
import os
import pytz
from datetime import datetime, timedelta
from random import shuffle

import numpy as np
import pandas as pd
from sqlalchemy import DATE
import yfinance as yf
from flask_login import current_user
from hemlock import User, Page, utils
from hemlock.app import db
from hemlock.functional import compile, validate, test_response
from hemlock.questions import Check, Input, Label, Range, Select, Textarea
from hemlock_ax import Assigner
from hemlock_ax.assign import get_data as get_data_base
from sqlalchemy_mutable.utils import partial

DATE_FORMAT = "%Y-%m-%d"

def get_crude_oil_price():
    df = yf.Ticker("CL=F").history("5d").reset_index().iloc[-2]
    return df.Date.strftime(DATE_FORMAT), df.Close

def get_s_and_p():
    df = yf.Ticker("^GSPC").history("5d").reset_index().iloc[-2]
    return df.Date.strftime(DATE_FORMAT), df.Close

forecast_questions = [
    (
        "crude",
        """
        What do you think the price of Crude Oil will be at the end of the day today?
        
        You can find a graph of recent Crude Oil prices
        <a href="https://finance.yahoo.com/chart/CL%3DF" target="_blank">here</a>.
        """,
        get_crude_oil_price
    ),
    (
        "s_and_p",
        """
        What do you think the S&P 500 index will close at today?

        You can find a graph of recent values of the S&P 500 index
        <a href="https://finance.yahoo.com/chart/%5EGSPC" target="_blank">here</a>.
        """,
        get_s_and_p
    )
]

def get_data(assigner):
    # update outcomes and targets from the previous day's outcomes
    dates_outcomes = [q[2]() for q in forecast_questions]
    users = User.query.filter(User.start_time>=datetime.utcnow()-timedelta(days=5)).all()
    for user in users:
        df = user.get_data()
        if "forecast" not in df:
            continue

        updated_outcomes = False
        for i, (date, outcome) in enumerate(dates_outcomes):
            if df.outcome[i] is None and df.start_time_et[i] == date:
                _, outcome_data, target_data = user.data
                outcome_data.data[i] = outcome
                target_data.data[i] = -(df.forecast[i] - outcome) ** 2
                updated_outcomes = True

        if updated_outcomes:
            user.cache_data()

    db.session.commit()

    # get dataframe and standardize targets by question and day
    df = get_data_base(assigner)
    if "target" not in df:
        return pd.DataFrame()

    gb = df.groupby(["question", "start_time_et"]).target
    df["target"] = (df.target - gb.transform("mean")) / gb.transform("std")
    return df

assigner = Assigner({"treatment": (0, 1, 2)}, get_data=get_data)


@User.route("/survey")
def seed():
    """Creates the main survey branch.

    Returns:
        List[Page]: List of pages shown to the user.
    """
    
    if os.getenv("FLASK_ENV") == "production":
        start = current_user.start_time
        current_user.meta_data["start_time_et"] = (
            datetime(start.year, start.month, start.day, tzinfo=pytz.utc)
            .astimezone(pytz.timezone("US/Eastern"))
            .strftime(DATE_FORMAT)
        )
    else:
        current_user.meta_data["start_time_et"] = forecast_questions[0][2]()[0]

    assignment = assigner.assign_user()
    current_user.data = [
        ("question", [q[0] for q in forecast_questions]),
        ("outcome", len(forecast_questions) * [None]),
        ("target", len(forecast_questions) * [None])
    ]
    forecast_pages = [
        Page(
            Label(
                f"You were assigned to {assignment}."
            ),
            Input(
                q[1],
                input_tag={"type": "number"},
                variable="forecast"
            )
        )
        for q in forecast_questions
    ]
    return forecast_pages + [
        Page(
            Label("Thanks for participating!"),
        )
    ]
