"""Main survey file.
"""
from __future__ import annotations

import os
from random import shuffle

from flask_login import current_user
from hemlock import User, Page, utils
from hemlock.app import db
from hemlock.questions import Check, Input, Label, Range, Select, Textarea
from hemlock_ax import Assigner

from .assigner_functions import get_data, fixed_effects_regression
from .questions import get_today, questions
from .treatments import CONTROL_ARM, treatments
from .utils import (
    OUTCOME_VARIABLE,
    QUESTION_VARIABLE,
    START_TIME_VARIABLE,
    TARGET,
    TREATMENT_VARIABLE,
)

User.has_target = db.Column(db.Boolean, default=False)

assigner = Assigner(
    {TREATMENT_VARIABLE: list(treatments.keys())},
    control=CONTROL_ARM,
    get_data=get_data,
    model=fixed_effects_regression,
)


@User.route("/survey")
def seed() -> list[Page]:
    """Creates the main survey branch.

    Returns:
        List[Page]: List of pages shown to the user.
    """

    if os.getenv("FLASK_ENV") == "production":
        start_time_et = get_today()
    else:
        start_time_et = (
            questions[0]["get_outcome"]()[START_TIME_VARIABLE].sample().iloc[0]
        )
    current_user.meta_data[START_TIME_VARIABLE] = start_time_et

    treatment = treatments[assigner.assign_user()[TREATMENT_VARIABLE]]
    questions_copy = questions.copy()
    shuffle(questions_copy)
    current_user.data = [
        (QUESTION_VARIABLE, [q["name"] for q in questions_copy]),
        (OUTCOME_VARIABLE, len(questions) * [None]),
        (TARGET, len(questions) * [None]),
    ]
    forecast_pages = treatment["make_pages"](questions_copy)
    return forecast_pages + [
        Page(
            Label("Thanks for participating!"),
        )
    ]
