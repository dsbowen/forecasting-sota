"""Main survey file.
"""
import os

import numpy as np
from flask_login import current_user
from hemlock import User, Page, utils
from hemlock.functional import compile, validate, test_response
from hemlock.questions import Check, Input, Label, Range, Select, Textarea
from hemlock_ax import Assigner
from sqlalchemy_mutable.utils import partial

assigner = Assigner({"factor0": (0, 1), "factor1": (0, 1, 2)})


@User.route("/survey")
def seed():
    """Creates the main survey branch.

    Returns:
        List[Page]: List of pages shown to the user.
    """
    assignment = assigner.assign_user()
    return [
        Page(
            Input(
                f"You were assigned to {assignment}",
                input_tag={"type": "number"},
                variable="target"
            )
        ),
        Page(
            Label("Thanks for participating!"),
        )
    ]
