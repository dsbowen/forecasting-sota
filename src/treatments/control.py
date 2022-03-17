"""Defines the control arm.
"""
from __future__ import annotations

from hemlock import Page
from hemlock.questions import Input, Label

from ..questions import QuestionType
from ..scoring import convert_to_nonparametric_elicitation

question_info = {
    "crude": (
        "the price of crude oil",
        "the price of crude oil will be less than $_____ at the end of the day",
    ),
    "s_and_p": (
        "the S&P 500 index",
        "the S&P 500 index will be less than _____ at the end of the day",
    ),
}


def make_forecast_pages(questions: list[QuestionType]) -> list[Page]:
    """Create forecasting pages.

    Args:
        questions (list[QuestionType]): Questions to forecast.

    Returns:
        list[Page]: Pages on which users make their forecasts.
    """
    return [make_forecast_page(q) for q in questions]


def make_forecast_page(question: QuestionType) -> Page:
    """Create a forecasting page.

    Args:
        question (QuestionType): Question to forecast.

    Returns:
        Page: Page on which the user makes his forecast.
    """
    info = question_info[question["name"]]
    return Page(
        Label(
            f"""
            We will now ask you some questions about {info[0]}.

            You can find some information <a href="{question["url"]}" target="_blank">here</a>.
            """
        ),
        *[
            Input(
                f"I think there is a {int(100*percent)}% chance that {info[1]}.",
                input_tag={"type": "number", "step": "any", "required": True},
                variable=f"ppf_{percent}",
            )
            for percent in (0, 0.25, 0.5, 0.75, 1)
        ],
    )


arm = {
    "make_pages": make_forecast_pages,
    "convert_to_distribution": convert_to_nonparametric_elicitation,
}
