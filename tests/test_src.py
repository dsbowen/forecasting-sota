import pytest
from hemlock import User, create_test_app
from hemlock.app import db
from hemlock_ax import init_test_app, run_test
from hemlock_ax.assign import assigners

import src

N_USERS = 100


@pytest.fixture
def app():
    app = create_test_app()
    init_test_app(app)
    yield app
    for user in User.query.all():
        db.session.delete(user)
    db.session.commit()


def test(app):
    run_test(N_USERS)
    for assigner in assigners:
        if not assigner.weights:
            raise RuntimeError(
                "Assigner weights not set."
                "Make sure the target variable is numeric."
            )
