"""Create the application.
"""
from hemlock import create_app
from hemlock.app import socketio
from hemlock_ax import init_app, init_test_app

import src

app = create_app()

if app.config["ENV"] == "production":
    # Don't allow users to restart and block duplicate IP addresses in production
    app.config.update(ALLOW_USERS_TO_RESTART=False, BLOCK_DUPLICATE_KEYS=["ipv4"])

if __name__ == "__main__":
    init_test_app(app)
    socketio.run(app, debug=True)
else:
    init_app(app)
