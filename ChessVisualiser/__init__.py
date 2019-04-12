import os

from flask import Flask, request


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'ChessVisualiser.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from .db_setup import cleanup
    app.teardown_appcontext(cleanup)

    from . import auth
    app.register_blueprint(auth.bp, url_prefix='/auth/')

    from . import match
    app.register_blueprint(match.bp, url_prefix='/match/')

    from . import site
    app.register_blueprint(site.bp)
    app.add_url_rule('/', endpoint='index')

    return app
