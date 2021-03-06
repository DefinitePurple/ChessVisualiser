from flask import (Blueprint, redirect, url_for)

from ChessVisualiser.auth import login_required

bp = Blueprint('site', __name__)


@bp.route('/')
@login_required
def index():
    """
    Index route - localhost/
    redirects to localhost/match/upload
    """
    return redirect(url_for('match.upload'))
