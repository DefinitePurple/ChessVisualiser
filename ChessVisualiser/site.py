from flask import (Blueprint, flash, g, redirect, render_template, request, url_for)
import json
from werkzeug.exceptions import abort

from ChessVisualiser.auth import login_required
from ChessVisualiser.db import get_db

bp = Blueprint('site', __name__)


@bp.route('/')
@login_required
def index():
    print('hello')

    moveList = ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5']
    return render_template('site/index.html', moves=json.dumps(moveList))
