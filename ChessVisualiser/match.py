import os
from flask import (Blueprint, g, render_template, request, flash)
import json
import ChessVisualiser.service_match as vid
import threading

from ChessVisualiser.auth import login_required

bp = Blueprint('match', __name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@bp.route('/user/<username>')
@login_required
def view_matches(username):
    data = []
    game = {
        'id': 1,
        'moves': ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5'],
        'date': '25/12/2018'
    }
    data.append(game)
    game = {
        'id': 2,
        'moves': ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5'],
        'date': '25/12/2018'
    }
    data.append(game)
    game = {
        'id': 3,
        'moves': ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5'],
        'date': '25/12/2018'
    }
    data.append(game)
    return render_template('match/matches.html', matches=json.dumps(data))


@bp.route('/<mid>')
@login_required
def view_match(mid):
    match = {
        'id': mid,
        'moves': ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5'],
        'date': '25/12/2018'
    }
    return render_template('match/match.html', match=json.dumps(match))


@bp.route('/upload', methods=('GET', 'POST'))
@login_required
def upload():
    if request.method == 'GET':
        return render_template('match/upload.html')
    elif request.method == 'POST':
        try:
            file = request.files['match']
        except:
            file = None

        if file is None:
            flash(u'File is corrupt')
        else:
            name = file.filename
            if name[len(name)-4:] != '.mp4':
                flash(u'Incorrect file format. Convert video to .mp4')
            else:
                target = os.path.join(APP_ROOT, 'videos')
                destination = "\\".join([target, file.filename])
                file.save(destination)

                # Now we want to insert to initialise the match in the database

                data = {
                    "path": destination,
                    "host": request.host_url,
                    "white": request.form['white'],
                    "black": request.form['black'],
                    "score": request.form['score'],
                    "userId": g.user.id
                }
                thread = threading.Thread(target=vid.initMatch, kwargs=data)
                thread.start()

                flash(u'Video has been uploaded. You will be emailed when the processing has completed', 'error')

        return render_template('match/upload.html')
