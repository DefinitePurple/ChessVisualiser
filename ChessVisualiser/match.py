import os
import threading

from flask import (Blueprint, g, render_template, request, flash, redirect, url_for, send_from_directory)

import ChessVisualiser.db_match as match_db
import ChessVisualiser.service_match as match_service
from ChessVisualiser.auth import login_required

bp = Blueprint('match', __name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@bp.route('/history')
@login_required
def history():
    resp = match_db.getMatchesByUser(g.user.id)
    uploads = []
    for upload in resp:
        uploads.append({
            'id': upload.id,
            'white': upload.white,
            'black': upload.black,
            'score': upload.score,
            'date': upload.date
        })
    data = list(chunks(uploads, 6))

    return render_template('match/history.html', data=data)


@bp.route('/download')
@login_required
def download():
    mid = request.args['mid']
    path = os.path.join(APP_ROOT, 'static')
    path = os.path.join(path, 'users')
    path = os.path.join(path, str(g.user.id))
    path = os.path.join(path, mid)
    check = os.path.join(path, 'processed.mp4')
    if os.path.exists(check):
        return send_from_directory(path, filename='processed.mp4', mimetype='video/mp4', as_attachment=True)

    flash('Images could not be found')
    return redirect(url_for('match.history'))


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
            if name[len(name) - 4:] != '.mp4':
                flash(u'Incorrect file format. Convert video to .mp4')
            else:
                # Setup the following directories:
                #   ChessVisualiser/static/users/[user id]
                #   ChessVisualiser/static/users/[user id]/videos
                #   ChessVisualiser/static/users/[user id]/images/[match id]
                #   ChessVisualiser/static/users/[user id]/videos/[match id]

                userDir = os.path.join(APP_ROOT, 'static')
                userDir = os.path.join(userDir, 'users')
                target = os.path.join(userDir, str(g.user.id))

                data = {
                    "path": userDir,
                    "white": request.form['white'],
                    "black": request.form['black'],
                    "score": request.form['score'],
                    "userId": g.user.id
                }

                # Insert match into the database
                matchId = match_db.initMatch(data)

                # ChessVisualiser/static/users/[user id]/[match id]
                target = os.path.join(target, str(matchId))

                # Create the directories if they don't exist
                # Creates all parent directories too
                # EG: A/B/C/D
                # If We want to create D and it B doesn't exist, I will create B, C, and D

                if not os.path.exists(target):
                    os.makedirs(target)

                # Create a new file name with the datetime.mp4

                filename = 'input.mp4'
                destination = "\\".join([target, filename])
                file.save(destination)

                thread = threading.Thread(target=match_service.startMatchProcessing,
                                          kwargs={'path': target,
                                                  'userId': g.user.id})
                thread.start()

                flash(u'Video has been uploaded. You will be emailed when the processing has completed', 'error')

        return render_template('match/upload.html')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
