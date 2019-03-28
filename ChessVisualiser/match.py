import os, errno
import json
import threading
import datetime
import ChessVisualiser.service_match as match_service

from flask import (Blueprint, g, render_template, request, flash)
from ChessVisualiser.auth import login_required
import ChessVisualiser.db_match as match_db

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


# @bp.route('/<mid>')
# @login_required
# def view_match(mid):
#     match = {
#         'id': mid,
#         'moves': ['c2-c4', 'e7-e5', 'b1-c3', 'g8-f6', 'g1-f3', 'b8-c6', 'g2-g3', 'f8-b4', 'f3-e5'],
#         'date': '25/12/2018'
#     }
#     return render_template('match/match.html', match=json.dumps(match))


@bp.route('/uploads')
@login_required
def uploads():
    resp = match_db.getMatchesByUser(g.user.id)
    data = []
    for upload in resp:
        data.append({
            'id': upload.id,
            'white': upload.white,
            'black': upload.black,
            'score': upload.score,
            'date': upload.date
        })

    return render_template('match/uploads.html', data=json.dumps(data))


@bp.route('/upload/<mid>')
@login_required
def view_upload():
    print(mid.mid)
    files = os.listdir('./ChessVisualiser/static/users/' + str(g.user.id) + '/images/' + mid)
    return render_template('match/uploaded.html', files=files)


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
                # Setup the following directories:
                #   ChessVisualiser/static/users/[user id]
                #   ChessVisualiser/static/users/[user id]/videos
                #   ChessVisualiser/static/users/[user id]/images/[match id]
                #   ChessVisualiser/static/users/[user id]/videos/[match id]

                userDir = os.path.join(APP_ROOT, 'static')
                userDir = os.path.join(userDir, 'users')
                userDir = os.path.join(userDir, str(g.user.id))
                target = os.path.join(userDir, 'videos')
                userImages = os.path.join(userDir, 'images')

                data = {
                    "path": userDir,
                    "host": request.host_url,
                    "white": request.form['white'],
                    "black": request.form['black'],
                    "score": request.form['score'],
                    "userId": g.user.id
                }

                # Insert match into the database
                matchId = match_db.initMatch(data)

                # ChessVisualiser/static/users/[user id]/videos/[match id]
                target = os.path.join(target, str(matchId))
                userImages = os.path.join(userImages, str(matchId))

                # Create the directories if they don't exist
                # Creates all parent directories too
                # EG: A/B/C/D
                # If We want to create D and it B doesn't exist, I will create B, C, and D
                """ Swap to this to avoid race conditions. 
                Commented out until figured out what to do with logs
                try:
                    os.makedirs(directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                """
                if not os.path.exists(userImages):
                    os.makedirs(userImages)

                if not os.path.exists(target):
                    os.makedirs(target)

                # Create a new file name with the datetime.mp4
                date = str(datetime.datetime.now())
                date = date.split('.')[0]
                date = ''.join(date.split('-'))
                date = ''.join(date.split(':'))
                filename = ''.join(date.split(' ')) + '.mp4'
                destination = "\\".join([target, filename])
                file.save(destination)

                thread = threading.Thread(target=match_service.startMatchProcessing,
                                          kwargs={'matchId': matchId,
                                                  'videoPath': destination,
                                                  'imagePath': userImages,
                                                  'userId': g.user.id})
                thread.start()

                flash(u'Video has been uploaded. You will be emailed when the processing has completed', 'error')

        return render_template('match/upload.html')
