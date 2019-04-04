import ChessVisualiser.db_user as db_user
import ChessVisualiser.emailHandler as emailer
import ChessVisualiser.processing as processing


def startMatchProcessing(matchId, path, userId):
    user = db_user.getUserBy({'id': userId})
    emailer.sendEmail('upload', user.email, {'username': user.username})
    # Start processing the video

    processing.beginVideoProcessing(path, {'email': user.email, 'username': user.username})
