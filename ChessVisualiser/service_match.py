import ChessVisualiser.db_user as db_user
import ChessVisualiser.email_handler as emailer
import ChessVisualiser.processing as processing

"""
Service layers are used for any processing that needs a thread to run
Only processing of uploaded videos currently needs this
"""


def startMatchProcessing(path, userId):
    """
    Get the user from the database by their id
    Send an email to the the user stating their upload is beginning processing
    Begin processing
    Send an email stating processing is finished

    :param path:
    :param userId:
    :return Nothing:
    """
    user = db_user.getUserBy({'id': userId})
    emailer.sendEmail('upload', user.email, {'username': user.username})
    processing.beginVideoProcessing(path, {'email': user.email, 'username': user.username})
    emailer.sendEmail('processed', user.email, {'username': user.username})
