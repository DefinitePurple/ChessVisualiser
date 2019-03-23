import ChessVisualiser.db_match as db_match


def initMatch(path, host, white, black, score, userId):
    data = {
        "path": path,
        "host": host,
        "white": white,
        "black": black,
        "score": score,
        "userId": userId
    }
    matchId = db_match.initMatch(data)
    # Start processing the video