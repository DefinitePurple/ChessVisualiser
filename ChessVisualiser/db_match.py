from sqlalchemy import func
from .db_setup import Session
from .models import Match


def initMatch(data):
    print(data)

    match = Match(file=data['path'],
                  white=data['white'],
                  black=data['black'],
                  url=data['host'],
                  score=data['score'],
                  user_id=data['userId'])
    Session.add(match)
    Session.commit()
    Session.flush()

    return match.id


