from .db_setup import Session
from .models import Match
import datetime

def initMatch(data):
    now = datetime.datetime.now()
    date = "{}-{}-{}".format(now.day, now.month, now.year)

    match = Match(white=data['white'],
                  black=data['black'],
                  score=data['score'],
                  user_id=data['userId'],
                  date=date)

    Session.add(match)
    Session.commit()
    Session.flush()

    return match.id


def getMatchesByUser(account):
    query = Session.query(Match).filter(Match.user_id == account)
    return query.all()



