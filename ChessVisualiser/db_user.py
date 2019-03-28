from sqlalchemy import func
from .db_setup import Session
from .models import Users


# By id, email, or username
def getUserBy(account):
    # {'username': 'definitepurple'}
    keys = list(account.keys())
    query = Session.query(Users)
    for key in keys:
        if key == 'email':
            query = query.filter(func.lower(Users.email) == func.lower(account[key]))
        elif key == 'username':
            query = query.filter(func.lower(Users.username) == func.lower(account[key]))
        elif key == 'id':
            query = query.filter(Users.id == account[key])

    return query.first()


def loginUser(username, password):
    # Session.execute('call login_user(:u, :p)', {'u': account, 'p': password)
    success = Session.execute(func.login_user(username, password)).fetchone()

    if success is None:
        return success
    return success[0]


def registerUser(email, username, password):
    Session.execute('call create_user(:e,:u,:p)', {'e': email, 'u': username, 'p': password})
    Session.commit()
