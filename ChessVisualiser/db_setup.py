import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

db_string = "postgres://Daniel:password@127.0.0.7:5432/chess"
db = create_engine(db_string)

# print(db.table_names())
Session = scoped_session(sessionmaker(bind=db))



def cleanup(obj):
    Session.remove()
