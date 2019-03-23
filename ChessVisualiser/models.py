from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Date, ForeignKey


Base = declarative_base()


class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String)
    username = Column(String)
    matches = relationship("Match")

    def __repr__(self):
        return "<Users(" \
               "id='%d' " \
               "email='%s' " \
               "username='%s')>" \
               % (self.id,
                  self.email,
                  self.username)


class Match(Base):
    __tablename__ = 'match'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    white = Column(String)
    black = Column(String)
    score = Column(String)
    moves = Column(String)
    url = Column(String)
    file = Column(String)
    date = Column(Date)




    def __repr__(self):
        return "<Match(" \
               "id='%d' " \
               "white='%s' " \
               "black='%s' " \
               "score='%s' " \
               "moves='%s' " \
               "date='%s')>" \
               % (self.id,
                  self.white,
                  self.black,
                  self.score,
                  self.moves,
                  str(self.date))
