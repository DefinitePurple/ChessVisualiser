import functools
import re
import string
import random
from datetime import datetime

from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for)
from sqlalchemy import create_engine
from werkzeug.security import check_password_hash
import ChessVisualiser.db_auth as db

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        error = None

        if not username:
            error = 'Username is required'
        elif not password:
            error = 'Password is required'
        elif not email:
            error = 'Email is required'
        elif db.getUserBy({'username': username}) is not None:
            error = 'User {} already exists'.format(username)
        elif db.getUserBy({'email': email}) is not None:
            error = 'Email {} is already in use'.format(email)

        if error is None:
            db.registerUser(email, username, password)
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        success = db.loginUser(username, password)

        if success is None:
            flash(u'User does not exist', 'error')
        elif success is True:
            user = db.getUserBy({'username': username})
            if user is not None:
                session.clear()
                session['user_id'] = user.id
            else:
                flash(u'Error finding user', 'error')
        else:
            flash(u'Invalid Username or Password', 'error')
        return redirect(url_for('index'))

    return render_template('auth/login.html')


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = db.getUserBy({'id': user_id})


@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
