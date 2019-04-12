import glob
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sender = "chessVisualiser@gmail.com"
password = "DT228ComputerScience"

"""
receiver = "c15345046@mydit.ie"
which = "welcome"
replace = {'username': 'DefinitePurple'}
sendEmail(which, receiver, replace)
"""


def register(html, text, replace):
    return html.format(username=replace['username']), text.format(username=replace['username'])


def upload(html, text, replace):
    return html.format(username=replace['username']), text.format(username=replace['username'])


def processed(html, text, replace):
    return html.format(username=replace['username']), text.format(username=replace['username'])


def subject(which):
    if which is 'register':
        return 'Welcome to ChessVisualiser'
    elif which is 'upload':
        return 'Be done soon!'
    elif which is 'processed':
        return 'We\'re done!'


def getFileContents(which):
    parentPath = os.path.dirname(os.path.realpath(__file__))
    emailPath = os.path.join(parentPath, 'static')
    emailPath = os.path.join(emailPath, 'emails')
    emailPath = os.path.join(emailPath, which)

    files = [f for f in glob.glob(emailPath + "**/*", recursive=True)]

    if '.html' in files[0]:
        htmlPath = files[0]
        textPath = files[1]
    else:
        htmlPath = files[1]
        textPath = files[0]

    with open(htmlPath, 'r') as htmlFile:
        html = htmlFile.read()

    with open(textPath, 'r') as textFile:
        text = textFile.read()

    return html, text


def sendEmail(which, receiver, replace):

    message = MIMEMultipart("alternative")
    message["Subject"] = subject(which)
    message["From"] = sender
    message["To"] = receiver

    # Create the plain-text and HTML version of your message
    html, text = getFileContents(which)

    html, text = globals()[which](html, text, replace)

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)
    message.attach(part2)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, message.as_string())
