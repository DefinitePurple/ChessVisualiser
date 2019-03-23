import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "chessVisualiser@gmail.com"
receiver_email = "c15345046@mydit.ie"
password = "DT228ComputerScience"

message = MIMEMultipart("alternative")
message["Subject"] = "multipart test"
message["From"] = sender_email
message["To"] = receiver_email

# Create the plain-text and HTML version of your message
text = """\
Hi [Username],

Your video has finished processing.
www.chessvisualiser.com

Regards,
The ChessVisualiser Team
"""

html = """\
<html>
  <body>
    <p>Hi [Username],<br><br>
        Your video has finished processing. <br>
        <b><a href="http://www.chessvisualiser.com">Click here to view your game</a></b> <br>
        <br>
        <b>Regards,<br>
        The ChessVisualiser Team <br>
        <a href="http://www.chessvisualiser.com">Chess visualiser</a></b> <br>
    </p>
  </body>
</html>
"""

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
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

# Create an email for welcoming the user after registering
# Create an email for uploading a video
# Create an email for finished processing a video
