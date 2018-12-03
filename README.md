# ChessVisualiser

File structure breakdown
root \n/
    | ChessVisualiser /
          | static /
                | css /
                      | chessboard.css - http://chessboardjs.com/
                      | chessboard.min.css - http://chessboardjs.com/
                      | style.css - created by Daniel Fitzpatrick
                | js /
                      | base.js - created by Daniel Fitzpatrick
                      | chessboard.js - http://chessboardjs.com/ - modified to work with flask jinja 2 syntax
                      | chessboard.min.js - http://chessboardjs.com/
                      | jquery.js - https://jquery.com/
                      | jquery.min.js - https://jquery.com/
                      | match.js - created by Daniel Fitzpatrick
                      | matches.js - created by Daniel Fitzpatrick
          | templates /
                | auth /
                      | login.html - created by Daniel Fitzpatrick
                      | register.html - created by Daniel Fitzpatrick
                | match /
                      | match.html - created by Daniel Fitzpatrick
                      | matches.html - created by Daniel Fitzpatrick
                      | upload.html - created by Daniel Fitzpatrick
                | base.html - created by Daniel Fitzpatrick
          | \_\_init\_\_.py  - created by Daniel Fitzpatrick
          | auth.py - created by Daniel Fitzpatrick
          | db.py - created by Daniel Fitzpatrick
          | match.py - created by Daniel Fitzpatrick
          | schema.sql - created by Daniel Fitzpatrick
          | site.py - created by Daniel Fitzpatrick
    | run.bat - created by Daniel Fitzpatrick
    | run.sh - created by Daniel Fitzpatrick
